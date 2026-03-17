# repro https://github.com/PRONTOLab/GB-25/actions/runs/23031192331/job/66889719360:
# XLA_FLAGS='--xla_force_host_platform_device_count=2' julia --project -O0 sharding/sharded_baroclinic_instability_simulation_run.jl

using Dates
@info "This is when the fun begins" now(UTC)

ENV["JULIA_DEBUG"] = "Reactant_jll,Reactant"

using Oceananigans

Oceananigans.defaults.FloatType = Float64

using Oceananigans.Units
using Oceananigans.Architectures: ReactantState
using Random
using Printf
using CUDA
using Reactant
using SeawaterPolynomials

get_jobid_procid() =
    string(
        get(ENV, "SLURM_JOB_ID", replace(string(now(UTC)), ':' => '-')),
        ".",
        get(ENV, "SLURM_PROCID", string(getpid()))
    )

function loop!(model, Ninner)
    Reactant.Profiler.annotate("loop") do
        Δt = model.clock.last_Δt + 0
        @trace track_numbers=false for _ = 1:Ninner
            Oceananigans.TimeSteppers.time_step!(model, Δt)
        end
    end
    return nothing
end

function preamble(; rendezvous_warn::Union{Nothing,Int}=nothing, rendezvous_terminate::Union{Nothing,Int}=nothing)
    # If we are in GitHub Actions, make `TMPDIR` be a local directory from which we
    # can upload artifacts at the end.
    if get(ENV, "GITHUB_ACTIONS", "false") == "true"
        Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
        ENV["TMPDIR"] = mkpath(joinpath(@__DIR__, "..", "tmp"))
    end

    # Unset environment variables which would cause XLA distributed to hang indefinitely.
    for key in ("no_proxy", "http_proxy", "https_proxy", "NO_PROXY", "HTTP_PROXY", "HTTPS_PROXY")
        delete!(ENV, key)
    end

    if rendezvous_warn isa Int || rendezvous_terminate isa Int
        error("""
              Setting rendezvous timeouts in `preamble` is not supported anymore.
              Use `XLA_FLAGS` instead, e.g.
                  XLA_FLAGS="--xla_gpu_first_collective_call_warn_stuck_timeout_seconds=40 --xla_gpu_first_collective_call_terminate_timeout_seconds=80"
              """)
    end
end

function initialize(; kwargs...)
    # TODO: improve the condition by checking the device we're on?
    if !(get(ENV, "CI", "false") == "true" || contains(get(ENV, "XLA_FLAGS", ""), "--xla_force_host_platform_device_count"))
        Reactant.Distributed.initialize(; kwargs...)
    end
end

# hack from codex to get it to not see the gpus
let state = getfield(Reactant.XLA, :global_backend_state)
    cpu_client = Reactant.Accelerators.CPU.make_ifrt_client()
    getfield(state, :clients)["cpu"] = cpu_client
    setfield!(state, :default_client, cpu_client)
    setfield!(state, :initialized, true)
end

function simple_latitude_longitude_grid(arch, Nx, Ny, Nz; halo=(8, 8, 8))
    z = ExponentialDiscretization(Nz, -4000, 0; scale=30) # may need changing for very large Nz

    grid = LatitudeLongitudeGrid(arch; size=(Nx, Ny, Nz), halo, z,
        latitude = (-80, 80),
        longitude = (0, 360)
    )

    return grid
end

function baroclinic_instability_model(arch, Nx, Ny, Nz; Δt,
    halo = (8, 8, 8),
    grid_type = :simple_lat_lon, # :gaussian_islands

    # Fewer substeps can be used at higher resolutions
    free_surface = SplitExplicitFreeSurface(substeps=30),

    # TEOS10 is a 54-term polynomial that relates temperature (T),
    # and salinity (S) to buoyancy
    buoyancy = SeawaterBuoyancy(
        equation_of_state = SeawaterPolynomials.TEOS10EquationOfState(Oceananigans.defaults.FloatType)),

    closure = nothing,    
    # closure = Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivity(),
    # closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), κ=1e-5, ν=1e-4),

    # Coriolis forces for a rotating Earth
    coriolis = HydrostaticSphericalCoriolis(),

    # Use the simplest timestepper
    timestepper = :QuasiAdamsBashforth2,

    # Simple momentum advection schemes. May need to be reconsidered
    # due to Float32.
    momentum_advection = WENOVectorInvariant(order=5),
    tracer_advection = WENO(order=5),
    )

    tracers = if buoyancy isa BuoyancyTracer
        (:b,)
    elseif buoyancy isa SeawaterBuoyancy
        (:T, :S)
    else
        ()
    end

    grid = if grid_type === :gaussian_islands
        gaussian_islands_tripolar_grid(arch, Nx, Ny, Nz; halo)
    elseif grid_type === :simple_lat_lon
        simple_latitude_longitude_grid(arch, Nx, Ny, Nz; halo)
    else
        error("grid_type=$grid_type must be :gaussian_islands or :simple_lat_lon.")
    end

    model = HydrostaticFreeSurfaceModel(grid;
        free_surface, closure, buoyancy, tracers, timestepper,
        coriolis, momentum_advection, tracer_advection,
    )

    Random.seed!(42)

    #=
    if buoyancy isa SeawaterBuoyancy
        set_baroclinic_instability!(model)
    elseif buoyancy isa BuoyancyTracer
        # set!(model, b=initial_buoyancy)
    end
    =#

    model.clock.last_Δt = Δt

    return model
end
jobid_procid = get_jobid_procid()

preamble()

using Libdl: dllist
@show filter(contains("nccl"), dllist())

Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = false # maybe makes it run faster?
Reactant.MLIR.IR.DUMP_MLIR_DIR[] = joinpath(@__DIR__, "mlir_dumps", jobid_procid)
Reactant.Compiler.DEBUG_DISABLE_RESHARDING[] = true
# Reactant.Compiler.DEBUG_PRINT_CODEGEN[] = true
Reactant.Compiler.WHILE_CONCAT[] = true
# Reactant.Compiler.DUS_TO_CONCAT[] = false
# Reactant.Compiler.SUM_TO_REDUCEWINDOW[] = true
# Reactant.Compiler.AGGRESSIVE_SUM_TO_CONV[] = true

initialize(; single_gpu_per_process=false)

local_arch = Oceananigans.ReactantState()
arch = local_arch

Ndev = length(Reactant.devices())
@show Ndev

Rx, Ry = 1, 2

arch = Oceananigans.Distributed(arch; partition = Partition(Rx, Ry, 1))
rank = Reactant.Distributed.local_rank()

H = 4
Tx = 16 * Rx
Ty = 16 * Ry
Nz = 1 

Nx = Tx - 2H
Ny = Ty - 2H

@info "[$rank] Generating model (Nx=$Nx, Ny=$Ny)..." now(UTC)
model = baroclinic_instability_model(arch, Nx, Ny, Nz; halo=(H, H, H), Δt=1)

@show model

sharding = Sharding.NamedSharding(arch.connectivity, ())
Ninner = ConcreteRNumber(256; sharding)

compile_options = CompileOptions(; sync=true, raise=true, strip_llvm_debuginfo=true, strip=["enzymexla.kernel_call", "(::Reactant.Compiler.LLVMFunc", "ka_with_reactant", "(::KernelAbstractions.Kernel", "var\"#_launch!;_launch!"])

@info "[$rank] Compiling loop..." now(UTC)

@compile compile_options=compile_options loop!(model, Ninner)
