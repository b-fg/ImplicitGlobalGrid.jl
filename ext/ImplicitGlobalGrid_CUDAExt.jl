module ImplicitGlobalGrid_CUDAExt
    include(joinpath(@__DIR__, "..", "src", "CUDAExt", "shared.jl"))
    include(joinpath(@__DIR__, "..", "src", "CUDAExt", "select_device.jl"))
    include(joinpath(@__DIR__, "..", "src", "CUDAExt", "update_halo.jl"))
    include(joinpath(@__DIR__, "..", "src", "CUDAExt", "fused_halo.jl"))
end