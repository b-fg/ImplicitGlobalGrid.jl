# The fused halo updates (IGG_FUSED=1 and 2, see src/fused_halo.jl), also split by update_halo_start! and
# update_halo_finish!, give the same arrays as update_halo!, bit for bit, over many process layouts, periodicities,
# overlaps, halowidths, dimension subsets and fields of different sizes.
# NOTE: run with several processes (e.g. 2, 4 and 8): with one, every periodic neighbour is the process itself and
# the fused update leaves it to update_halo!.

using Test
import MPI
using CUDA
using ImplicitGlobalGrid; GG = ImplicitGlobalGrid

test_cuda = CUDA.functional()

MPI.Init()
nprocs = MPI.Comm_size(MPI.COMM_WORLD)
me = MPI.Comm_rank(MPI.COMM_WORLD)

# Process layouts (0 lets init_global_grid choose) for this number of processes
layouts = unique([(0, 0, 0); filter(d -> all(>(0), d) && prod(d) == nprocs,
                                    [(nprocs, 1, 1), (1, nprocs, 1), (1, 1, nprocs), (2, nprocs ÷ 2, 1), (1, 2, nprocs ÷ 2)])])
periodicities = [(1, 1, 1), (0, 0, 0), (1, 0, 1), (0, 1, 0)]
halos = [((2, 2, 2), (1, 1, 1)), ((4, 4, 4), (2, 2, 2)), ((3, 3, 3), (1, 1, 1)), ((2, 4, 2), (1, 2, 1))]

function check(nxyz, layout, period, overlaps, halowidths)
    init_global_grid(nxyz...; dimx=layout[1], dimy=layout[2], dimz=layout[3], periodx=period[1], periody=period[2],
                     periodz=period[3], overlaps=overlaps, halowidths=halowidths, init_MPI=false, select_device=false, quiet=true)
    nxyz_l = Tuple(GG.global_grid().nxyz)
    splitdims = Tuple(k for k in 1:3 if GG.global_grid().dims[k] > 1)
    ok = true
    # one field; three fields of the grid's size (a velocity); three of different sizes (staggered, one without halo in x)
    for sizes in ([nxyz_l], fill(nxyz_l, 3), [nxyz_l, nxyz_l .+ (1, 0, 0), nxyz_l .- (overlaps[1] - 1, 0, 0)])
        for dims in unique([(1, 2, 3), (3, 1, 2), splitdims, (1, 3)])
            A0 = [CUDA.rand(Float32, s...) for s in sizes]
            # update_halo! with each mode, then update_halo_start! and update_halo_finish! with each mode
            res = map(Iterators.product(0:2, (false, true))) do (mode, halves)
                GG.FUSED[] = mode
                A = copy.(A0)
                CUDA.synchronize()  # update_halo! packs on its own streams, which do not wait for the copies
                for _ in 1:2  # twice: the second call runs the cached plan
                    halves ? update_halo_finish!(update_halo_start!(A...; dims=dims)) : update_halo!(A...; dims=dims)
                end
                CUDA.synchronize()
                Array.(A)
            end
            GG.FUSED[] = 0
            ok &= all(==(res[1]), res)
        end
    end
    finalize_global_grid(finalize_MPI=false)
    return ok
end

@testset "fused halo updates, $nprocs process(es)" begin
    if !test_cuda
        @test_skip false
    else
        for layout in layouts, period in periodicities, (overlaps, halowidths) in halos
            @test check((14, 11, 9), layout, period, overlaps, halowidths)
        end
    end
end

MPI.Finalize()
