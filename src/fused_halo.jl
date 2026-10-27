# Fused halo updates (opt-in, environment variable IGG_FUSED, read by init_global_grid; CUDA arrays only so far).
#
# update_halo! packs and unpacks each side of each dimension of each field with its own kernel launch or copy, each
# waited for on the host, one dimension after the other. The fused path packs all the regions of a round of messages
# into one flat buffer with one kernel, waits once, sends one message per neighbour, and unpacks with one kernel on
# the caller's stream, without waiting: later work on that stream is ordered after it. It caches everything that does
# not depend on the arrays themselves (regions, buffers, kernels, MPI buffers and requests) per call signature.
#
#   IGG_FUSED=1  one round per dimension, in the order of `dims`: the same messages as update_halo!, the same result
#   IGG_FUSED=2  one round for all dimensions: the edges and corners come straight from the diagonal neighbours,
#                the same result as one round per dimension (the ghosts of a side without neighbour travel with the
#                messages of the other dimensions, as they do when the dimensions go one after the other)
#
# update_halo! then returns before the halos are unpacked: use the arrays on the task's stream (as CUDA.jl does by
# default), or synchronize it, before reading them from the host or from another stream.
#
# update_halo_start! and update_halo_finish! split a fused update after the messages of its last round are posted,
# so that work that does not read the halos runs while they travel.

const FUSED = Ref(0)
const HALO_PLANS = Dict{Any,Any}()  # call signature => plan of the GPU extension (nothing: not supported, use update_halo!)
const FUSED_TAG = 2^14               # message tags: FUSED_TAG + round
const HALO_STARTED = Ref(false)      # an update_halo_start! waits for its update_halo_finish!

function init_fused()
    FUSED[] = parse(Int, get(ENV, "IGG_FUSED", "0"))
    FUSED[] in (0, 1, 2) || error("IGG_FUSED must be 0, 1 or 2, got $(FUSED[])")
    return nothing
end

# Handled by a GPU extension for its arrays: return true if it updated the halos of A; for the start, the HaloUpdate
# to finish, or nothing if update_halo! has to do it.
fused_halo!(A::Tuple, dims) = false
fused_halo_start!(A::Tuple, dims) = nothing
function fused_halo_finish! end
free_plan!(plan) = nothing
free_halo_plans!() = (foreach(free_plan!, values(HALO_PLANS)); empty!(HALO_PLANS); HALO_STARTED[] = false; nothing)

# A fused update started by update_halo_start!: the plan of the GPU extension and the arrays
struct HaloUpdate{P,T<:Tuple}
    plan::P
    A::T
end

# A box of one field, copied to or from a flat buffer (also read by the kernels: isbits)
struct HaloRegion
    field::Int32
    first::NTuple{3,Int32}
    size::NTuple{3,Int32}
    offset::Int32           # of its first element in the flat buffer, from 0
end
HaloRegion(field, ranges, offset) = HaloRegion(field, Int32.(first.(ranges)), Int32.(length.(ranges)), offset)
nelems(r::HaloRegion) = prod(Int.(r.size))

struct HaloMessage
    peer::Int     # rank in the grid's communicator
    offset::Int   # of its first element in the flat buffer, from 0
    count::Int
end

struct HaloRound
    send::Vector{HaloRegion}
    recv::Vector{HaloRegion}
    sendmsgs::Vector{HaloMessage}
    recvmsgs::Vector{HaloMessage}
    sendlen::Int
    recvlen::Int
end

diridx(d) = (d[1] + 1) + 3(d[2] + 1) + 9(d[3] + 1)

# The rank at direction d (each component -1, 0 or 1) in the active grid, or nothing
function neighbor_at(d)
    gg = global_grid()
    c = copy(gg.coords)
    for k in 1:3
        c[k] += d[k]
        if !(0 <= c[k] < gg.dims[k])
            gg.periods[k] == 0 && return nothing
            c[k] = mod(c[k], gg.dims[k])
        end
    end
    return MPI.Cart_rank(gg.comm, c)
end

# The ranges of field A in dimension k for a message component d_k: the halowidth layer sent to (received from) the
# neighbour on that side, or for d_k = 0 the cells the message carries across: all of them when the dimensions go one
# after the other (schedule 1), else all but the ghosts that a neighbour in k fills (schedule 2).
function halo_range(A, k, dk, send::Bool, schedule::Integer, exchanged::Bool)
    h, n = global_grid().halowidths[k], size(A, k)
    if dk == 0
        (schedule == 1 || !exchanged) && return 1:n
        return (has_neighbor(1, k) ? h + 1 : 1):(has_neighbor(2, k) ? n - h : n)
    end
    o = ol(k, A)
    i = send ? (dk < 0 ? 1 + (o - h) : n - (o - 1)) : (dk < 0 ? 1 : n - h + 1)
    return i:i+h-1
end

# The rounds of messages of an update of the halos of the arrays A along dims, or nothing if not supported
function halo_rounds(A::Tuple, dims, schedule::Integer)
    gg = global_grid()
    all(ndims(a) == 3 for a in A) || return nothing
    for k in dims
        any(n -> neighbor(n, k) == me(), 1:NNEIGHBORS_PER_DIM) && return nothing  # periodic over one rank: update_halo! copies locally
    end
    # the dimensions in which a field has a halo (an overlap of at least twice the halowidth, as in update_halo!)
    exchanged(i, k) = k in dims && ol(k, A[i]) >= 2gg.halowidths[k]
    unit(k, s) = ntuple(j -> j == k ? s : 0, 3)
    dirsets = schedule == 1 ? [[unit(k, -1), unit(k, 1)] for k in dims] :
              [[d for d in Iterators.product(-1:1, -1:1, -1:1) if any(!=(0), d) && all(k -> d[k] == 0 || k in dims, 1:3)]]
    rounds = HaloRound[]
    for dirs in dirsets
        # send: to each neighbour, the regions of all directions that lead to it, by direction then field;
        # receive: from each neighbour, the regions in the order it sends them (by the direction from it to us)
        out = Dict{Int,Vector{Tuple{Int,Int,Any}}}(); inc = Dict{Int,Vector{Tuple{Int,Int,Any}}}()
        for d in dirs, i in eachindex(A)
            all(k -> d[k] == 0 || exchanged(i, k), 1:3) || continue
            peer = neighbor_at(d)
            peer === nothing && continue
            sr = ntuple(k -> halo_range(A[i], k, d[k], true, schedule, exchanged(i, k)), 3)
            rr = ntuple(k -> halo_range(A[i], k, d[k], false, schedule, exchanged(i, k)), 3)
            push!(get!(out, peer, []), (diridx(d), i, sr))
            push!(get!(inc, peer, []), (diridx(.-d), i, rr))
        end
        regions(dict) = begin
            regs, msgs, off = HaloRegion[], HaloMessage[], 0
            for peer in sort!(collect(keys(dict)))
                start = off
                for (_, i, r) in sort!(dict[peer], by=x -> (x[1], x[2]))
                    push!(regs, HaloRegion(i, r, off)); off += prod(length.(r))
                end
                push!(msgs, HaloMessage(peer, start, off - start))
            end
            regs, msgs, off
        end
        send, sendmsgs, sendlen = regions(out)
        recv, recvmsgs, recvlen = regions(inc)
        isempty(sendmsgs) && isempty(recvmsgs) && continue
        push!(rounds, HaloRound(send, recv, sendmsgs, recvmsgs, sendlen, recvlen))
    end
    return rounds
end

halo_plan_key(A::Tuple, dims) = (global_grid().comm.val, Tuple(global_grid().nxyz), Tuple(global_grid().coords),
                                 dims, FUSED[], typeof(A), map(size, A))
