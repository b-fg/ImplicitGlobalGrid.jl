import ImplicitGlobalGrid: HaloRegion, HaloRound, HaloUpdate, HALO_PLANS, FUSED, FUSED_TAG, halo_rounds, halo_plan_key, global_grid,
                           TRACE, trace_setup!, trace_lap!, trace_stamp!, TQ_CHECKS, TQ_BUFFERS, TQ_PACK, TQ_WAIT_PACK,
                           TQ_IRECV, TQ_ISEND, TQ_WAIT_RECV, TQ_UNPACK, TQ_WAITALL_SEND, TQ_T_FIRST_ISEND, TQ_T_RECV_DONE
import MPI

# The fused halo update of CUDA arrays (see src/fused_halo.jl): per round, one flat send and one flat receive buffer
# (page-locked host memory that the kernels access through its device mapping, or device memory with CUDA-aware MPI),
# the regions on the device, and the MPI buffers and requests of its messages.

struct CuHaloRound{T}
    round::HaloRound
    sendbuf::CuVector{T}               # what the kernels write and read
    recvbuf::CuVector{T}
    host::Vector{Any}                  # the host arrays and their registrations (nothing with CUDA-aware MPI)
    sendregs::CuVector{HaloRegion}
    recvregs::CuVector{HaloRegion}
    sendmpi::Vector{MPI.Buffer}
    recvmpi::Vector{MPI.Buffer}
    sendreqs::MPI.UnsafeMultiRequest
    recvreqs::MPI.UnsafeMultiRequest
    blocks_send::Int
    blocks_recv::Int
end

struct CuHaloPlan{T,KP,KU}
    rounds::Vector{CuHaloRound{T}}
    comm::MPI.Comm
    pack::KP      # the kernels, compiled once
    unpack::KU
end

const HALO_THREADS = 256

# The field i of a tuple through a chain of branches: indexing the tuple with a runtime index makes the kernel keep
# the tuple in local memory, which made packing 3 to 5 times slower.
@inline pick(t::Tuple{Any}, i) = t[1]
@inline pick(t::Tuple, i) = i == 1 ? t[1] : pick(Base.tail(t), i - 1)

# Copy the regions of the fields to (pack) or from the flat buffer: block row y handles region y.
function halo_copy!(buf, fields, regions, ::Val{pack}) where {pack}
    reg = @inbounds regions[blockIdx().y]
    n1, n2 = reg.size[1], reg.size[2]
    len = n1 * n2 * reg.size[3]
    A = pick(fields, reg.field)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    stride = gridDim().x * blockDim().x
    while i < len
        ix = i % n1; t = i ÷ n1; iy = t % n2; iz = t ÷ n2
        x, y, z = reg.first[1] + ix, reg.first[2] + iy, reg.first[3] + iz
        if pack
            @inbounds buf[reg.offset + i + Int32(1)] = A[x, y, z]
        else
            @inbounds A[x, y, z] = buf[reg.offset + i + Int32(1)]
        end
        i += stride
    end
    return nothing
end

nblocks(regs) = isempty(regs) ? 0 : clamp(cld(maximum(ImplicitGlobalGrid.nelems, regs), HALO_THREADS), 1, 1024)

function flatbuf(::Type{T}, len, device::Bool) where {T}
    if device
        return CuVector{T}(undef, max(len, 1)), nothing
    end
    h = zeros(T, max(len, 1))
    d, reg = ImplicitGlobalGrid.register(CuArray, h)
    return d, (h, reg)
end

mpibuf(dev::CuVector, host, m) = MPI.Buffer(host === nothing ? view(dev, m.offset+1:m.offset+m.count) : view(host[1], m.offset+1:m.offset+m.count))

function CuHaloPlan(A::Tuple{Vararg{CuArray{T,3}}}, rounds) where {T}
    aware = all(ImplicitGlobalGrid.cudaaware_MPI())
    curounds = map(rounds) do r
        sbuf, shost = flatbuf(T, r.sendlen, aware)
        rbuf, rhost = flatbuf(T, r.recvlen, aware)
        CuHaloRound{T}(r, sbuf, rbuf, Any[shost, rhost], CuArray(r.send), CuArray(r.recv),
                       [mpibuf(sbuf, shost, m) for m in r.sendmsgs], [mpibuf(rbuf, rhost, m) for m in r.recvmsgs],
                       MPI.UnsafeMultiRequest(length(r.sendmsgs)), MPI.UnsafeMultiRequest(length(r.recvmsgs)),
                       nblocks(r.send), nblocks(r.recv))
    end
    isempty(curounds) && return CuHaloPlan{T,Nothing,Nothing}(curounds, global_grid().comm, nothing, nothing)
    r = first(curounds)
    pack = @cuda launch=false halo_copy!(r.sendbuf, A, r.sendregs, Val(true))
    unpack = @cuda launch=false halo_copy!(r.recvbuf, A, r.recvregs, Val(false))
    return CuHaloPlan{T,typeof(pack),typeof(unpack)}(curounds, global_grid().comm, pack, unpack)
end

function ImplicitGlobalGrid.free_plan!(plan::CuHaloPlan)
    for r in plan.rounds, h in r.host
        h === nothing || CUDA.unregister(h[2])
    end
    return nothing
end

# Left to update_halo!, which packs on its own streams: first wait for a fused unpack that may still run on this one.
unfused() = (CUDA.synchronize(CUDA.stream(); blocking=true); false)
ImplicitGlobalGrid.fused_halo!(A::Tuple{Vararg{AnyCuArray}}, dims) = unfused()  # not 3-D, or of different types

# The cached plan of the fused update of A along dims, or nothing if update_halo! has to do it
function cached_plan(A::Tuple{Vararg{CuArray{T,3}}}, dims) where {T}
    for i in eachindex(A), j in i+1:length(A)
        A[i] === A[j] && return nothing  # update_halo! reports the duplicate
    end
    # mixed CUDA-aware and staged dimensions
    any(ImplicitGlobalGrid.cudaaware_MPI()) && !all(ImplicitGlobalGrid.cudaaware_MPI()) && return nothing
    key = halo_plan_key(A, dims)
    plan = get(HALO_PLANS, key, missing)
    if plan === missing
        # update_halo!'s argument checks depend on what the key holds (sizes, types, the grid) and duplicates
        ImplicitGlobalGrid.check_fields(ImplicitGlobalGrid.wrap_field.(A)...)
        rounds = halo_rounds(A, dims, FUSED[])
        plan = HALO_PLANS[key] = rounds === nothing ? nothing : CuHaloPlan(A, rounds)
    end
    TRACE[] && (trace_setup!(TQ_CHECKS); trace_setup!(TQ_BUFFERS))
    return plan
end

function ImplicitGlobalGrid.fused_halo!(A::Tuple{Vararg{CuArray{T,3}}}, dims) where {T}
    plan = cached_plan(A, dims)
    plan === nothing && return unfused()
    start_halo_plan!(plan, A, TRACE[])
    finish_halo_plan!(plan, A, TRACE[])
    return true
end

function ImplicitGlobalGrid.fused_halo_start!(A::Tuple{Vararg{CuArray{T,3}}}, dims) where {T}
    plan = cached_plan(A, dims)
    plan === nothing && return nothing  # update_halo! next, which waits for this stream
    start_halo_plan!(plan, A, TRACE[])
    return HaloUpdate(plan, A)
end
ImplicitGlobalGrid.fused_halo_finish!(h::HaloUpdate{<:CuHaloPlan}) = finish_halo_plan!(h.plan, h.A, TRACE[])

# Every round but the receives and the unpack of the last one, which finish_halo_plan! does, then waits for the sends
function start_halo_plan!(plan::CuHaloPlan, A, tr::Bool)
    for k in eachindex(plan.rounds)
        k > 1 && finish_round!(plan, k - 1, A, tr)
        start_round!(plan, k, A, tr)
    end
end

function finish_halo_plan!(plan::CuHaloPlan, A, tr::Bool)
    isempty(plan.rounds) || finish_round!(plan, length(plan.rounds), A, tr)
    t = tr ? time_ns() : UInt64(0)
    for r in plan.rounds
        MPI.Waitall(r.sendreqs)
    end
    tr && trace_lap!(TQ_WAITALL_SEND, length(plan.rounds), t)
    return nothing
end

# Pack round k, wait for the pack and post its messages
function start_round!(plan::CuHaloPlan, k, A, tr::Bool)
    r, s, tag = plan.rounds[k], CUDA.stream(), FUSED_TAG + k
    t = tr ? time_ns() : UInt64(0)
    if r.blocks_send > 0
        plan.pack(r.sendbuf, A, r.sendregs, Val(true); threads=HALO_THREADS, blocks=(r.blocks_send, length(r.round.send)), stream=s)
    end
    tr && (t = trace_lap!(TQ_PACK, k, t))
    CUDA.synchronize(s; blocking=true)  # also orders this round's receives after the last unpack from its buffer
    tr && (t = trace_lap!(TQ_WAIT_PACK, k, t))
    for (m, msg) in enumerate(r.round.recvmsgs)
        MPI.Irecv!(r.recvmpi[m], msg.peer, tag, plan.comm, r.recvreqs[m])
    end
    tr && (t = trace_lap!(TQ_IRECV, k, t))
    for (m, msg) in enumerate(r.round.sendmsgs)
        MPI.Isend(r.sendmpi[m], msg.peer, tag, plan.comm, r.sendreqs[m])
    end
    tr && (trace_lap!(TQ_ISEND, k, t); trace_stamp!(TQ_T_FIRST_ISEND, k))
    return nothing
end

# Wait for the messages of round k and unpack them on the task's stream, without waiting
function finish_round!(plan::CuHaloPlan, k, A, tr::Bool)
    r = plan.rounds[k]
    t = tr ? time_ns() : UInt64(0)
    MPI.Waitall(r.recvreqs)
    tr && (t = trace_lap!(TQ_WAIT_RECV, k, t); trace_stamp!(TQ_T_RECV_DONE, k))
    if r.blocks_recv > 0
        plan.unpack(r.recvbuf, A, r.recvregs, Val(false); threads=HALO_THREADS, blocks=(r.blocks_recv, length(r.round.recv)), stream=CUDA.stream())
    end
    tr && trace_lap!(TQ_UNPACK, k, t)
    return nothing
end
