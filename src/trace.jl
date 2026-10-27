export halo_trace, halo_trace_reset!

# Opt-in timers for update_halo! (environment variable IGG_TRACE=1, read by init_global_grid). Each rank adds up the
# host time of every phase of every call, per dimension (per round of messages with IGG_FUSED) and per call signature
# (size of the first array, number of fields, IGG_FUSED); halo_trace() gathers them on one rank and prints them,
# halo_trace_reset!() clears them (e.g. after warmup).
# Off by default; the cost when off is one branch per phase.

const TRACE = Ref(false)
const TRACE_T0 = Ref(UInt64(0))   # time_ns() at the entry of the current update_halo! call
const TRACE_GC0 = Ref((0, 0))     # allocated bytes and GC time [ns] at that entry

# Per dimension: host time of each phase [s], and two instants since the call entry [s]: the first Isend of the
# dimension posted (what the neighbours wait for) and the last receive of the dimension completed.
const TRACE_DIM_ROWS = (:pack, :irecv, :wait_pack, :isend, :wait_recv, :unpack, :wait_unpack, :waitall_send,
                        :t_first_isend, :t_recv_done)
const TQ_PACK, TQ_IRECV, TQ_WAIT_PACK, TQ_ISEND, TQ_WAIT_RECV, TQ_UNPACK, TQ_WAIT_UNPACK, TQ_WAITALL_SEND,
      TQ_T_FIRST_ISEND, TQ_T_RECV_DONE = 1:length(TRACE_DIM_ROWS)
# Per call: the argument checks (grid activation, wrapping, check_fields) and the buffer, task and stream allocation
# before the first dimension [s], the total [s], the allocated bytes and the GC time [s].
const TRACE_CALL_ROWS = (:checks, :buffers, :total, :alloc_bytes, :gc)
const TQ_CHECKS, TQ_BUFFERS, TQ_TOTAL, TQ_ALLOC, TQ_GC = 1:length(TRACE_CALL_ROWS)

mutable struct HaloTrace
    ncalls::Int
    dim::Matrix{Float64}   # length(TRACE_DIM_ROWS) x 3, summed over the calls
    call::Vector{Float64}  # length(TRACE_CALL_ROWS), summed over the calls
end
HaloTrace() = HaloTrace(0, zeros(length(TRACE_DIM_ROWS), 3), zeros(length(TRACE_CALL_ROWS)))

const TRACE_CUR = HaloTrace()                     # the current call
const TRACE_STATS = Dict{Any,HaloTrace}()         # (size of the first array, number of fields) => sums

init_trace() = (TRACE[] = parse(Int, get(ENV, "IGG_TRACE", "0")) > 0; nothing)

_gc_state() = (Int(Base.gc_bytes()), Int(Base.gc_num().total_time))

function trace_begin!()
    TRACE_CUR.dim .= 0; TRACE_CUR.call .= 0
    TRACE_GC0[] = _gc_state()
    TRACE_T0[] = time_ns()
    return nothing
end

# Add the time since t to phase q of dimension dim; return the current time.
@inline function trace_lap!(q::Integer, dim::Integer, t::UInt64)
    t1 = time_ns()
    @inbounds TRACE_CUR.dim[q, dim] += (t1 - t) * 1e-9
    return t1
end

# Record the current instant (since the call entry) in row q of dimension dim; `first` keeps an earlier record.
@inline function trace_stamp!(q::Integer, dim::Integer; first::Bool=false)
    @inbounds if !(first && TRACE_CUR.dim[q, dim] > 0)
        TRACE_CUR.dim[q, dim] = (time_ns() - TRACE_T0[]) * 1e-9
    end
    return nothing
end

# Record the time since the call entry in row q of the call (TQ_BUFFERS: minus the checks).
function trace_setup!(q::Integer)
    TRACE_CUR.call[q] = (time_ns() - TRACE_T0[]) * 1e-9 - (q == TQ_BUFFERS ? TRACE_CUR.call[TQ_CHECKS] : 0.0)
    return nothing
end

function trace_end!(sz::Tuple, nfields::Integer, fused::Integer)
    t1 = time_ns()
    b, g = _gc_state()
    c = TRACE_CUR.call
    c[TQ_TOTAL] = (t1 - TRACE_T0[]) * 1e-9
    c[TQ_ALLOC] = b - TRACE_GC0[][1]
    c[TQ_GC]    = (g - TRACE_GC0[][2]) * 1e-9
    s = get!(HaloTrace, TRACE_STATS, (sz, nfields, fused))
    s.ncalls += 1; s.dim .+= TRACE_CUR.dim; s.call .+= c
    return nothing
end

"""
    halo_trace_reset!()

Clear the `update_halo!` timers of this rank (see [`halo_trace`](@ref)).
"""
halo_trace_reset!() = (empty!(TRACE_STATS); nothing)

"""
    halo_trace(; io=stdout, root=0, per=1, unit="run", byrank=false)

Gather the `update_halo!` timers of all ranks (collective) and print them on rank `root`: per call signature (size of
the first array, number of fields, `IGG_FUSED` if the fused update ran), the mean time per call of each phase and
dimension (round of messages) [µs], as the median and the range over the ranks, or (`byrank=true`) for each rank.
The medians of the phases come from different ranks and do not add up: ranks wait (`wait_recv`) in different
dimensions, for neighbours that are late. `t_first_isend` and `t_recv_done` are instants since the call entry; a rank
waits for its neighbours' `t_first_isend`. The totals are divided by `per` (e.g. the number of time steps, with
`unit="step"`). Recorded only with `IGG_TRACE=1` set before `init_global_grid`.
"""
function halo_trace(; io::IO=stdout, root::Integer=0, per::Integer=1, unit::AbstractString="run", byrank::Bool=false)
    check_initialized()
    all_stats = MPI.gather(TRACE_STATS, comm(); root=root)
    me() == root || return nothing
    TRACE[] || println(io, "halo_trace: IGG_TRACE was not set at init_global_grid, nothing recorded")
    sigs = unique(reduce(vcat, [collect(keys(s)) for s in all_stats]))
    isempty(sigs) && return nothing
    fmt(x) = isnan(x) ? "-" : x < 1e3 ? string(round(x, digits=1)) : string(round(Int, x))
    # (min, median, max) of f(sums of a rank) over the ranks that made calls of signature k, per call by default
    function over_ranks(f, k; per_call=true)
        v = [f(s[k]) / (per_call ? s[k].ncalls : 1) for s in all_stats if haskey(s, k) && s[k].ncalls > 0]
        isempty(v) ? (NaN, NaN, NaN) : (minimum(v), sort(v)[cld(length(v), 2)], maximum(v))
    end
    cell(t; scale=1e6) = "$(fmt(scale * t[2])) [$(fmt(scale * t[1])), $(fmt(scale * t[3]))]"
    total(s, k) = haskey(s, k) ? s[k].call[TQ_TOTAL] : 0.0
    sort!(sigs, by=k -> -sum(total(s, k) for s in all_stats))
    grand = sort([sum(total(s, k) for k in sigs) / per for s in all_stats])
    println(io, "update_halo! timers of $(length(all_stats)) ranks: mean per call [µs], median [min, max] over the ranks")
    println(io, "all calls: $(cell((grand[1], grand[cld(end, 2)], grand[end]); scale=1e3)) ms per $unit")
    for k in sigs
        dims_used = [d for d in 1:3 if any(s -> haskey(s, k) && any(>(0), view(s[k].dim, :, d)), all_stats)]
        println(io, "\n$(join(k[1], "x")), $(k[2]) field(s)$(k[3] > 0 ? ", IGG_FUSED=$(k[3])" : ""): $(cell(over_ranks(st -> st.ncalls, k; per_call=false); scale=1)) " *
                    "calls, $(cell(over_ranks(st -> st.call[TQ_TOTAL] / per, k; per_call=false); scale=1e3)) ms per $unit")
        println(io, "  per call: total $(cell(over_ranks(st -> st.call[TQ_TOTAL], k))), " *
                    "checks $(cell(over_ranks(st -> st.call[TQ_CHECKS], k))), buffers $(cell(over_ranks(st -> st.call[TQ_BUFFERS], k))), " *
                    "gc $(cell(over_ranks(st -> st.call[TQ_GC], k))), " *
                    "allocated $(cell(over_ranks(st -> st.call[TQ_ALLOC], k); scale=1)) bytes")
        println(io, "  ", rpad("", 14), join((rpad("$(k[3] > 0 ? "round" : "dim") $d", 28) for d in dims_used)))
        for (q, name) in enumerate(TRACE_DIM_ROWS)
            println(io, "  ", rpad(string(name), 14), join((rpad(cell(over_ranks(st -> st.dim[q, d], k)), 28) for d in dims_used)))
        end
        byrank || continue
        println(io, "  by rank:", join((" | $(k[3] > 0 ? "round" : "dim") $d: " * join(TRACE_DIM_ROWS, " ") for d in dims_used)))
        for (r, s) in enumerate(all_stats)
            haskey(s, k) && s[k].ncalls > 0 || continue
            st = s[k]
            println(io, "  ", lpad(r - 1, 6), " total ", fmt(1e6 * st.call[TQ_TOTAL] / st.ncalls),
                    join((" | " * join((fmt(1e6 * st.dim[q, d] / st.ncalls) for q in eachindex(TRACE_DIM_ROWS)), " ") for d in dims_used)))
        end
    end
    return nothing
end
