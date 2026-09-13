module DieboldMariano

using Statistics: mean
using Distributions: Normal, TDist, cdf

export newey_west_lrv, dm_test, DMResult

"""
    newey_west_lrv(d, q) -> Float64

Long-run variance of `d` with `q` lags and Bartlett weights:

    S = gamma_0 + 2 * sum_{j=1}^{q} (1 - j/(q+1)) * gamma_j

The taper and the 1/T divisor on the autocovariances are what keep S from
coming out negative. q=0 gives back gamma_0.
"""
function newey_west_lrv(d::AbstractVector{<:Real}, q::Int)::Float64
    T = length(d)
    T >= 2 || error("Need at least 2 observations for a variance, got $T.")
    q >= 0 || error("Bandwidth q must be nonnegative, got $q.")
    q < T || error("Bandwidth q=$q must be less than the sample size T=$T.")

    dbar = mean(d)
    dev = Float64.(d) .- dbar

    gamma(j) = sum(dev[t] * dev[t - j] for t in (j + 1):T) / T

    S = gamma(0)
    for j in 1:q
        S += 2.0 * (1.0 - j / (q + 1)) * gamma(j)
    end
    return S
end

"""
Diebold-Mariano result. Positive `stat` means model A carries the larger loss.
`stat_hln` is the Harvey-Leybourne-Newbold small-sample version, read against a
t with n-1 df.
"""
struct DMResult
    dbar::Float64
    lrv::Float64
    se::Float64
    stat::Float64
    pvalue::Float64
    stat_hln::Float64
    pvalue_hln::Float64
    n::Int
    q::Int
end

"""
    dm_test(loss_a, loss_b; q=nothing, h=21) -> DMResult

Tests whether the mean of `d_t = a_t - b_t` differs from zero, using a
Newey-West standard error because overlapping h-day windows make `d` strongly
autocorrelated. A naive standard error here is roughly 3x too small.

`q` defaults to `h - 1`: optimal h-step errors are MA(h-1), so later
autocovariances are zero. Pairs with a missing loss are dropped.
"""
function dm_test(loss_a, loss_b; q::Union{Nothing,Int}=nothing, h::Int=21)::DMResult
    length(loss_a) == length(loss_b) ||
        error("Loss series must be the same length, got $(length(loss_a)) and $(length(loss_b)).")
    h >= 1 || error("Horizon h must be at least 1 day.")

    d = Float64[]
    for (a, b) in zip(loss_a, loss_b)
        (ismissing(a) || ismissing(b)) && continue
        push!(d, Float64(a) - Float64(b))
    end

    T = length(d)
    T >= 2 || error("Need at least 2 paired observations, got $T.")

    bandwidth = q === nothing ? h - 1 : q
    bandwidth = min(bandwidth, T - 1)

    dbar = mean(d)
    S = newey_west_lrv(d, bandwidth)

    # S == 0 means a constant differential. Identical models are a real answer;
    # a constant nonzero gap is a wiring fault, and reporting "no difference"
    # for a model that loses every day would be worse than failing.
    if S <= 0.0
        dbar == 0.0 && return DMResult(dbar, S, 0.0, 0.0, 1.0, 0.0, 1.0, T, bandwidth)
        error("Degenerate test: loss differential is constant at $dbar, so its " *
              "long-run variance is zero and the statistic is infinite.")
    end

    se = sqrt(S / T)
    stat = dbar / se
    pvalue = 2.0 * (1.0 - cdf(Normal(), abs(stat)))

    adj = (T + 1 - 2h + h * (h - 1) / T) / T
    stat_hln = adj > 0 ? stat * sqrt(adj) : stat
    pvalue_hln = 2.0 * (1.0 - cdf(TDist(T - 1), abs(stat_hln)))

    return DMResult(dbar, S, se, stat, pvalue, stat_hln, pvalue_hln, T, bandwidth)
end

end # module
