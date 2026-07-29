module RealizedVol

export forward_realized_variance, trailing_realized_variance

"""
    forward_realized_variance(returns, h) -> Vector{Union{Missing,Float64}}

Forward-looking h-day realized variance, the *target* for volatility
forecasting. Aligned so that `out[t]` uses only returns from days t+1 .. t+h:

    RV_t = sum_{i=1}^{h} r_{t+i}^2

A forecast made at time t (using information up to and including day t) is
scored against `out[t]`. The last h entries are `missing` because their
future window extends past the end of the sample.

No mean is subtracted: over a few weeks the squared mean daily return is
negligible relative to the variance, and the sum of squared returns is the
standard realized-variance estimator (Patton 2011, J. Econometrics 160).
Realized *volatility* over the same window is sqrt(RV_t); models are fit and
scored in variance units, and vol is only taken at the reporting stage.
"""
function forward_realized_variance(
    returns::AbstractVector{<:Real}, h::Int,
)::Vector{Union{Missing,Float64}}
    h >= 1 || error("Horizon h must be at least 1 day.")
    n = length(returns)
    n > h || error("Need more than h=$h returns, got $n.")
    out = Vector{Union{Missing,Float64}}(missing, n)
    for t in 1:(n - h)
        out[t] = sum(abs2, @view returns[(t + 1):(t + h)])
    end
    return out
end

"""
    trailing_realized_variance(returns, w) -> Vector{Union{Missing,Float64}}

Backward-looking w-day realized variance, safe to use as a *feature*:
`out[t]` uses only returns from days t-w+1 .. t, i.e. information available
at the close of day t:

    TRV_t = sum_{i=0}^{w-1} r_{t-i}^2

The first w-1 entries are `missing` (not enough history yet). By
construction a feature at index t and a target from
`forward_realized_variance` at the same index t share no return
observations — the feature window ends at t, the target window starts at
t+1. That non-overlap is the no-look-ahead guarantee.
"""
function trailing_realized_variance(
    returns::AbstractVector{<:Real}, w::Int,
)::Vector{Union{Missing,Float64}}
    w >= 1 || error("Window w must be at least 1 day.")
    n = length(returns)
    n >= w || error("Need at least w=$w returns, got $n.")
    out = Vector{Union{Missing,Float64}}(missing, n)
    for t in w:n
        out[t] = sum(abs2, @view returns[(t - w + 1):t])
    end
    return out
end

end # module
