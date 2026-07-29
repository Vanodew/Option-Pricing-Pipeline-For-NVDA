module EWMA

export ewma_variance_path, ewma_hstep_variance

"""
    ewma_variance_path(returns; lambda=0.94, init=first(returns)^2)

RiskMetrics exponentially weighted moving average of variance
(J.P. Morgan/Reuters, RiskMetrics — Technical Document, 4th ed., 1996).

Returns a vector `v` where `v[t]` is the one-step-ahead conditional variance
forecast sigma^2_{t+1|t}, built from returns 1..t only:

    sigma^2_{t+1|t} = lambda * sigma^2_{t|t-1} + (1 - lambda) * r_t^2

`lambda = 0.94` is the RiskMetrics standard for daily data. There is nothing
to fit: lambda is a fixed constant chosen once by RiskMetrics across many
assets, not estimated per series. EWMA is the special case of GARCH(1,1)
with omega = 0, alpha = 1 - lambda, beta = lambda, so alpha + beta = 1
(integrated GARCH): shocks never decay toward a long-run level.

`init` seeds sigma^2_{1|0}. The default (first squared return) is arbitrary,
but its weight decays like lambda^t — after ~100 days (0.94^100 ≈ 0.002) the
seed is irrelevant. Discard a burn-in period before using the forecasts.
"""
function ewma_variance_path(
    returns::AbstractVector{<:Real};
    lambda::Float64=0.94,
    init::Float64=Float64(first(returns))^2,
)::Vector{Float64}
    0.0 < lambda < 1.0 || error("lambda must be strictly between 0 and 1.")
    init >= 0.0 || error("init must be a nonnegative variance.")
    n = length(returns)
    v = Vector{Float64}(undef, n)
    prev = init                       # sigma^2_{1|0}
    for t in 1:n
        prev = lambda * prev + (1.0 - lambda) * abs2(Float64(returns[t]))
        v[t] = prev                   # sigma^2_{t+1|t}
    end
    return v
end

"""
    ewma_hstep_variance(v_next, h) -> Float64

Total forecast variance over the next h days given the one-step forecast
`v_next` = sigma^2_{t+1|t}.

Because EWMA has alpha + beta = 1 and omega = 0, the k-step-ahead variance
forecast is flat: sigma^2_{t+k|t} = sigma^2_{t+1|t} for every k >= 1 (the
conditional variance is a martingale — today's level is the best guess for
every future day). So the h-day total is simply h times the one-step
forecast. Contrast with GARCH(1,1), where alpha + beta < 1 makes the k-step
forecast decay toward the unconditional variance.
"""
function ewma_hstep_variance(v_next::Real, h::Int)::Float64
    h >= 1 || error("Horizon h must be at least 1 day.")
    v_next >= 0 || error("Variance forecast must be nonnegative.")
    return h * Float64(v_next)
end

end # module
