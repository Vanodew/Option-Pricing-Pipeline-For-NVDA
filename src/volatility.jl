module Volatility

using Statistics

export log_returns, annualized_volatility

"""
    log_returns(prices) -> Vector{Float64}

Daily log returns from a price series: r_t = ln(P_t / P_{t-1}).

Returns a vector one element shorter than `prices` (the first day has no
previous close to compare against).
"""
function log_returns(prices::AbstractVector{<:Real})::Vector{Float64}
    length(prices) >= 2 || error("Need at least 2 prices to compute a return.")
    all(p -> p > 0, prices) || error("All prices must be positive to take logs.")
    return diff(log.(prices))
end

"""
    annualized_volatility(returns; trading_days=252) -> Float64

Annualized volatility: sample standard deviation of daily log returns,
scaled by sqrt(trading_days).

The sqrt scaling assumes returns are independent across days, so variance
grows linearly with time and standard deviation grows with its square root.
"""
function annualized_volatility(
    returns::AbstractVector{<:Real};
    trading_days::Int=252,
)::Float64
    length(returns) >= 2 || error("Need at least 2 returns for a standard deviation.")
    return std(returns) * sqrt(trading_days)
end

end # module
