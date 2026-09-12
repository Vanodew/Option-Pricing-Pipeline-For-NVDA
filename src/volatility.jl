module Volatility

using Statistics

export log_returns, annualized_volatility

function log_returns(prices::AbstractVector{<:Real})::Vector{Float64}
    length(prices) >= 2 || error("Need at least 2 prices to compute a return.")
    all(p -> p > 0, prices) || error("All prices must be positive to take logs.")
    return diff(log.(prices))
end

function annualized_volatility(
    returns::AbstractVector{<:Real};
    trading_days::Int=252,
)::Float64
    length(returns) >= 2 || error("Need at least 2 returns for a standard deviation.")
    return std(returns) * sqrt(trading_days)
end

end # module
