module BlackScholes

using Distributions

export bs_d1_d2, bs_call_price

const std_normal = Normal(0.0, 1.0)

function bs_d1_d2(S::Real, K::Real, r::Real, sigma::Real, T::Real)
    d1 = (log(S / K) + (r + sigma^2 / 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return d1, d2
end

#Black-Scholes price of a European call on a non-dividend-paying stock:

function bs_call_price(S::Real, K::Real, r::Real, sigma::Real, T::Real)::Float64
    (S > 0 && K > 0) || error("Spot and strike must be positive.")
    sigma > 0 || error("Volatility must be positive.")
    T > 0 || error("Time to expiry must be positive (at T=0 the call is just max(S-K, 0)).")

    d1, d2 = bs_d1_d2(S, K, r, sigma, T)
    return S * cdf(std_normal, d1) - K * exp(-r * T) * cdf(std_normal, d2)
end

end # module
