# Sanity checks for the four core pieces of the pipeline.
# Run with:  julia --project=. checks.jl
# Each check uses inputs whose correct answer is known by hand.

include("src/volatility.jl")
include("src/black_scholes.jl")

using .Volatility
using .BlackScholes
using Statistics

approx(a, b; tol=1e-4) = abs(a - b) <= tol

# --- (a) prices -> log returns -----------------------------------------------
# 100 -> 110 -> 121 is +10% twice, so both log returns equal ln(1.1).
let
    r = log_returns([100.0, 110.0, 121.0])
    @assert length(r) == 2
    @assert approx(r[1], log(1.1)) && approx(r[2], log(1.1))
    println("(a) log_returns          OK  [100,110,121] -> [$(round(r[1], digits=5)), $(round(r[2], digits=5))], expected ln(1.1)=$(round(log(1.1), digits=5))")
end

# --- (b) std dev of returns ---------------------------------------------------
# [0.01, -0.01, 0.01, -0.01]: mean 0, sample variance 4*(0.01^2)/3.
let
    rets = [0.01, -0.01, 0.01, -0.01]
    expected = sqrt(4 * 0.01^2 / 3)
    @assert approx(std(rets), expected; tol=1e-8)
    println("(b) std of returns       OK  std=$(round(std(rets), digits=6)), expected sqrt(4*0.0001/3)=$(round(expected, digits=6))")
end

# --- (c) annualizing with sqrt(252) -------------------------------------------
# Alternating +/-1% daily returns: daily std from (b), annualized = that * sqrt(252).
let
    rets = [0.01, -0.01, 0.01, -0.01]
    expected = std(rets) * sqrt(252)
    got = annualized_volatility(rets)
    @assert approx(got, expected; tol=1e-10)
    println("(c) annualized_vol       OK  $(round(std(rets), digits=6)) daily -> $(round(got, digits=4)) annualized ($(round(100got, digits=1))%)")
end

# --- (d) Black-Scholes call ---------------------------------------------------
# Textbook case: S=100, K=100, r=5%, sigma=20%, T=1 year -> C = 10.4506.
let
    price = bs_call_price(100.0, 100.0, 0.05, 0.20, 1.0)
    @assert approx(price, 10.4506; tol=1e-3)

    # Structural checks: deep in-the-money call ~ forward intrinsic value,
    # deep out-of-the-money call ~ 0.
    itm = bs_call_price(100.0, 1.0, 0.05, 0.20, 1.0)
    otm = bs_call_price(100.0, 10_000.0, 0.05, 0.20, 1.0)
    @assert approx(itm, 100.0 - 1.0 * exp(-0.05); tol=1e-6)
    @assert otm < 1e-10

    println("(d) bs_call_price        OK  S=K=100, r=5%, sigma=20%, T=1 -> $(round(price, digits=4)), textbook 10.4506")
    println("                             deep ITM -> S - K*exp(-rT), deep OTM -> 0: both hold")
end

println("\nAll sanity checks passed.")
