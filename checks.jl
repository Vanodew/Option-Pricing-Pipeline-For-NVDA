# Sanity checks for the four core pieces of the pipeline.
# Run with:  julia --project=. checks.jl
# Each check uses inputs whose correct answer is known by hand.

include("src/volatility.jl")
include("src/black_scholes.jl")
include("src/realized_vol.jl")
include("src/ewma.jl")

using .Volatility
using .BlackScholes
using .RealizedVol
using .EWMA
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

# --- (e) realized-variance target alignment -----------------------------------
# returns = [0.01, 0.02, 0.03, 0.04], h = 2.
# Target at t=1 must use ONLY the two future returns 0.02 and 0.03:
#   RV_1 = 0.02^2 + 0.03^2 = 0.0013.  Likewise RV_2 = 0.03^2 + 0.04^2 = 0.0025.
# t=3 and t=4 have no full 2-day future window -> missing.
# Trailing (feature) side with w=2: TRV_2 = 0.01^2 + 0.02^2 = 0.0005 uses only
# past returns; TRV_1 is missing (not enough history).
let
    r = [0.01, 0.02, 0.03, 0.04]
    fwd = forward_realized_variance(r, 2)
    @assert approx(fwd[1], 0.0013; tol=1e-12)
    @assert approx(fwd[2], 0.0025; tol=1e-12)
    @assert ismissing(fwd[3]) && ismissing(fwd[4])

    trl = trailing_realized_variance(r, 2)
    @assert ismissing(trl[1])
    @assert approx(trl[2], 0.0005; tol=1e-12)
    @assert approx(trl[4], 0.0025; tol=1e-12)

    # No-leakage property: the target at t must not change if we alter the
    # past (returns 1..t), and must change if we alter the future.
    r2 = copy(r); r2[1] = 99.0
    @assert forward_realized_variance(r2, 2)[1] == fwd[1]
    r3 = copy(r); r3[2] = 99.0
    @assert forward_realized_variance(r3, 2)[1] != fwd[1]

    println("(e) realized variance    OK  fwd[1]=0.0013 from future returns only; trailing[2]=0.0005 from past only; leakage probe holds")
end

# --- (f) EWMA collapses on a constant-return series ----------------------------
# If every return is 0.02 and the seed is 0.02^2, the recursion's fixed point
# is exactly 0.0004: lambda*0.0004 + (1-lambda)*0.0004 = 0.0004 at every step.
# With a WRONG seed the path must still converge to 0.0004 (seed weight
# decays like lambda^t): after 300 steps the gap is lambda^300 * |seed error|.
let
    r = fill(0.02, 300)
    v_exact = ewma_variance_path(r; lambda=0.94, init=0.0004)
    @assert all(x -> approx(x, 0.0004; tol=1e-15), v_exact)

    v_conv = ewma_variance_path(r; lambda=0.94, init=0.10)  # absurd seed
    @assert approx(v_conv[end], 0.0004; tol=1e-8)
    @assert abs(v_conv[1] - 0.0004) > 0.05  # early values still polluted by seed

    println("(f) EWMA fixed point     OK  constant 2% returns -> variance pins at 0.0004; wrong seed washes out by t=300")
end

# --- (g) EWMA recursion by hand -------------------------------------------------
# lambda = 0.5, seed sigma^2_{1|0} = 0, returns [0.1, 0.2]:
#   v[1] = 0.5*0     + 0.5*0.01 = 0.005
#   v[2] = 0.5*0.005 + 0.5*0.04 = 0.0225
# And the h-step rule: 5-day total variance = 5 * one-step forecast,
# because alpha + beta = 1 makes the multi-step forecast flat.
let
    v = ewma_variance_path([0.1, 0.2]; lambda=0.5, init=0.0)
    @assert approx(v[1], 0.005; tol=1e-15)
    @assert approx(v[2], 0.0225; tol=1e-15)
    @assert approx(ewma_hstep_variance(v[2], 5), 0.1125; tol=1e-15)
    println("(g) EWMA hand recursion  OK  [0.005, 0.0225] match pencil-and-paper; 5-day total = 5x one-step = 0.1125")
end

println("\nAll sanity checks passed.")
