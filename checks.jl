# Sanity checks for the core pieces of the pipeline.
# Run with:  julia --project=. checks.jl
# Each check uses inputs whose correct answer is known by hand.

include("src/volatility.jl")
include("src/black_scholes.jl")
include("src/realized_vol.jl")
include("src/ewma.jl")
include("src/garch.jl")

using .Volatility
using .BlackScholes
using .RealizedVol
using .EWMA
using .Garch
using Statistics
using Random: MersenneTwister, randn
using CSV: File
using DataFrames: DataFrame
import ARCHModels

approx(a, b; tol=1e-4) = abs(a - b) <= tol
relclose(a, b; rtol=1e-4) = abs(a - b) <= rtol * abs(b)
throws(f) = try (f(); false) catch; true end

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

# --- (h) GARCH(1,1) recursion by hand -------------------------------------------
# omega = 0.0001, alpha = 0.1, beta = 0.8, mu = 0, seed h_1 = 0.0004,
# returns [0.02, -0.03]:
#   h[1] = 0.0004                                     (the seed)
#   h[2] = 0.0001 + 0.1*(0.02)^2 + 0.8*0.0004
#        = 0.0001 + 0.00004 + 0.00032 = 0.00046
# Only returns[1] enters h[2] -- today's return cannot move today's variance,
# which is the no-look-ahead property of the recursion itself.
let
    h = garch11_variance_path([0.02, -0.03], 0.0001, 0.1, 0.8; mu=0.0, h0=0.0004)
    @assert approx(h[1], 0.0004; tol=1e-15)
    @assert approx(h[2], 0.00046; tol=1e-15)

    # Perturbing returns[2] must not change either variance (it is only used
    # to form h[3], which does not exist here).
    h2 = garch11_variance_path([0.02, 99.0], 0.0001, 0.1, 0.8; mu=0.0, h0=0.0004)
    @assert h2 == h

    println("(h) GARCH recursion      OK  h=[0.0004, 0.00046] match pencil-and-paper; today's return does not move today's variance")
end

# --- (i) GARCH nests EWMA ---------------------------------------------------------
# EWMA is GARCH(1,1) with omega = 0, alpha = 1-lambda, beta = lambda. With the
# same seed the one-step forecast paths must agree EXACTLY, not approximately:
# they are the same arithmetic in the same order.
let
    lambda = 0.94
    r = [0.01, -0.02, 0.015, 0.03, -0.005, 0.02]
    seed = 0.0004

    ewma = ewma_variance_path(r; lambda=lambda, init=seed)
    g = garch11_forecast_path(r, GARCH11Params(0.0, 1 - lambda, lambda, 0.0); h0=seed)
    # ewma_variance_path returns sigma^2_{t+1|t}, so it is forecast-aligned;
    # tagging it says so explicitly rather than assuming it.
    @assert g == OneStepForecast(ewma)

    println("(i) GARCH nests EWMA     OK  omega=0, alpha=1-lambda, beta=lambda reproduces the EWMA path bit-for-bit")
end

# --- (j) multi-step variance forecast ---------------------------------------------
# omega = 0.0001, alpha = 0.1, beta = 0.4 -> rho = 0.5, h_bar = 0.0001/0.5 = 0.0002.
# Starting from a one-step forecast of 0.0006, the per-day forecasts decay
# halfway to h_bar each day:
#   k=1: 0.0006
#   k=2: 0.0002 + 0.5 *(0.0006-0.0002) = 0.0004
#   k=3: 0.0002 + 0.25*(0.0006-0.0002) = 0.0003
#   total = 0.0013
# The closed form must reproduce that sum.
let
    p = GARCH11Params(0.0001, 0.1, 0.4, 0.0)
    @assert approx(garch11_hstep_variance(0.0006, p, 3), 0.0013; tol=1e-15)

    # Degenerate case rho = 0: no persistence at all, every future day's
    # variance is just omega, so the h-day total is h*omega.
    flat = GARCH11Params(0.0002, 0.0, 0.0, 0.0)
    @assert approx(garch11_hstep_variance(0.0002, flat, 5), 0.001; tol=1e-15)

    # IGARCH limit rho = 1: mean reversion switches off and the GARCH formula
    # must collapse onto the EWMA rule rather than divide by zero.
    igarch = GARCH11Params(0.0, 0.06, 0.94, 0.0)
    @assert garch11_hstep_variance(0.0004, igarch, 21) == ewma_hstep_variance(0.0004, 21)

    println("(j) GARCH h-step         OK  3-day total 0.0013 matches day-by-day decay; rho=0 gives h*omega; rho=1 collapses to the EWMA rule")
end

# --- (k) forecast/conditional alignment is exactly one index ----------------------
# The two series are related by an identity that holds term by term:
#   forecast[t] = omega + alpha*a_t^2 + beta*h_t = h_{t+1} = conditional[t+1]
# so an off-by-one introduced into EITHER function breaks this immediately.
# The second assertion is what gives the first one teeth: it confirms the two
# alignments genuinely differ, so the identity is not passing trivially.
let
    r = [0.01, -0.02, 0.015, 0.03, -0.005, 0.02, -0.01, 0.025]
    p = GARCH11Params(0.0001, 0.1, 0.8, 0.002)

    cv = garch11_variance_path(r, p.omega, p.alpha, p.beta; mu=p.mu, h0=0.0004)
    fc = garch11_forecast_path(r, p; h0=0.0004)

    @assert variances(fc)[1:end-1] == variances(cv)[2:end]
    @assert variances(fc)[1:end-1] != variances(cv)[1:end-1]
    @assert timing(cv) === :conditional && timing(fc) === :forecast

    # Mixing the two alignments must be an error, not a silent `false`.
    @assert throws(() -> cv == fc)
    @assert throws(() -> fc == cv)
    @assert throws(() -> isapprox(cv, fc))
    # A bare vector carries no timing convention, so that is an error too.
    @assert throws(() -> cv == variances(cv))
    @assert throws(() -> variances(fc) == fc)
    # Same tag still compares normally.
    @assert cv == garch11_variance_path(r, p.omega, p.alpha, p.beta; mu=p.mu, h0=0.0004)

    println("(k) GARCH alignment      OK  forecast[t] == conditional[t+1] exactly; cross-alignment comparison throws")
end

# --- (l) walk-forward seeding does not look ahead ----------------------------------
# h0_window=k seeds h_1 with var(returns[1:k]) and nothing else. So h[1..k+1]
# -- which depend only on the seed and returns[1..k] -- must be untouched when
# every return after k is replaced with garbage.
# The final assertion deliberately shows the DEFAULT seed failing that same
# probe: var() over the whole vector really does pull future data into h_1.
# That is why h0_window exists, and why this check would fail if someone
# "simplified" the seeding back to always using the full sample.
let
    r = [0.01, -0.02, 0.015, 0.03, -0.005, 0.02, -0.01, 0.025, -0.03, 0.005,
         0.01, -0.015, 0.02, -0.02, 0.01, 0.03, -0.025, 0.005, -0.01, 0.015]
    k = 8
    wrecked = copy(r)
    wrecked[(k + 1):end] .*= 50      # the future becomes wildly more volatile

    safe_a = garch11_variance_path(r, 0.0001, 0.1, 0.8; mu=0.0, h0_window=k)
    safe_b = garch11_variance_path(wrecked, 0.0001, 0.1, 0.8; mu=0.0, h0_window=k)
    @assert variances(safe_a)[1:(k + 1)] == variances(safe_b)[1:(k + 1)]

    # Same guarantee through the forecast-aligned path (forecast[t] uses
    # returns 1..t, so the safe prefix is one shorter).
    p = GARCH11Params(0.0001, 0.1, 0.8, 0.0)
    fa = garch11_forecast_path(r, p; h0_window=k)
    fb = garch11_forecast_path(wrecked, p; h0_window=k)
    @assert variances(fa)[1:k] == variances(fb)[1:k]

    # ...and the default full-sample seed does NOT have that guarantee.
    leak_a = garch11_variance_path(r, 0.0001, 0.1, 0.8; mu=0.0)
    leak_b = garch11_variance_path(wrecked, 0.0001, 0.1, 0.8; mu=0.0)
    @assert variances(leak_a)[1] != variances(leak_b)[1]

    # Asking for the seed two ways at once is a mistake, not a precedence rule.
    @assert throws(() -> garch11_variance_path(r, 0.0001, 0.1, 0.8; h0=0.0004, h0_window=k))
    # And fit_garch11 honours the same option.
    @assert fit_garch11(r; h0_window=k) isa GARCH11Fit

    println("(l) GARCH h0 seeding     OK  h0_window=$k keeps h[1..$(k+1)] invariant to a wrecked future; the default full-sample seed does not")
end

# --- (m) MLE recovers known parameters --------------------------------------------
# Simulate from a GARCH(1,1) with parameters we chose, then fit. The estimator
# must land near the truth. Not a pencil-and-paper case -- it is the only way
# to test that the likelihood is maximized at the data-generating process
# rather than at some other point that merely fits.
let
    rng = MersenneTwister(20260811)
    n = 6_000
    omega_t, alpha_t, beta_t, mu_t = 2.0e-6, 0.08, 0.90, 0.0005

    r = Vector{Float64}(undef, n)
    h = omega_t / (1 - alpha_t - beta_t)   # start at the long-run variance
    a = 0.0
    for t in 1:n
        t > 1 && (h = omega_t + alpha_t * a^2 + beta_t * h)
        a = sqrt(h) * randn(rng)
        r[t] = mu_t + a
    end

    f = fit_garch11(r)
    p = f.params
    @assert f.converged
    @assert abs(p.alpha - alpha_t) < 0.03
    @assert abs(p.beta - beta_t) < 0.05
    @assert abs((p.alpha + p.beta) - (alpha_t + beta_t)) < 0.02

    # The truth should sit inside a couple of standard errors of the estimate.
    # This is a weak claim on purpose -- it is one draw, not a coverage study --
    # but it catches standard errors that are wrong by an order of magnitude.
    @assert abs(p.alpha - alpha_t) < 3 * f.se.alpha
    @assert abs(p.beta - beta_t) < 3 * f.se.beta

    println("(m) GARCH MLE recovery   OK  true (a=$alpha_t, b=$beta_t) -> fitted (a=$(round(p.alpha, digits=3)), b=$(round(p.beta, digits=3))) on $n simulated days, truth within 3 se")
end

# --- (n) hand-coded MLE and standard errors vs ARCHModels.jl ----------------------
# The real cross-check: same data, same specification, independent
# implementation. Agreement to ~7 significant figures on the coefficients means
# the likelihood and the optimizer are both right. The likelihood is also
# evaluated at ARCHModels' OWN fitted coefficients, which separates "my
# objective function is the same function" from "my optimizer found the same
# point" -- if only the second check failed, the bug would be in the optimizer.
#
# The standard errors are checked the same way. ARCHModels reports the Huber
# sandwich (general.jl: vcov = J^-1 (S'S) J^-1), so that is what `se` must
# match; agreement also confirms the numerical Hessian and score matrix, since
# ARCHModels computes both by automatic differentiation instead.
let
    df = DataFrame(File(joinpath(@__DIR__, "data", "prices_10y.csv")))
    r = log_returns(df.close)

    am = ARCHModels.fit(ARCHModels.GARCH{1,1}, r)
    lib = ARCHModels.coef(am)          # ordered [omega, beta, alpha, mu]
    libse = ARCHModels.stderror(am)    # same order
    lib_ll = ARCHModels.loglikelihood(am)

    f = fit_garch11(r)
    mine = f.params
    @assert f.converged
    @assert approx(mine.omega, lib[1]; tol=1e-9)
    @assert approx(mine.beta, lib[2]; tol=1e-5)
    @assert approx(mine.alpha, lib[3]; tol=1e-5)
    @assert approx(mine.mu, lib[4]; tol=1e-8)
    @assert approx(f.loglik, lib_ll; tol=1e-6)

    # Same likelihood function, evaluated at their optimum.
    @assert approx(garch11_loglik(r, lib[1], lib[3], lib[2]; mu=lib[4]), lib_ll; tol=1e-6)

    # Robust standard errors, finite differences vs their automatic
    # differentiation. 1e-4 relative; the observed gap is ~2e-6.
    @assert relclose(f.se.omega, libse[1]; rtol=1e-4)
    @assert relclose(f.se.beta, libse[2]; rtol=1e-4)
    @assert relclose(f.se.alpha, libse[3]; rtol=1e-4)
    @assert relclose(f.se.mu, libse[4]; rtol=1e-4)

    # The naive Hessian errors must be strictly SMALLER than the sandwich here.
    # Under a correctly specified model the two would agree (the information
    # matrix equality); daily returns are fat-tailed under a Gaussian
    # likelihood, so they do not, and the naive version understates the
    # uncertainty. If this ever flipped, the sandwich would be suspect.
    @assert f.se_hessian.alpha < f.se.alpha
    @assert f.se_hessian.beta < f.se.beta
    @assert approx(f.tstat.alpha, mine.alpha / f.se.alpha; tol=1e-12)

    ratio = f.se.alpha / f.se_hessian.alpha
    println("(n) GARCH vs ARCHModels  OK  omega=$(round(mine.omega, sigdigits=6)), alpha=$(round(mine.alpha, digits=6)), beta=$(round(mine.beta, digits=6)) vs library $(round(lib[1], sigdigits=6)), $(round(lib[3], digits=6)), $(round(lib[2], digits=6))")
    println("                             logL $(round(f.loglik, digits=6)) vs $(round(lib_ll, digits=6)); persistence $(round(mine.alpha + mine.beta, digits=4))")
    println("                             robust se alpha $(round(f.se.alpha, sigdigits=6)) vs $(round(libse[3], sigdigits=6)); t=$(round(f.tstat.alpha, digits=2)) (naive Hessian se would be $(round(ratio, digits=2))x smaller)")
end

println("\nAll sanity checks passed.")
