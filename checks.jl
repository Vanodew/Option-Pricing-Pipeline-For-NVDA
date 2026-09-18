#sanity checks for the core pieces of the pipeline.
#run with:  julia --project=. checks.jl
#every check uses inputs where the right answer can be worked out by hand.

include("src/volatility.jl")
include("src/black_scholes.jl")
include("src/realized_vol.jl")
include("src/ewma.jl")
include("src/garch.jl")
include("src/loss.jl")
include("src/features.jl")
include("src/walk_forward.jl")
include("src/dm.jl")

using .Volatility
using .BlackScholes
using .RealizedVol
using .EWMA
using .Garch
using .Loss
using .WalkForward
using .DieboldMariano
using .Features
using Statistics
using Random: MersenneTwister, randn
using CSV: File
using DataFrames: DataFrame, nrow
import ARCHModels

approx(a, b; tol=1e-4) = abs(a - b) <= tol
relclose(a, b; rtol=1e-4) = abs(a - b) <= rtol * abs(b)
throws(f) = try (f(); false) catch; true end

# --- (a) prices -> log returns -----------------------------------------------
#100 -> 110 -> 121 is +10% twice, so both log returns come out as ln(1.1).
let
    r = log_returns([100.0, 110.0, 121.0])
    @assert length(r) == 2
    @assert approx(r[1], log(1.1)) && approx(r[2], log(1.1))
    println("(a) log_returns          OK  [100,110,121] -> [$(round(r[1], digits=5)), $(round(r[2], digits=5))], expected ln(1.1)=$(round(log(1.1), digits=5))")
end

# --- (b) std dev of returns ---------------------------------------------------
#[0.01, -0.01, 0.01, -0.01], the mean is 0 so the sample variance is 4*(0.01^2)/3.
let
    rets = [0.01, -0.01, 0.01, -0.01]
    expected = sqrt(4 * 0.01^2 / 3)
    @assert approx(std(rets), expected; tol=1e-8)
    println("(b) std of returns       OK  std=$(round(std(rets), digits=6)), expected sqrt(4*0.0001/3)=$(round(expected, digits=6))")
end

# --- (c) annualizing with sqrt(252) -------------------------------------------
#alternating +/-1% daily returns, daily std comes from (b) and annualized is that * sqrt(252).
let
    rets = [0.01, -0.01, 0.01, -0.01]
    expected = std(rets) * sqrt(252)
    got = annualized_volatility(rets)
    @assert approx(got, expected; tol=1e-10)
    println("(c) annualized_vol       OK  $(round(std(rets), digits=6)) daily -> $(round(got, digits=4)) annualized ($(round(100got, digits=1))%)")
end

# --- (d) Black-Scholes call ---------------------------------------------------
#textbook case, S=100, K=100, r=5%, sigma=20%, T=1 year gives C = 10.4506.
let
    price = bs_call_price(100.0, 100.0, 0.05, 0.20, 1.0)
    @assert approx(price, 10.4506; tol=1e-3)

    #structural checks, a deep in-the-money call should be about the forward intrinsic
    #value and a deep out-of-the-money one should be about 0.
    itm = bs_call_price(100.0, 1.0, 0.05, 0.20, 1.0)
    otm = bs_call_price(100.0, 10_000.0, 0.05, 0.20, 1.0)
    @assert approx(itm, 100.0 - 1.0 * exp(-0.05); tol=1e-6)
    @assert otm < 1e-10

    println("(d) bs_call_price        OK  S=K=100, r=5%, sigma=20%, T=1 -> $(round(price, digits=4)), textbook 10.4506")
    println("                             deep ITM -> S - K*exp(-rT), deep OTM -> 0: both hold")
end

# --- (e) realized-variance target alignment -----------------------------------
# returns = [0.01, 0.02, 0.03, 0.04], h = 2.
#the target at t=1 has to use ONLY the two future returns 0.02 and 0.03:
#   RV_1 = 0.02^2 + 0.03^2 = 0.0013.  Likewise RV_2 = 0.03^2 + 0.04^2 = 0.0025.
#t=3 and t=4 dont have a full 2-day window ahead of them so they come back missing.
#on the trailing (feature) side with w=2, TRV_2 = 0.01^2 + 0.02^2 = 0.0005 only uses past
#returns, and TRV_1 is missing because theres not enough history behind it.
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

    #the no-leakage property, the target at t must not move if the past (returns 1..t)
    #changes, and it must move if the future changes.
    r2 = copy(r); r2[1] = 99.0
    @assert forward_realized_variance(r2, 2)[1] == fwd[1]
    r3 = copy(r); r3[2] = 99.0
    @assert forward_realized_variance(r3, 2)[1] != fwd[1]

    println("(e) realized variance    OK  fwd[1]=0.0013 from future returns only; trailing[2]=0.0005 from past only; leakage probe holds")
end

# --- (f) EWMA collapses on a constant-return series ----------------------------
#if every return is 0.02 and the seed is 0.02^2 then the fixed point of the recursion is
#exactly 0.0004, since lambda*0.0004 + (1-lambda)*0.0004 = 0.0004 at every step.
#with a WRONG seed the path still has to converge to 0.0004 because the seed's weight decays
#like lambda^t, so after 300 steps the gap left is lambda^300 * |seed error|.
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
#and the h-step rule, 5-day total variance = 5 * the one-step forecast, because
#alpha + beta = 1 makes the multi-step forecast flat.
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
#only returns[1] goes into h[2]. today's return cant move today's variance, which is the
#no-look-ahead property of the recursion itself.
let
    h = garch11_variance_path([0.02, -0.03], 0.0001, 0.1, 0.8; mu=0.0, h0=0.0004)
    @assert approx(h[1], 0.0004; tol=1e-15)
    @assert approx(h[2], 0.00046; tol=1e-15)

    #changing returns[2] must not move either variance, it only ever feeds h[3] and that
    #doesnt exist here.
    h2 = garch11_variance_path([0.02, 99.0], 0.0001, 0.1, 0.8; mu=0.0, h0=0.0004)
    @assert h2 == h

    println("(h) GARCH recursion      OK  h=[0.0004, 0.00046] match pencil-and-paper; today's return does not move today's variance")
end

# --- (i) GARCH nests EWMA ---------------------------------------------------------
#EWMA is just GARCH(1,1) with omega = 0, alpha = 1-lambda, beta = lambda. given the same
#seed the one-step forecast paths have to agree EXACTLY and not approximately, its the same
#arithmetic done in the same order.
let
    lambda = 0.94
    r = [0.01, -0.02, 0.015, 0.03, -0.005, 0.02]
    seed = 0.0004

    ewma = ewma_variance_path(r; lambda=lambda, init=seed)
    g = garch11_forecast_path(r, GARCH11Params(0.0, 1 - lambda, lambda, 0.0); h0=seed)
    #ewma_variance_path returns sigma^2_{t+1|t} so its forecast-aligned, and tagging it says
    #that out loud instead of just assuming it.
    @assert g == OneStepForecast(ewma)

    println("(i) GARCH nests EWMA     OK  omega=0, alpha=1-lambda, beta=lambda reproduces the EWMA path bit-for-bit")
end

# --- (j) multi-step variance forecast ---------------------------------------------
# omega = 0.0001, alpha = 0.1, beta = 0.4 -> rho = 0.5, h_bar = 0.0001/0.5 = 0.0002.
#starting from a one-step forecast of 0.0006, the per-day forecasts decay halfway to h_bar
#every day:
#   k=1: 0.0006
#   k=2: 0.0002 + 0.5 *(0.0006-0.0002) = 0.0004
#   k=3: 0.0002 + 0.25*(0.0006-0.0002) = 0.0003
#   total = 0.0013
#the closed form has to reproduce that sum.
let
    p = GARCH11Params(0.0001, 0.1, 0.4, 0.0)
    @assert approx(garch11_hstep_variance(0.0006, p, 3), 0.0013; tol=1e-15)

    #degenerate case rho = 0, no persistence at all so every future day's variance is just
    #omega and the h-day total is h*omega.
    flat = GARCH11Params(0.0002, 0.0, 0.0, 0.0)
    @assert approx(garch11_hstep_variance(0.0002, flat, 5), 0.001; tol=1e-15)

    #IGARCH limit rho = 1, mean reversion switches off and the GARCH formula has to collapse
    #onto the EWMA rule instead of dividing by zero.
    igarch = GARCH11Params(0.0, 0.06, 0.94, 0.0)
    @assert garch11_hstep_variance(0.0004, igarch, 21) == ewma_hstep_variance(0.0004, 21)

    println("(j) GARCH h-step         OK  3-day total 0.0013 matches day-by-day decay; rho=0 gives h*omega; rho=1 collapses to the EWMA rule")
end

# --- (k) forecast/conditional alignment is exactly one index ----------------------
#the two series are tied together by an identity that holds term by term:
#   forecast[t] = omega + alpha*a_t^2 + beta*h_t = h_{t+1} = conditional[t+1]
#so an off-by-one in EITHER function breaks this straight away.
#the second assertion is what gives the first one teeth, it confirms the two alignments
#really do differ so the identity isnt passing trivially.
let
    r = [0.01, -0.02, 0.015, 0.03, -0.005, 0.02, -0.01, 0.025]
    p = GARCH11Params(0.0001, 0.1, 0.8, 0.002)

    cv = garch11_variance_path(r, p.omega, p.alpha, p.beta; mu=p.mu, h0=0.0004)
    fc = garch11_forecast_path(r, p; h0=0.0004)

    @assert variances(fc)[1:end-1] == variances(cv)[2:end]
    @assert variances(fc)[1:end-1] != variances(cv)[1:end-1]
    @assert timing(cv) === :conditional && timing(fc) === :forecast

    #mixing the two alignments has to be an error and not a silent `false`.
    @assert throws(() -> cv == fc)
    @assert throws(() -> fc == cv)
    @assert throws(() -> isapprox(cv, fc))
    #a bare vector carries no timing convention at all, so thats an error too.
    @assert throws(() -> cv == variances(cv))
    @assert throws(() -> variances(fc) == fc)
    #same tag still compares normally.
    @assert cv == garch11_variance_path(r, p.omega, p.alpha, p.beta; mu=p.mu, h0=0.0004)

    println("(k) GARCH alignment      OK  forecast[t] == conditional[t+1] exactly; cross-alignment comparison throws")
end

# --- (l) walk-forward seeding does not look ahead ----------------------------------
#h0_window=k seeds h_1 with var(returns[1:k]) and nothing else. so h[1..k+1], which only
#depend on the seed and returns[1..k], have to be untouched when every return after k is
#replaced with garbage.
#the last assertion deliberately shows the DEFAULT seed failing that same probe, var() over
#the whole vector really does pull future data into h_1. thats the reason h0_window exists,
#and this check would fail if someone "simplified" the seeding back to the full sample.
let
    r = [0.01, -0.02, 0.015, 0.03, -0.005, 0.02, -0.01, 0.025, -0.03, 0.005,
         0.01, -0.015, 0.02, -0.02, 0.01, 0.03, -0.025, 0.005, -0.01, 0.015]
    k = 8
    wrecked = copy(r)
    wrecked[(k + 1):end] .*= 50      # the future becomes wildly more volatile

    safe_a = garch11_variance_path(r, 0.0001, 0.1, 0.8; mu=0.0, h0_window=k)
    safe_b = garch11_variance_path(wrecked, 0.0001, 0.1, 0.8; mu=0.0, h0_window=k)
    @assert variances(safe_a)[1:(k + 1)] == variances(safe_b)[1:(k + 1)]

    #same guarantee through the forecast-aligned path, forecast[t] uses returns 1..t so the
    #safe prefix is one shorter.
    p = GARCH11Params(0.0001, 0.1, 0.8, 0.0)
    fa = garch11_forecast_path(r, p; h0_window=k)
    fb = garch11_forecast_path(wrecked, p; h0_window=k)
    @assert variances(fa)[1:k] == variances(fb)[1:k]

    #...and the default full-sample seed does NOT give you that guarantee.
    leak_a = garch11_variance_path(r, 0.0001, 0.1, 0.8; mu=0.0)
    leak_b = garch11_variance_path(wrecked, 0.0001, 0.1, 0.8; mu=0.0)
    @assert variances(leak_a)[1] != variances(leak_b)[1]

    #asking for the seed two ways at once is a mistake, not a precedence rule.
    @assert throws(() -> garch11_variance_path(r, 0.0001, 0.1, 0.8; h0=0.0004, h0_window=k))
    #and fit_garch11 respects the same option.
    @assert fit_garch11(r; h0_window=k) isa GARCH11Fit

    println("(l) GARCH h0 seeding     OK  h0_window=$k keeps h[1..$(k+1)] invariant to a wrecked future; the default full-sample seed does not")
end

# --- (m) MLE recovers known parameters --------------------------------------------
#simulate from a GARCH(1,1) with parameters we picked, then fit it, and the estimator has to
#land near the truth. not a pencil-and-paper case, but its the only way to test that the
#likelihood is actually maximized at the data-generating process and not at some other point
#that just happens to fit.
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

    #the truth should sit within a couple of standard errors of the estimate. weak claim on
    #purpose since this is one draw and not a coverage study, but it does catch standard
    #errors that are off by an order of magnitude.
    @assert abs(p.alpha - alpha_t) < 3 * f.se.alpha
    @assert abs(p.beta - beta_t) < 3 * f.se.beta

    println("(m) GARCH MLE recovery   OK  true (a=$alpha_t, b=$beta_t) -> fitted (a=$(round(p.alpha, digits=3)), b=$(round(p.beta, digits=3))) on $n simulated days, truth within 3 se")
end

# --- (n) hand-coded MLE and standard errors vs ARCHModels.jl ----------------------
#this is the real cross-check, same data and same specification but a totally separate
#implementation. agreeing to ~7 significant figures on the coefficients means the likelihood
#and the optimizer are both right. the likelihood also gets evaluated at ARCHModels' OWN
#fitted coefficients, which separates "my objective function is the same function" from "my
#optimizer found the same point", if only the second one failed the bug would be in the
#optimizer.
#the standard errors get checked the same way. ARCHModels reports the huber sandwich
#(general.jl: vcov = J^-1 (S'S) J^-1) so thats what `se` has to match, and agreeing also
#confirms the numerical hessian and score matrix since ARCHModels does both by automatic
#differentiation instead.
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

    #same likelihood function, just evaluated at their optimum.
    @assert approx(garch11_loglik(r, lib[1], lib[3], lib[2]; mu=lib[4]), lib_ll; tol=1e-6)

    #robust standard errors, my finite differences against their automatic differentiation.
    #tolerance is 1e-4 relative and the gap that actually shows up is ~2e-6.
    @assert relclose(f.se.omega, libse[1]; rtol=1e-4)
    @assert relclose(f.se.beta, libse[2]; rtol=1e-4)
    @assert relclose(f.se.alpha, libse[3]; rtol=1e-4)
    @assert relclose(f.se.mu, libse[4]; rtol=1e-4)

    #the naive hessian errors have to be strictly SMALLER than the sandwich ones here. if the
    #model were correctly specified the two would agree, thats the information matrix
    #equality, but daily returns are fat-tailed under a gaussian likelihood so they dont and
    #the naive version understates the uncertainty. if this ever flipped the sandwich would
    #be the thing to suspect.
    @assert f.se_hessian.alpha < f.se.alpha
    @assert f.se_hessian.beta < f.se.beta
    @assert approx(f.tstat.alpha, mine.alpha / f.se.alpha; tol=1e-12)

    ratio = f.se.alpha / f.se_hessian.alpha
    println("(n) GARCH vs ARCHModels  OK  omega=$(round(mine.omega, sigdigits=6)), alpha=$(round(mine.alpha, digits=6)), beta=$(round(mine.beta, digits=6)) vs library $(round(lib[1], sigdigits=6)), $(round(lib[3], digits=6)), $(round(lib[2], digits=6))")
    println("                             logL $(round(f.loglik, digits=6)) vs $(round(lib_ll, digits=6)); persistence $(round(mine.alpha + mine.beta, digits=4))")
    println("                             robust se alpha $(round(f.se.alpha, sigdigits=6)) vs $(round(libse[3], sigdigits=6)); t=$(round(f.tstat.alpha, digits=2)) (naive Hessian se would be $(round(ratio, digits=2))x smaller)")
end

# --- (o) QLIKE: zero at a perfect forecast, asymmetric off it ---------------------
#x = RV/F. qlike(x) = x - ln(x) - 1, which is exactly 0 at x=1 since log(1)=0.
#under-forecasting (F=1, RV=2 -> x=2) and over-forecasting by the same factor
#(F=2, RV=1 -> x=0.5) are NOT symmetric:
#   qlike(2,1): x=2   -> 2 - ln(2) - 1   = 1 - ln(2)   ~= 0.306853
#   qlike(1,2): x=0.5 -> 0.5 - ln(0.5) - 1 = ln(2) - 0.5 ~= 0.193147
#under-forecasting costs more, and that asymmetry is the whole reason for using QLIKE over
#MSE here, its not a side detail.
let
    @assert qlike(0.0004, 0.0004) == 0.0
    @assert approx(qlike(2.0, 1.0), 1 - log(2); tol=1e-12)
    @assert approx(qlike(1.0, 2.0), log(2) - 0.5; tol=1e-12)
    @assert qlike(2.0, 1.0) > qlike(1.0, 2.0)

    #a non-positive forecast variance means something upstream is broken, its not something
    #to divide by or take the log of.
    @assert throws(() -> qlike(1.0, 0.0))
    @assert throws(() -> qlike(1.0, -1.0))

    println("(o) qlike                OK  qlike(F,F)=0 exactly; under-forecast qlike(2,1)=$(round(qlike(2.0,1.0), digits=6)) > over-forecast qlike(1,2)=$(round(qlike(1.0,2.0), digits=6)); forecast<=0 throws")
end

# --- (p) MSE: symmetric squared error ----------------------------------------------
#(5-3)^2 = 4 and (3-5)^2 = 4, so unlike QLIKE the MSE doesnt care which side the error
#falls on.
let
    @assert mse(5.0, 3.0) == 4.0
    @assert mse(3.0, 5.0) == 4.0
    @assert mse(0.0004, 0.0004) == 0.0
    println("(p) mse                  OK  mse(5,3)=mse(3,5)=4.0; symmetric, unlike qlike")
end

# --- (q) mean_loss skips any pair with a missing side ------------------------------
#realized and forecasts each have one missing entry and theyre in DIFFERENT positions, same
#as a real walk-forward column would be (RV missing at the tail of the series, forecast
#missing wherever GARCH didnt converge). only day 1 has both sides present so mse(4,2) = 4.
#days 2 and 3 have to be dropped, not treated as 0 and not errored on.
let
    realized  = [4.0, missing, 9.0]
    forecasts = [2.0, 5.0, missing]
    @assert mean_loss(realized, forecasts, mse) == 4.0

    #mean_loss doesnt know which loss its aggregating, so the same skip-logic has to work
    #with qlike passed in instead.
    realized2  = [1.0, missing]
    forecasts2 = [1.0, 2.0]
    @assert mean_loss(realized2, forecasts2, qlike) == 0.0

    println("(q) mean_loss            OK  mismatched missing positions both drop out; mse case -> 4.0; generic over qlike too")
end

# --- (r) walkforward_splits: embargo and boundary arithmetic, by hand -------------
#n=20, min_train=8, test_size=4, h=3. train_end starts at 8 and steps by test_size, each
#test window starts h+1 days after train_end (thats the embargo) and gets capped at either
#test_size days or the end of the sample, whichever is smaller:
#   train_end=8  -> test_start=8+3+1=12,  test_end=min(15,20)=15 -> (8,  12:15)
#   train_end=12 -> test_start=16,        test_end=min(19,20)=19 -> (12, 16:19)
#   train_end=16 -> test_start=20,        test_end=min(23,20)=20 -> (16, 20:20)
#   train_end=20 -> 20+3=23 is not < 20 -> stop. 3 splits.
#a second run with n=18 and everything else the same hits the min(...,n) cap on the LAST
#window instead of stopping cleanly. the second window wants to run to day 19 but there are
#only 18 days, so it gets truncated to 16:18, thats 3 days not the full test_size=4, and
#theres no third window because train_end=16 fails 16+3=19 < 18.
let
    s = walkforward_splits(20, 8, 4, 3)
    @assert length(s) == 3
    @assert s[1].train_end == 8  && s[1].test_range == 12:15
    @assert s[2].train_end == 12 && s[2].test_range == 16:19
    @assert s[3].train_end == 16 && s[3].test_range == 20:20

    #expanding, so train_end grows by test_size every step and never resets.
    @assert s[2].train_end - s[1].train_end == 4
    @assert s[3].train_end - s[2].train_end == 4

    #every test window starts exactly h days after its train_end finishes, and consecutive
    #test windows are contiguous, so no day gets scored twice and no day gets skipped. the
    #days train_end+1..train_end+h are never tested by that window, thats the embargo.
    for sp in s
        @assert first(sp.test_range) == sp.train_end + 3 + 1
    end
    @assert last(s[1].test_range) + 1 == first(s[2].test_range)
    @assert last(s[2].test_range) + 1 == first(s[3].test_range)

    s_capped = walkforward_splits(18, 8, 4, 3)
    @assert length(s_capped) == 2
    @assert s_capped[2].test_range == 16:18   # capped to n=18, not 16:19
    @assert length(s_capped[2].test_range) == 3

    println("(r) walkforward_splits   OK  n=20 -> 3 splits, embargo=h and contiguous test windows hold by hand; n=18 caps the last window to 16:18")
end

# --- (s) walk_forward: wiring, not re-deriving GARCH/EWMA's own numbers -----------
#GARCH's numbers are already covered in (h)-(n) and EWMA's in (f)-(g). what is NOT covered
#yet is whether walk_forward hands back the RIGHT numbers for the RIGHT day. does
#results.realized[row] for day t actually equal forward_realized_variance(r,h)[t] worked out
#independently, does results.ewma_1step do the same against ewma_variance_path, does the
#(window,day) bookkeeping line up with the split's test_range exactly, and does a missing
#garch_1step match converged==false and nothing else. thats a wiring and indexing check
#rather than a numerical one, so cross-referencing against primitives that are already
#verified is the right tool, same idea as (i) checking GARCH against EWMA.
let
    r = [0.01 * sin(0.7t) + 0.002 for t in 1:30]
    min_train, test_size, h = 15, 5, 3

    out = walk_forward(r; min_train=min_train, test_size=test_size, h=h)
    splits = walkforward_splits(30, min_train, test_size, h)

    expected_rows = sum(length(sp.test_range) for sp in splits)
    @assert nrow(out) == expected_rows

    realized_ref = forward_realized_variance(r, h)
    ewma_ref = ewma_variance_path(r; lambda=0.94)

    row = 1
    for (window_id, sp) in enumerate(splits)
        for day in sp.test_range
            @assert out.window[row] == window_id
            @assert out.day[row] == day
            @assert isequal(out.realized[row], realized_ref[day])
            @assert out.ewma_1step[row] == ewma_ref[day]
            @assert out.ewma_hstep[row] == ewma_hstep_variance(ewma_ref[day], h)
            row += 1
        end
    end

    #a window's GARCH columns are missing exactly when that window's fit didnt converge,
    #never missing on a converged window and never present on a failed one.
    @assert all(out.converged .== .!ismissing.(out.garch_1step))
    @assert all(out.converged .== .!ismissing.(out.garch_hstep))
    @assert all(ismissing(out.garch_1step[i]) == ismissing(out.garch_hstep[i]) for i in 1:nrow(out))

    n_failed = length(splits) - length(unique(out.window[out.converged]))
    println("(s) walk_forward         OK  $(nrow(out)) rows match $(expected_rows) expected from the splits; realized/ewma columns match the independently-computed reference at every (window,day); $(n_failed)/$(length(splits)) windows failed to converge and their garch columns are missing, nothing else")
end

# --- (t) Newey-West long-run variance, by hand ------------------------------------
#d = [1,2,3,4], dbar = 2.5, deviations [-1.5,-0.5,0.5,1.5]. the divisor is 1/T.
#   gamma_0 = (2.25 + 0.25 + 0.25 + 2.25)/4 = 1.25
#   gamma_1 = [0.75 - 0.25 + 0.75]/4        = 0.3125
#   S(q=1)  = 1.25 + 2(0.5)(0.3125)         = 1.5625
let
    d = [1.0, 2.0, 3.0, 4.0]

    @assert approx(newey_west_lrv(d, 0), 1.25; tol=1e-15)
    @assert approx(newey_west_lrv(d, 1), 1.5625; tol=1e-15)
    @assert newey_west_lrv(d, 1) > newey_west_lrv(d, 0)

    #raising q re-weights every lag rather than just adding one on the end, the weight is
    #1 - j/(q+1). going q=1 -> q=2 moves lag 1 from 1/2 to 2/3 and adds lag 2 at 1/3.
    #   gamma_2 = [(0.5)(-1.5) + (1.5)(-0.5)]/4 = -0.375
    #   S(q=2)  = 1.25 + 2(2/3)(0.3125) + 2(1/3)(-0.375) = 17/12
    #S(q=2) < S(q=1) here, so more lags doesnt automatically mean a bigger variance.
    @assert approx(newey_west_lrv(d, 2), 17 / 12; tol=1e-15)
    @assert newey_west_lrv(d, 2) < newey_west_lrv(d, 1)

    #bad bandwidths get rejected instead of silently clamped.
    @assert throws(() -> newey_west_lrv(d, -1))
    @assert throws(() -> newey_west_lrv(d, 4))
    @assert throws(() -> newey_west_lrv([1.0], 0))

    println("(t) newey_west_lrv       OK  q=0 -> 1.25 = gamma_0; q=1 -> 1.5625 by hand; S grows with q on a positively autocorrelated series")
end

# --- (u) Diebold-Mariano statistic, by hand ---------------------------------------
# loss_a - loss_b = [1,2,3,4] from (t): dbar = 2.5, S(q=1) = 1.5625.
#   se   = sqrt(1.5625/4) = 0.625
#   stat = 2.5/0.625      = 4.0 exactly
#q=0 gives 2.5/sqrt(0.3125) = 2*sqrt(5) = 4.4721, so the HAC correction shrinks it, which is
#the right direction for a positively autocorrelated differential.
# HLN at h=1, T=4: adj = (4 + 1 - 2 + 0)/4 = 3/4, stat_hln = 4*sqrt(0.75)
#                      = 2*sqrt(3) = 3.4641
let
    loss_a = [3.0, 4.0, 5.0, 6.0]
    loss_b = [2.0, 2.0, 2.0, 2.0]

    r = dm_test(loss_a, loss_b; q=1, h=1)
    @assert r.n == 4 && r.q == 1
    @assert approx(r.dbar, 2.5; tol=1e-15)
    @assert approx(r.lrv, 1.5625; tol=1e-15)
    @assert approx(r.se, 0.625; tol=1e-15)
    @assert approx(r.stat, 4.0; tol=1e-15)
    @assert approx(r.stat_hln, 2 * sqrt(3); tol=1e-12)
    @assert r.stat_hln < r.stat          # the small-sample correction shrinks it

    r0 = dm_test(loss_a, loss_b; q=0, h=1)
    @assert approx(r0.stat, 2 * sqrt(5); tol=1e-12)
    @assert r.stat < r0.stat             # HAC shrinks vs the naive standard error

    #default bandwidth is h-1, thats the MA(h-1) structure of optimal h-step errors.
    @assert dm_test(loss_a, loss_b; h=4).q == 3
    #...but it can never go above T-1.
    @assert dm_test(loss_a, loss_b; h=99).q == 3

    println("(u) dm_test              OK  dbar=2.5, S=1.5625, se=0.625 -> stat=4.0 exactly; HAC shrinks 4.472=2sqrt(5) -> 4.0; HLN -> 2sqrt(3)=3.4641; default q = h-1")
end

# --- (v) DM sign convention, degeneracy, and missing handling ----------------------
#no arithmetic needed here. a model tied with itself gives stat 0 and p 1 rather than 0/0,
#swapping the arguments flips the sign and changes nothing else, and a pair only counts when
#both losses are present.
let
    a = [1.0, 4.0, 2.0, 7.0, 3.0]
    b = [2.0, 1.0, 5.0, 3.0, 4.0]

    same = dm_test(a, a; q=1, h=1)
    @assert same.dbar == 0.0 && same.lrv == 0.0
    @assert same.stat == 0.0 && same.pvalue == 1.0
    @assert !isnan(same.stat)

    fwd = dm_test(a, b; q=1, h=1)
    rev = dm_test(b, a; q=1, h=1)
    @assert approx(fwd.stat, -rev.stat; tol=1e-12)
    @assert approx(fwd.dbar, -rev.dbar; tol=1e-15)
    @assert approx(fwd.lrv, rev.lrv; tol=1e-15)     # variance is sign-blind
    @assert fwd.pvalue == rev.pvalue

    #higher loss on A => positive statistic => A is the worse one.
    worse = dm_test([5.0, 6.0, 7.0, 9.0], [1.0, 2.0, 3.0, 4.0]; q=0, h=1)
    @assert worse.dbar > 0 && worse.stat > 0

    #a constant nonzero differential has zero long-run variance and an infinite statistic.
    #reporting "no difference" there would be the worst possible answer so it refuses.
    @assert throws(() -> dm_test([5.0, 6.0, 7.0, 8.0], [1.0, 2.0, 3.0, 4.0]; q=0, h=1))

    #a missing value in either series drops that pair and only that pair.
    am = [1.0, missing, 2.0, 7.0, 3.0]
    bm = [2.0, 1.0, missing, 3.0, 4.0]
    dropped = dm_test(am, bm; q=1, h=1)
    @assert dropped.n == 3
    kept = dm_test([1.0, 7.0, 3.0], [2.0, 3.0, 4.0]; q=1, h=1)
    @assert approx(dropped.stat, kept.stat; tol=1e-15)

    @assert throws(() -> dm_test([1.0, 2.0], [1.0]))

    println("(v) dm_test conventions  OK  a vs itself -> stat 0, p 1 (not NaN); swapping args flips the sign only; positive stat = first model worse; missing pairs drop out and match the pre-filtered series")
end

# --- (w) GBT features: values by hand, and no look-ahead --------------------------
# r = [0.1,0.2,0.3,0.4,0.5], windows (2,3,4):
#   trv_2[2] = 0.01+0.04 = 0.05          trv_2[5] = 0.16+0.25 = 0.41
#   trv_3[3] = 0.01+0.04+0.09 = 0.14     trv_4[4] = 0.30      trv_4[5] = 0.54
#   ret_1[3] = 0.3                       ret_mean_5[5] = 1.5/5 = 0.3
#valid needs every column present, so trv_4 (t>=4) and ret_mean_5 (t>=5) leave only day 5.
let
    r = [0.1, 0.2, 0.3, 0.4, 0.5]
    f = build_features(r; windows=(2, 3, 4))

    @assert f.names == ["trv_2", "trv_3", "trv_4", "ret_1", "ret_mean_5", "ewma_1step"]
    @assert size(f.X) == (5, 6)

    @assert approx(f.X[2, 1], 0.05; tol=1e-15)
    @assert approx(f.X[5, 1], 0.41; tol=1e-15)
    @assert approx(f.X[3, 2], 0.14; tol=1e-15)
    @assert approx(f.X[4, 3], 0.30; tol=1e-15)
    @assert approx(f.X[5, 3], 0.54; tol=1e-15)
    @assert approx(f.X[3, 4], 0.3; tol=1e-15)
    @assert approx(f.X[5, 5], 0.3; tol=1e-15)
    @assert f.X[:, 6] == ewma_variance_path(r)

    @assert f.valid == [false, false, false, false, true]
    @assert all(isnan, f.X[1, 1:3])          # undefined rows are NaN, never 0.0

    #leakage probe, same shape as the ones in (e) and (l).
    long = [0.01 * sin(0.7t) + 0.002 for t in 1:100]
    wrecked = copy(long); wrecked[71:end] .*= 50.0
    a = build_features(long)
    b = build_features(wrecked)

    @assert isequal(a.X[1:70, :], b.X[1:70, :])
    @assert a.X[71, :] != b.X[71, :]         # and day 71 itself must move
    @assert a.valid == b.valid
    @assert findfirst(a.valid) == 63         # default max window is 63 days

    println("(w) build_features       OK  trv/ret/mean columns match pencil-and-paper; ewma column equals ewma_variance_path; valid starts at day 63 and undefined rows are NaN; wrecking returns from day 71 leaves rows 1..70 bit-identical")
end

# --- (x) GBT inside the walk-forward ----------------------------------------------
#wiring again like (s), not arithmetic. EvoTrees subsamples both rows and columns so the
#repeat-run assertion is the thing that pins the seed, without it a fresh clone wouldnt
#reproduce the committed numbers. (s) covers the other branch, where the sample is shorter
#than the feature window and GBT gets skipped.
let
    r = [0.01 * sin(0.7t) + 0.002 * cos(0.31t) for t in 1:250]
    min_train, test_size, h, fw = 120, 20, 5, (2, 3, 5)

    out = walk_forward(r; min_train=min_train, test_size=test_size, h=h, feature_windows=fw)
    feats = build_features(r; windows=fw)

    for row in 1:nrow(out)
        day = out.day[row]
        @assert ismissing(out.gbt_hstep[row]) == !feats.valid[day]
    end

    g = collect(skipmissing(out.gbt_hstep))
    @assert !isempty(g)
    @assert all(>(0), g) && all(isfinite, g)

    again = walk_forward(r; min_train=min_train, test_size=test_size, h=h, feature_windows=fw)
    @assert isequal(out.gbt_hstep, again.gbt_hstep)

    @assert isequal(out.ewma_hstep, again.ewma_hstep)
    @assert isequal(out.garch_hstep, again.garch_hstep)

    println("(x) walk_forward GBT     OK  $(length(g)) forecasts, present exactly where features exist, all positive and finite; two runs identical so the pinned seed reproduces; ewma/garch columns unchanged")
end

println("\nAll sanity checks passed.")
