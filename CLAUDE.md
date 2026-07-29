# CLAUDE.md

Project instructions for Claude Code. Read before making changes.

## What this project is

Two halves that must eventually connect:

1. **Volatility forecasting** — predict 21-day-ahead realized variance for NVDA.
2. **Option pricing** — feed the forecast into a pricer (Black-Scholes v0, Monte Carlo later).

Headline research question: *does a gradient-boosted-tree model beat GARCH(1,1)/HAR at
21-day-ahead realized-variance forecasting, measured by out-of-sample QLIKE under
walk-forward validation?*

The link between the halves: the forecast, annualized and horizon-matched to the option
tenor, is the sigma fed into the pricer. The Monte Carlo half only earns its keep once it
prices a payoff with no closed form (arithmetic Asian) — otherwise it is just a slow
Black-Scholes.

## Layout

```
main.jl                 entry point: fetch -> vol -> price -> report
checks.jl               hand-verifiable sanity checks, one per core piece
src/data.jl             DataFetch: Yahoo chart endpoint + CSV cache
src/volatility.jl       Volatility: log_returns, annualized_volatility
src/black_scholes.jl    BlackScholes: bs_d1_d2, bs_call_price
src/realized_vol.jl     RealizedVol: forward (target) and trailing (feature) RV
src/ewma.jl             EWMA: RiskMetrics lambda=0.94 variance path, h-step rule
data/prices.csv         2y cached closes
data/prices_10y.csv     10y cached closes (2512 rows, 2016-07-19 -> 2026-07-16)
```

## Setup

```sh
julia --project=. -e "using Pkg; Pkg.instantiate()"   # one-time, reproduces pinned env
julia --project=. checks.jl                            # must print "All sanity checks passed."
julia --project=. main.jl                              # full pipeline
```

`Manifest.toml` is committed on purpose — it pins exact dependency versions so a fresh
clone rebuilds the identical environment.

## Conventions that are not negotiable

- **Forecast horizon h = 21 trading days.** One month. Used consistently as the target
  window and the option tenor.
- **Fit and score in variance units**, not volatility. Take the square root only at the
  reporting stage.
- **No look-ahead.** A feature at index `t` may only use returns up to and including day
  `t`; the target at index `t` uses days `t+1 .. t+h`. `trailing_realized_variance` and
  `forward_realized_variance` are built so their windows never overlap at a given index.
  Any new feature must preserve this. `checks.jl` case (e) has a leakage probe — a target
  must be unchanged when the past is perturbed and must change when the future is.
- **Loss function: QLIKE** = `RV/F - ln(RV/F) - 1`, where `F` is the forecast. Chosen over
  MSE because it is robust to the heavy right tail of variance.
- **Overlapping windows.** 21-day forward windows at consecutive `t` share 20 returns, so
  errors are strongly autocorrelated. Any significance test (Diebold-Mariano) needs HAC
  standard errors. This is a known caveat, not something to silently ignore.
- Every module gets sanity checks in `checks.jl` whose correct answer is derivable by hand.
  Not property tests, not fuzzing — pencil-and-paper cases.

## Plan order

Done:
- [x] BS v0 pipeline (on `main`)
- [x] `realized_vol.jl` — forward/trailing RV targets and features
- [x] `ewma.jl` — RiskMetrics EWMA baseline

Next, in this order:
- [ ] GARCH(1,1) by hand-coded MLE, cross-checked against `ARCHModels.jl`
- [ ] HAR baseline (Corsi 2009: daily/weekly/monthly RV regressors)
- [ ] Walk-forward evaluation harness + QLIKE + Diebold-Mariano with HAC
- [ ] GBT via `EvoTrees.jl` — only after the baselines have a number to beat
- [ ] BS Greeks by hand (analytic, then finite-difference cross-check)
- [ ] Monte Carlo pricer validated against BS closed form
- [ ] GPU benchmark (note: target hardware is weak at FP64 — Float32 accuracy caveat)
- [ ] Arithmetic Asian payoff — the thing that justifies simulation
- [ ] Connect the two halves

Deferred until a headline number exists: PINNs, rough Bergomi, deep hedging. Do not start
these; they are resume buzzwords without the baseline result behind them.

## How to work with the author

- **Hand-code the model, use libraries for plumbing.** The point of this project is
  understanding the estimator, so GARCH MLE, the Greeks, and the loss functions get written
  out; CSV parsing and HTTP do not. When a library would hide the thing being learned, say
  so and write it out instead.
- **Blunt over encouraging.** Say what is wrong. Skip the praise.
- **One module at a time.** After each module, stop so the author can read it and run
  `checks.jl` before moving on. Do not chain three modules together in one pass.
- Explain *why* a formula has the shape it does, not just what the code does. Docstrings in
  `ewma.jl` and `realized_vol.jl` set the expected depth — match it.

## Branch rules

Work happens on `volatility-forecasting`. **Do not merge to `main` until the author
confirms they understand every module on the branch** — their explicit rule.

On merge, `README.md` needs updating: it currently claims "no GARCH/ML/GPU — those are
later projects", which will be false.
