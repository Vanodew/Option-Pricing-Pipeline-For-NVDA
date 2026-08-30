# CLAUDE.md

Project instructions for Claude Code. Read before making changes.

## Scope — read this first

This project was deliberately cut down on 2026-08-24. It had grown into a 13-phase plan
ending in a published paper; almost none of it was built, and the size of the plan was the
main thing stopping progress. What follows is the reduced version. **Do not re-expand it.**

**The project is now:** does GARCH(1,1) beat EWMA at forecasting NVDA's 21-day-ahead
realized variance, out-of-sample, under walk-forward evaluation?

Two models. One dataset. One honest answer.

### Done looks like this

- [ ] A results table: EWMA vs GARCH, scored on QLIKE and MSE, out-of-sample
- [ ] One chart: forecast vs realized variance over the test period
- [ ] A README stating the question, the method, the answer, and the limitations
- [ ] `checks.jl` green, and a fresh clone reproduces every number from the committed CSV
- [ ] Merged to `main`

When those five boxes are ticked, the project is **finished** — not paused, finished. Any
further ambition starts as a new decision, not as a continuation.

### Parked, not deleted

The full research plan lives in `README.md` (Phases 0–12). It is **out of scope** and stays
untouched until the five boxes above are done. Specifically out of scope right now:

- Gradient-boosted trees / `EvoTrees.jl`
- HAR (Corsi 2009)
- Any ticker other than NVDA
- The SSRN working paper, literature review, and regime/robustness sections
- Transaction-cost overlays
- The option-pricing strand: Greeks, Monte Carlo, GPU, Asian payoffs

If work drifts toward any of these, stop and say so rather than following it.

## Layout

```
main.jl                 entry point: fetch -> vol -> price -> report
checks.jl               hand-verifiable sanity checks, one per core piece
src/data.jl             DataFetch: Yahoo chart endpoint + CSV cache
src/volatility.jl       Volatility: log_returns, annualized_volatility
src/black_scholes.jl    BlackScholes: bs_d1_d2, bs_call_price
src/realized_vol.jl     RealizedVol: forward (target) and trailing (feature) RV
src/ewma.jl             EWMA: RiskMetrics lambda=0.94 variance path, h-step rule
src/garch.jl            Garch: GARCH(1,1) by hand-coded MLE, h-step forecast
data/prices_10y.csv     frozen sample: 2512 closes, 2016-07-19 -> 2026-07-16
```

`data/prices_10y.csv` is close-only. That is **sufficient** for this scope — high–low range
and volume were only ever needed for GBT features, which are out. Do not re-pull the data,
and do not spend time on the Julia TLS handshake failure; the frozen CSV is the data source.

## Setup

```sh
julia --project=. -e "using Pkg; Pkg.instantiate()"   # one-time, reproduces pinned env
julia --project=. checks.jl                            # must print "All sanity checks passed."
julia --project=. main.jl                              # full pipeline
```

`Manifest.toml` is committed on purpose — it pins exact versions so a fresh clone rebuilds
the identical environment. Any dependency bump is a reviewable event with its own commit.

## Conventions that are not negotiable

- **Forecast horizon h = 21 trading days.** One month. The target window and the option
  tenor. Expect to defend the choice out loud.

- **Fit and score in variance units**, not volatility. Square root only at the reporting
  stage. QLIKE on volatility is a different loss with different properties. Units are the
  top silent-failure mode here — variance vs volatility, daily vs annualized, one stray
  √252 — and a wrong result still looks plausible.

- **Log returns, not simple returns.** They add cleanly across time.

- **No look-ahead.** A feature at index `t` may use returns only up to and including day
  `t`; the target at index `t` uses days `t+1 .. t+h`. `trailing_realized_variance` and
  `forward_realized_variance` never overlap at a given index. `checks.jl` case (e) is the
  probe: perturb the past and the target must not move; perturb the future and it must.
  This extends to model *initialization*: `garch.jl` seeds its variance recursion from
  `var(returns)` by default, which is fine for a one-shot fit but leaks in a backtest.
  Under walk-forward, always pass `h0_window=T_train`. Case (l) is that probe.

- **21-day embargo gap** between the end of each training window and the start of its test
  window. With a 21-day forward target, the last training labels are computed from days
  that live inside the test period. Train → skip 21 days → test.

- **Timing conventions live in the type.** GARCH returns `ConditionalVariance` (`s[t]` known
  at `t-1`) or `OneStepForecast` (`s[t]` known at `t`, EWMA's convention); they differ by one
  index and refuse to be compared. Unwrap via `variances()` only where the alignment is a
  deliberate decision.

- **Report QLIKE and MSE, both.** QLIKE = `RV/F - ln(RV/F) - 1`. QLIKE is the headline: it
  is built for variance and punishes *under*-prediction harder, which is the error that
  costs money. MSE alone is dominated by a handful of crisis days, so the comparison ends up
  decided by March 2020. Both are the losses that stay honest when the "truth" is itself a
  noisy proxy (Patton 2011) — cite it for *why these two*. **If they disagree, that is a
  finding**, not an inconvenience, and it gets a paragraph in the README.

- **Overlapping windows.** Consecutive 21-day forward windows share 20 returns, so errors
  are strongly autocorrelated. The effective sample is roughly **120 independent
  observations**, not 2,500 — and walk-forward leaves perhaps 50–70. Any significance test
  needs HAC / Newey-West standard errors.

- Every module gets sanity checks in `checks.jl` whose correct answer is derivable by hand.
  Pencil-and-paper cases, not property tests, not fuzzing.

- **Standing rule: when a number surprises you on the upside, hunt for the leak before you
  celebrate.** Both real bugs found so far (full-sample `h0` seeding, and the GARCH/EWMA
  off-by-one) made the results look *better*. Leakage never hurts your numbers.

## The two models

- **EWMA** — the baseline. RiskMetrics λ = 0.94 for daily data. In precisely because it has
  no fitted parameters, is a few lines of code, and is genuinely hard to beat. λ is a
  convention, not a law.
- **GARCH(1,1)** — the challenger, and the thing that closes the author's self-identified
  biggest gap (time-series). Written by hand — likelihood, optimizer setup, robust standard
  errors — and cross-checked against `ARCHModels.jl`. "I called a library" and "I derived
  and fitted the likelihood" are different interview answers.

## Evaluation design

- **Never a random split.** Time-ordered only.
- **Walk-forward**: fit through date T, predict forward, roll, refit, repeat. It mimics
  deployment and is what makes this a study rather than a homework exercise.
- **Refit GARCH inside every window** — never reuse a full-sample fit. Log convergence
  status per window and report how many failed; a few garbage fits (α+β pinned at 1)
  silently poison the aggregate score.
- **Identical harness for both models**, or the comparison is not a comparison.
- **Persist per-day forecasts to disk.** They get reused constantly.
- **Diebold-Mariano with HAC standard errors** as the final step — the answer to "is this
  difference real or luck." This is the one piece of statistics that stays in scope.

**Acceptable outcomes, agreed in advance so there is no incentive to fish:** given the
effective sample size, "no statistically significant difference" is the *likely* answer and
counts as a complete result. The README should be written so that outcome reads as a
finding, not a failure.

## State

Built and verified:
- BS v0 pipeline (on `main`)
- `realized_vol.jl` — forward/trailing RV, with the leakage probe
- `ewma.jl` — RiskMetrics EWMA baseline
- `garch.jl` — GARCH(1,1) by hand-coded MLE; matches `ARCHModels.jl` to ~7 s.f. on ω, β, α,
  μ and logL; Huber sandwich SEs match to 2e-6 (robust SEs ≈ 2× naive Hessian, halving the
  t-stats on α and β)
- `h0_window` walk-forward-safe seeding, and the `ConditionalVariance` / `OneStepForecast`
  timing types
- 14 checks (a–n) passing, ~54s cold

Not built: the loss functions, the walk-forward harness, the results table, the chart.

`git status` is the authority on what is actually committed — check it before assuming a
module is safe.

## Build order

1. **Commit `src/garch.jl` + `checks.jl` + this file.** Currently untracked; it is the
   best-verified work in the repo and exists in one place only.
2. **`src/loss.jl`** — QLIKE and MSE, with hand-verifiable cases in `checks.jl`.
3. **`src/walkforward.jl`** — rolling loop, 21-day embargo, per-window GARCH refit with
   `h0_window`, forecasts persisted to disk. Everything is blocked on this.
4. **Run it.** Produce the results table and the forecast-vs-realized chart.
5. **Diebold-Mariano with HAC**, then rewrite `README.md` around the actual answer and merge.

One module at a time, stopping after each.

## Open decisions — do not silently pick these

Scoping down removed most of these. Three remain, and they are the author's calls:

1. **Expanding vs rolling training window** in the walk-forward harness.
2. **What to do when a GARCH window fails to converge** — drop the window, carry the previous
   fit forward, or exclude the day.
3. **Variance vs volatility units for scoring.** The code and the convention above say
   variance; `README.md` Phase 2 says "annualized realized vol." Genuinely unresolved. The
   code's convention governs until the author says otherwise.

## Hand-code vs library

The point of this project is understanding the estimator.

- **By hand:** EWMA; the GARCH likelihood, its optimization and its standard errors; the
  walk-forward split; the loss functions; the Diebold-Mariano statistic.
- **Library or agent-written is fine:** plotting, CSV parsing, HTTP, `Project.toml`, README
  plumbing. `Optim.jl` supplies only the search, not the likelihood.

When a library would hide the thing being learned, say so and write it out instead.

## How to work with the author

- **Tutor mode, not a code vending machine.** Explain the concept → give the function name
  and signature → let the author implement → review → ask a question or two to confirm
  understanding before moving on. Explain *why* a formula has the shape it does. Docstrings
  in `ewma.jl` and `realized_vol.jl` set the expected depth — match it.
- **Blunt over encouraging.** Say what is wrong. Skip the praise.
- **One module at a time.** Stop after each so the author can read it and run `checks.jl`.
  Never chain three modules together in one pass.
- **Watch for scope creep and name it.** This project has been cut down once already; the
  failure mode is adding ambition faster than code. Suggesting extra models, extra tickers,
  or extra analysis is a regression, not helpfulness.
- **Unrelated concerns get their own prompt and their own commit.** A dependency bump does
  not ride along with research code.
- **Audit before building.** When resuming: report the real state with file-and-line
  evidence, run the suite, check `git log` and `git status`, and write no code in that pass.
- Tests must pass, and be reported as passing, before reporting back.
- Plans are wanted as tickable checklists in execution order.

## Branch and commit rules

Work happens on `volatility-forecasting`. **Nothing is committed until the author reviews
it**, and **do not merge to `main` until the author confirms they understand every module on
the branch** — their explicit rule.

The merge is step 5 of the build order and is the finish line, not an afterthought. On merge
`README.md` gets rewritten around the actual result, replacing both the stale v0 text ("no
GARCH/ML/GPU — those are later projects", already false) and the parked 13-phase plan.
