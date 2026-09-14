# CLAUDE.md

Project instructions for Claude Code. Read before making changes.

## Scope — read this first

This project was deliberately cut down on 2026-08-24. It had grown into a 13-phase plan
ending in a published paper; almost none of it was built, and the size of the plan was the
main thing stopping progress. What follows is the reduced version.

On 2026-09-12 the author re-expanded it by exactly one model: gradient-boosted trees. That
was an explicit decision, not drift. **It is the only re-expansion.** Everything under
"Parked" below stays parked, and the next addition needs the same explicit call.

**The project is now:** can GARCH(1,1) or a gradient-boosted tree beat EWMA at forecasting
NVDA's 21-day-ahead realized variance, out-of-sample, under walk-forward evaluation?

Three models. One dataset. One honest answer.

### Done looks like this

- [x] A results table: EWMA vs GARCH vs GBT, scored on QLIKE and MSE, out-of-sample
- [x] One chart: forecast vs realized variance over the test period
- [x] A README stating the question, the method, the answer, and the limitations
- [x] `checks.jl` green, and a fresh clone reproduces every number from the committed CSV
- [ ] Merged to `main`

When those five boxes are ticked, the project is **finished** — not paused, finished. Any
further ambition starts as a new decision, not as a continuation.

### Parked, not deleted

The full research plan lives in `README.md` (Phases 0–12). It is **out of scope** and stays
untouched until the five boxes above are done. Specifically out of scope right now:

- HAR (Corsi 2009) as a *fitted model*. Multi-horizon trailing RV (5/21/63) appears as GBT
  *features*; that is not a HAR regression, and none gets estimated or scored.
- Any ticker other than NVDA
- The SSRN working paper, literature review, and regime/robustness sections
- Transaction-cost overlays
- The option-pricing strand: Greeks, Monte Carlo, GPU, Asian payoffs

If work drifts toward any of these, stop and say so rather than following it.

## Layout

```
experiment.jl           entry point for the study: walk-forward -> table -> DM tests
chart.jl                forecast-vs-realized SVG, redrawn from the results CSV
checks.jl               hand-verifiable sanity checks, one per core piece
main.jl                 the original BS v0 pipeline: fetch -> vol -> price -> report
src/data.jl             DataFetch: Yahoo chart endpoint + CSV cache
src/volatility.jl       Volatility: log_returns, annualized_volatility
src/black_scholes.jl    BlackScholes: bs_d1_d2, bs_call_price
src/realized_vol.jl     RealizedVol: forward (target) and trailing (feature) RV
src/ewma.jl             EWMA: RiskMetrics lambda=0.94 variance path, h-step rule
src/garch.jl            Garch: GARCH(1,1) by hand-coded MLE, h-step forecast
src/features.jl         Features: six close-derived GBT features + valid mask
src/loss.jl             Loss: qlike, mse, mean_loss
src/walk_forward.jl     WalkForward: expanding-window splits + per-window refits
src/dm.jl               DieboldMariano: newey_west_lrv, dm_test
data/prices_10y.csv     frozen sample: 2512 closes, 2016-07-19 -> 2026-07-16
results/walkforward.csv per-day forecasts, committed so results reproduce
```

`docs/original-plan.md` holds the pre-2026-09-13 README and its 13-phase plan, unedited.
It is **deliberately untracked** (see `.gitignore`) — kept on the author's machine, not
published. The same content is in history at `git show a2e458a:README.md`, so nothing is
lost if the local copy goes.

`data/prices_10y.csv` is close-only. Do not re-pull the data, and do not spend time on the
Julia TLS handshake failure; the frozen CSV is the data source. The cost of close-only is
now a documented limitation rather than a non-issue: GBT sees no high–low range and no
volume, so it works from the same information EWMA does. See the GBT feature constraint
above.

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

## The three models

- **EWMA** — the baseline. RiskMetrics λ = 0.94 for daily data. In precisely because it has
  no fitted parameters, is a few lines of code, and is genuinely hard to beat. λ is a
  convention, not a law.
- **GARCH(1,1)** — the challenger, and the thing that closes the author's self-identified
  biggest gap (time-series). Written by hand — likelihood, optimizer setup, robust standard
  errors — and cross-checked against `ARCHModels.jl`. "I called a library" and "I derived
  and fitted the likelihood" are different interview answers.
- **Gradient-boosted trees** (`EvoTrees.jl`) — the nonlinear challenger, added 2026-09-12.
  The learning here is feature construction and leak-free target alignment, not tree
  boosting internals, so the library supplies the fitting. It earns its place only if it
  beats EWMA on QLIKE under the identical harness.

**GBT feature constraint — read before designing features.** `data/prices_10y.csv` is
close-only, and re-pulling data is blocked (see Layout). So GBT sees nothing EWMA and GARCH
do not: no high-low range, no volume, no order flow. Features are close-derived only —
trailing RV at several windows, lagged returns, and optionally the EWMA/GARCH forecasts
themselves. That is a defensible setup, but it caps what GBT can plausibly add, and the
README must say so rather than dressing up a null result as a surprise.

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

Current branch is `loss-function`, **9 commits ahead of `main`** and not yet merged. Only
the first group below is on `main`; everything else is on the branch. Working tree is
clean — nothing untracked.

**The study is finished and written up. The only thing left is the merge, and it is gated
on the author's module walkthrough (see Branch and commit rules).**

Built and verified:
- BS v0 pipeline
- `realized_vol.jl` — forward/trailing RV, with the leakage probe
- `ewma.jl` — RiskMetrics EWMA baseline
- `garch.jl` — GARCH(1,1) by hand-coded MLE; matches `ARCHModels.jl` to ~7 s.f. on ω, β, α,
  μ and logL; Huber sandwich SEs match to 2e-6 (robust SEs ≈ 2× naive Hessian, halving the
  t-stats on α and β)
- `h0_window` walk-forward-safe seeding, and the `ConditionalVariance` / `OneStepForecast`
  timing types
- `loss.jl` — QLIKE, MSE, `mean_loss`; checks (o)(p)(q)
- `walk_forward.jl` — expanding-window splits with the 21-day embargo, per-window GARCH
  refit under `h0_window=train_end`, convergence logged, forecasts written to `results/`;
  checks (r)(s)
- `dm.jl` — Diebold-Mariano with Newey-West/Bartlett HAC and the HLN small-sample
  correction; checks (t)(u)(v)
- `features.jl` — six close-derived GBT features, NaN + `valid` flag rather than filled;
  check (w) includes the leakage probe
- `walk_forward.jl` GBT arm — per-window EvoTrees refit on `log(target)`, hyperparameters
  as constants, `seed` pinned (`rng` is ignored in EvoTrees 0.18.7); check (x)
- `experiment.jl` — three-model table plus all six DM comparisons, one command
- `chart.jl` — forecast-vs-realized SVG, written directly, no plotting dependency
- `README.md` — rewritten around the result. Reviewed 2026-09-14: both versions kept, the
  old one local-only and untracked
- 24 checks (a–x) passing, ~38s

**The answer, on 1469 out-of-sample days across 71 windows, none failed:** QLIKE — EWMA
0.2444, GARCH 0.2064, GBT 0.2868. Nothing beats EWMA significantly. The only significant
difference anywhere is GARCH over GBT on QLIKE (t = −2.20, p = 0.028). The HAC correction
is doing the work: the Newey-West SE runs 2.2–3.6× the naive one, and under a naive SE
GARCH over EWMA would read t = 3.94, p < 0.0001.

`main.jl` is still the v0 Black-Scholes script and is left that way on purpose — it is a
working deliverable and the README describes it as the project's origin. `experiment.jl`
is the entry point for the study.

`git status` is the authority on what is actually committed — check it before assuming a
module is safe.

## Build order

1. ~~**Commit `src/garch.jl` + `checks.jl` + this file.**~~ Done — merged to `main` in PR #1
   on 2026-08-31. The author has not yet read through `garch.jl`; do not add to or extend
   that module until they have.
2. ~~**`src/loss.jl`**~~ — done on `loss-function`, checks (o)(p)(q).
3. ~~**`src/walk_forward.jl`**~~ — done on `loss-function`, checks (r)(s).
4. ~~**Run it on real data.**~~ Done — `experiment.jl`, `chart.jl`.
5. ~~**GBT as a third model.**~~ Done — `features.jl` + the `walk_forward.jl` GBT arm.
6. ~~**Diebold-Mariano with HAC**, then rewrite `README.md`.~~ Done — `dm.jl`, README.
7. **Merge.** Blocked on the author's walkthrough of every module on the branch — their
   rule, not a formality. Outstanding: `garch.jl` (debt from PR #1), `dm.jl`,
   `features.jl`, the `walk_forward.jl` GBT arm, `experiment.jl`, `chart.jl`, and checks
   (s)–(x).

One module at a time, stopping after each.

## Open decisions — do not silently pick these

Scoping down removed most of these. Two remain, and they are the author's calls:

1. ~~**Expanding vs rolling training window.**~~ Resolved 2026-09-12: **expanding**. Training
   always starts at index 1. This matches what `walk_forward` already did; the decision is
   now recorded rather than implicit. It applies to GBT too.
2. **What to do when a GARCH window fails to converge** — drop the window, carry the previous
   fit forward, or exclude the day.
3. **Variance vs volatility units for scoring.** The code and the convention above say
   variance; the original plan's Phase 2 (`git show a2e458a:README.md`) says "annualized
   realized vol." The code's convention won in practice and the README reports in
   variance. The
   code's convention governs until the author says otherwise.

## Hand-code vs library

The point of this project is understanding the estimator.

- **By hand:** EWMA; the GARCH likelihood, its optimization and its standard errors; the
  walk-forward split; the loss functions; the Diebold-Mariano statistic.
- **Library or agent-written is fine:** plotting, CSV parsing, HTTP, `Project.toml`, README
  plumbing. `Optim.jl` supplies only the search, not the likelihood. `EvoTrees.jl` supplies
  the boosting — the hand-built part of the GBT strand is the feature matrix and its
  alignment, which is where the leaks live.

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

`volatility-forecasting` is merged and `main` is the current branch. Open a fresh branch for
the next module rather than committing research code straight to `main`. **Nothing is
committed until the author reviews it**, and **do not merge to `main` until the author
confirms they understand every module on the branch** — their explicit rule. The PR #1 merge
landed `garch.jl` on `main` ahead of that walkthrough, so the review debt on it is real and
outstanding, not waived.

The merge is step 5 of the build order and is the finish line, not an afterthought. On merge
`README.md` gets rewritten around the actual result, replacing both the stale v0 text ("no
GARCH/ML/GPU — those are later projects", already false) and the parked 13-phase plan.
