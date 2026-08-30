# Option Pricing Pipeline for NVDA (v0)

Minimal end-to-end options-pricing prototype in Julia:

1. Pull ~2 years of daily NVDA closes (Yahoo Finance chart endpoint, cached to `data/prices.csv`).
2. Compute daily log returns and their sample standard deviation.
3. Annualize with sqrt(252) to get historical volatility.
4. Price a European call with Black-Scholes and print the result with all inputs.

Deliberately simple: constant volatility, no dividends, no GARCH/ML/GPU. Those are later projects.

## Layout

```
main.jl               entry point: fetch -> vol -> price -> report
checks.jl             hand-verifiable sanity checks for each core piece
src/data.jl           DataFetch: Yahoo fetch + CSV cache
src/volatility.jl     Volatility: log_returns, annualized_volatility
src/black_scholes.jl  BlackScholes: bs_d1_d2, bs_call_price
data/prices.csv       cached price history
```

## Setup and run

```sh
julia --project=. -e "using Pkg; Pkg.instantiate()"   # one-time dependency install
julia --project=. checks.jl                            # sanity checks
julia --project=. main.jl                              # full pipeline
```

Pass `force_refresh=true` to `fetch_price_history` in `main.jl` to re-download prices instead of using the cache.

## Model assumptions worth knowing for interviews

- Log returns are treated as i.i.d. normal; sqrt(252) annualization relies on the independence part.
- Volatility is *historical* (backward-looking). Real option markets trade on *implied* vol.
- Black-Scholes here assumes no dividends, constant r and sigma, European exercise.


**Research question:** For NVDA over ~2015–2025, does a gradient-boosted-tree forecast of 21-day-ahead realized volatility beat EWMA and GARCH(1,1) out-of-sample under walk-forward evaluation — and does any edge survive a breakdown by volatility regime?

**Deliverable:** public repo + a working paper (15–25 pages) posted to SSRN. Not peer-reviewed, and say so.

**Target dates:** empirical core by end of Aug → expansion Sept → robustness Oct → writing Nov → ship Dec.

**Standing rule:** every time a number surprises you on the upside, look for the leak before you celebrate.

---

## Phase 0 — Audit what already exists

- [ ] Run the full test suite, confirm all 12 tests still pass, record the runtime
- [ ] Confirm the `h0_window` seeding fix actually landed (GARCH must NOT seed from full-sample variance)
- [ ] Confirm the GARCH/EWMA variance-indexing alignment fix landed (the off-by-one)
- [ ] Write down, in the repo README, exactly which day each model's variance output refers to — one sentence, unambiguous
- [ ] Fix the expired-cert / HTTP dependency issue so data pulls work, or commit to a static CSV and move on
- [ ] Freeze the data: pull once, save to CSV, commit it. Every result from here reproduces from that file
- [ ] `git log` sanity check — know what's committed vs sitting uncommitted in your working tree

---

## Phase 1 — Data and returns

- [ ] Daily OHLCV for NVDA, ~2015 to present, from a single source
- [ ] Compute daily log returns
- [ ] Plot the return series. Confirm you can _see_ volatility clustering with your own eyes
- [ ] Check for gaps, splits, and zero-return days; document how you handled each
- [ ] Record the exact date range and row count — this goes in the paper's data section

---

## Phase 2 — Define the target (the highest-risk step)

- [ ] Define the target: annualized realized vol over days t+1 … t+21, computed from log returns
- [ ] Implement it
- [ ] **Hand-verify on 3 rows.** Pick three dates, compute the target manually in a spreadsheet, match to the code
- [ ] Confirm the last 21 rows of the dataset have no target (they can't — the future isn't there yet)
- [ ] Write a test that fails if the target ever uses a return from day ≤ t
- [ ] Note the overlap problem in your notes: 21-day windows share 20 of 21 days, so ~2,500 rows ≈ ~120 independent observations

---

## Phase 3 — EWMA baseline

- [ ] Implement EWMA variance recursion with λ = 0.94
- [ ] Confirm it's genuinely causal — forecast for t+1 uses nothing after t
- [ ] Produce forecasts for the full sample
- [ ] Score it on MSE and QLIKE. **This is the number everything else must beat**
- [ ] Sanity check: does EWMA vol track the visible spikes in the return plot?
- [ ] Optional: try a couple of λ values, but pick 0.94 for the headline and say why

---

## Phase 4 — GARCH(1,1)

- [ ] Verify the hand-rolled likelihood against a library fit (Python `arch` is fine as a cross-check)
- [ ] Confirm fitted params on full sample match what you recorded: long-run vol ~52.5% vs ~49.6% realized
- [ ] Add a convergence check: flag any fit that fails, hits max iterations, or lands with α+β ≥ 1
- [ ] Add a parameter-recovery test: simulate from known (ω, α, β), refit, confirm you get them back
- [ ] Implement multi-step-ahead forecasting to a 21-day horizon
- [ ] Confirm the EWMA-equivalence test still holds (GARCH with ω=0, α+β=1 should reproduce EWMA)
- [ ] Score full-sample forecasts on MSE and QLIKE
- [ ] Be able to explain, out loud, what α and β mean and what α+β measures

---

## Phase 5 — Features for the GBT

- [ ] Realized vol at 5, 10, 21, 63 day lookbacks
- [ ] Recent returns and absolute returns (a few lags)
- [ ] High–low range (Parkinson-style), which carries info close-to-close misses
- [ ] Volume relative to its own trailing average
- [ ] A leverage-effect feature (something asymmetric in the sign of recent returns)
- [ ] **Cap the feature count at 10.** More than that on ~120 independent observations is overfitting with extra steps
- [ ] Write one test per feature confirming it uses no data after day t
- [ ] Plot each feature against the target — anything with suspiciously high correlation is a leak, not a discovery

---

## Phase 6 — Gradient-boosted trees

- [ ] Fit a first GBT on a simple time-ordered split just to get it running
- [ ] Set up an **inner validation split** inside the training window for hyperparameter tuning
- [ ] Tune depth, learning rate, and number of trees on that inner split only — never on test
- [ ] Record the chosen hyperparameters and how you picked them
- [ ] Extract feature importances; check they're economically sensible, not random
- [ ] Score against EWMA and GARCH on the same split

---

## Phase 7 — Walk-forward harness

- [ ] Build the rolling loop: fit on data through date T, predict forward, roll, refit, repeat
- [ ] **Insert a 21-day embargo gap** between the end of each training window and the start of its test window
- [ ] Refit GARCH inside every window (no reusing a full-sample fit)
- [ ] Log convergence status per window; report how many failed
- [ ] Retune GBT hyperparameters inside each window, or fix them once and state that you did
- [ ] Decide expanding vs rolling window; justify the choice in one sentence
- [ ] Run all three models through the identical harness so the comparison is apples-to-apples
- [ ] Store per-day forecasts from every model to disk — you'll need them repeatedly

---

## Phase 8 — Evaluation

- [ ] Compute MSE for all three models across the full out-of-sample period
- [ ] Compute QLIKE for all three
- [ ] Build the headline results table: model × loss function
- [ ] **Diebold–Mariano test** on each pairwise comparison
- [ ] Use HAC / Newey–West standard errors — forecast errors are autocorrelated by construction
- [ ] State plainly whether differences are statistically significant, and don't oversell if they aren't
- [ ] If MSE and QLIKE disagree, write a paragraph on why — that disagreement is a finding
- [ ] Read Patton (2011) on loss functions robust to a noisy volatility proxy; cite it for why these two

---

## Phase 9 — Regime breakdown and robustness

- [ ] Define volatility regimes (e.g. terciles of trailing realized vol)
- [ ] Re-score every model within each regime
- [ ] Answer directly: does GBT win everywhere, only in calm periods, or only in crises?
- [ ] Break results out by year to check the edge isn't one lucky period
- [ ] Test horizon sensitivity: rerun at 5-day and 10-day targets
- [ ] Test λ sensitivity for EWMA
- [ ] Note explicitly: does any edge survive transaction costs if traded? (Even a qualitative answer.)

---

## Phase 10 — Expand beyond NVDA

- [ ] Pick 10–20 liquid names plus SPY, spanning sectors and vol levels
- [ ] Run the identical pipeline unchanged across all of them
- [ ] Build a cross-sectional results table: which models win, on which names
- [ ] Check whether NVDA was unusual — it's a high-vol, high-persistence name, and it may not generalize
- [ ] Report per-name results, not just an average

---

## Phase 11 — Write the paper

- [ ] Abstract (write it last, ~150 words)
- [ ] Introduction: the question, why it matters, what you find
- [ ] Literature: 10–15 papers. State clearly that this is a replication-style study, not a novel result
- [ ] Data: source, range, cleaning decisions
- [ ] Methodology: EWMA, GARCH, GBT, the walk-forward design, the embargo gap
- [ ] Results: tables and figures, with DM tests
- [ ] Regime and robustness section
- [ ] **Limitations, written honestly** — overlapping windows, small effective sample, single-source data, no transaction costs, proxy noise
- [ ] Conclusion
- [ ] Figures: return series with clustering, forecast vs realized overlay, loss-by-regime bar chart
- [ ] Reread hunting for any claim the results don't actually support; delete or soften each one

---

## Phase 12 — Ship

- [ ] Clean the repo: README, reproduction instructions, `Project.toml`, one-command run
- [ ] Confirm a fresh clone reproduces every number in the paper
- [ ] Post to SSRN. Ask the research department head for help with the submission mechanics
- [ ] Link the paper from the repo and the repo from the paper
- [ ] Add to resume as **working paper / preprint** — never as "published"
- [ ] Uncomment the pipeline entry in `resume_final.tex`, keeping only bullets that describe code that actually runs
- [ ] Email Prof. Asgher Ali: lead with the optimal-control ↔ optimal-execution overlap or deep BSDE, attach the ADE paper and this study, ask about supervising project two

---

## Interview defense — rehearse these cold

- [ ] "How do you know you didn't leak the future?"
- [ ] "Why 21 days?"
- [ ] "Why QLIKE as well as MSE?"
- [ ] "What do α and β mean, and what does α+β tell you?"
- [ ] "Your effective sample size is what, exactly?"
- [ ] "Why should I believe this difference isn't noise?"
- [ ] "What would change your conclusion?"
- [ ] "Hasn't this been done before?" (Answer: yes. Say so.)
