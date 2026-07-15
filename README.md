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
