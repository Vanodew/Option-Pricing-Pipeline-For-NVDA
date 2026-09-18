#v0 options-pricing pipeline for NVDA.
#run with:  julia --project=. main.jl

include("src/data.jl")
include("src/volatility.jl")
include("src/black_scholes.jl")

using .DataFetch
using .Volatility
using .BlackScholes
using DataFrames

function main()
    symbol = "NVDA"

    #1. historical daily closes, cached in data/prices.csv after the first fetch.
    df = fetch_price_history(symbol)

    #2. annualized historical vol from the daily log returns.
    returns = log_returns(df.close)
    sigma = annualized_volatility(returns)

    #3. black-scholes inputs. the strike is rounded to the nearest $5 just so it looks like
    #a strike thats actually listed, and r is roughly the 3-month t-bill yield.
    S = Float64(last(df.close))   # spot = most recent close
    K = round(S / 5) * 5          # roughly at-the-money strike
    r = 0.045                     # risk-free rate (annualized, cont. comp.)
    T = 0.25                      # time to expiry in years (~3 months)

    price = bs_call_price(S, K, r, sigma, T)
    d1, d2 = bs_d1_d2(S, K, r, sigma, T)

    #4. print the result together with every input that produced it.
    println("=== $symbol European call, Black-Scholes v0 ===")
    println("Data window     : $(first(df.date)) to $(last(df.date)) ($(nrow(df)) closes, $(length(returns)) returns)")
    println("Spot S          : $(round(S, digits=2))")
    println("Strike K        : $(round(K, digits=2))")
    println("Risk-free r     : $r")
    println("Expiry T        : $T years")
    println("Volatility      : $(round(sigma, digits=4))  ($(round(100 * sigma, digits=2))% annualized)")
    println("d1, d2          : $(round(d1, digits=4)), $(round(d2, digits=4))")
    println("Call price      : $(round(price, digits=2))")
end

main()
