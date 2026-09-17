# Walk-forward experiment: EWMA vs GARCH(1,1) vs GBT on NVDA's 21-day realized
# variance.
# Run with:  julia --project=. experiment.jl
#
# Reads the frozen CSV directly, not through DataFetch -- that points at a
# different cache path and can hit the network.

include("src/volatility.jl")
include("src/realized_vol.jl")
include("src/ewma.jl")
include("src/garch.jl")
include("src/loss.jl")
include("src/features.jl")
include("src/walk_forward.jl")
include("src/dm.jl")

using .Volatility
using .RealizedVol
using .EWMA
using .Garch
using .Loss
using .WalkForward
using .DieboldMariano

using CSV
using DataFrames
using Statistics

const DATA_PATH = "data/prices_10y.csv"
const OUT_PATH = "results/walkforward.csv"

const H = 21            # forecast horizon, trading days -- the project convention
const MIN_TRAIN = 1000  # ~4 years before the first forecast is made
const TEST_SIZE = 21    # refit monthly; also the embargo length

# Mask is passed in, not derived here, so every model is scored on the same days.
function score_column(realized, forecast, mask)
    rv = collect(skipmissing(realized[mask]))
    fc = collect(skipmissing(forecast[mask]))
    length(rv) == length(fc) || error("mask left a ragged pair: $(length(rv)) vs $(length(fc))")
    return (
        n = length(rv),
        qlike = mean_loss(rv, fc, qlike),
        mse = mean_loss(rv, fc, mse),
    )
end

# Reporting only -- fitting and scoring stay in variance.
ann_vol(total_var_h) = sqrt(total_var_h / H * 252)

fmt(x; d=6) = string(round(x, sigdigits=d))

function main()
    df = CSV.read(DATA_PATH, DataFrame)
    returns = log_returns(df.close)

    println("=== NVDA walk-forward: EWMA vs GARCH(1,1) vs GBT ===")
    println("Data        : $(first(df.date)) to $(last(df.date))  " *
            "($(nrow(df)) closes, $(length(returns)) returns)")
    println("Horizon h   : $H trading days (target = total variance over t+1..t+h)")
    println("Window      : expanding, min_train=$MIN_TRAIN, test_size=$TEST_SIZE, " *
            "embargo=$H")
    println()

    results = walk_forward(
        returns;
        min_train = MIN_TRAIN,
        test_size = TEST_SIZE,
        h = H,
        output_path = OUT_PATH,
    )

    # A day any model is missing is dropped for all of them.
    has_target = .!ismissing.(results.realized)
    has_all = .!ismissing.(results.ewma_hstep) .&
              .!ismissing.(results.garch_hstep) .&
              .!ismissing.(results.gbt_hstep)
    mask = has_target .& has_all

    n_rows = nrow(results)
    n_no_target = count(.!has_target)
    n_garch_missing = count(ismissing, results.garch_hstep)
    n_gbt_missing = count(ismissing, results.gbt_hstep)
    n_scored = count(mask)

    println()
    println("Rows produced          : $n_rows")
    println("Dropped, no target     : $n_no_target  (last $H days have no forward window)")
    println("Dropped, GARCH missing : $n_garch_missing  (non-converged windows)")
    println("Dropped, GBT missing   : $n_gbt_missing  (no feature history, or window skipped)")
    println("Scored on              : $n_scored days, identical set for all three models")

    if n_garch_missing > 0
        println()
        println("!! Some GARCH windows did not converge. Open decision 2 (drop the window /")
        println("!! carry the previous fit forward / exclude the day) is now live and unresolved.")
        println("!! The table below excludes those days from BOTH models, which is the")
        println("!! 'exclude the day' branch -- it has not been agreed. Treat as provisional.")
    end

    rv = collect(skipmissing(results.realized[mask]))
    models = (
        ("EWMA", collect(skipmissing(results.ewma_hstep[mask]))),
        ("GARCH", collect(skipmissing(results.garch_hstep[mask]))),
        ("GBT", collect(skipmissing(results.gbt_hstep[mask]))),
    )
    scores = [(name, score_column(results.realized, col, mask)) for (name, col) in
              (("EWMA", results.ewma_hstep), ("GARCH", results.garch_hstep), ("GBT", results.gbt_hstep))]

    println()
    println("=== Results: $H-day total variance, out-of-sample ===")
    println(rpad("Model", 10), lpad("n", 7), lpad("QLIKE", 14), lpad("MSE", 16))
    for (name, s) in scores
        println(rpad(name, 10), lpad(string(s.n), 7), lpad(fmt(s.qlike), 14), lpad(fmt(s.mse), 16))
    end

    println()
    println("Lower is better for both losses.")
    qlike_winner = scores[argmin([s.qlike for (_, s) in scores])][1]
    mse_winner = scores[argmin([s.mse for (_, s) in scores])][1]
    println("QLIKE favours : $qlike_winner")
    println("MSE favours   : $mse_winner")
    if qlike_winner != mse_winner
        println("The two losses DISAGREE on the winner. That is a finding,")
        println("not an inconvenience, and it gets its own paragraph in the README.")
    end

    println()
    println("=== Diebold-Mariano, HAC standard errors (q = h-1 = $(H - 1)) ===")
    println("Positive statistic means the FIRST model is worse. Statistic and p-value are")
    println("the Harvey-Leybourne-Newbold small-sample versions.")
    println()
    println(rpad("Comparison", 18), rpad("Loss", 7), lpad("stat", 8), lpad("p", 10),
            lpad("HAC/naive se", 15), "   verdict at 5%")

    for (a, b) in (("EWMA", "GARCH"), ("EWMA", "GBT"), ("GARCH", "GBT"))
        fa = models[findfirst(m -> m[1] == a, models)][2]
        fb = models[findfirst(m -> m[1] == b, models)][2]
        for (lname, lf) in (("QLIKE", qlike), ("MSE", mse))
            la = [lf(rv[i], fa[i]) for i in eachindex(rv)]
            lb = [lf(rv[i], fb[i]) for i in eachindex(rv)]
            d = dm_test(la, lb; h=H)
            naive = sqrt(var(la .- lb) / length(la))
            println(
                rpad("$a vs $b", 18), rpad(lname, 7),
                lpad(string(round(d.stat_hln, digits=3)), 8),
                lpad(string(round(d.pvalue_hln, digits=4)), 10),
                lpad(string(round(d.se / naive, digits=2)) * "x", 15),
                d.pvalue_hln < 0.05 ? "   SIGNIFICANT" : "   not significant",
            )
        end
    end

    println()
    println("--- level sanity check (reporting units only) ---")
    println("mean realized     : ", fmt(mean(rv)), "  (", fmt(100 * ann_vol(mean(rv)); d=4), "% annualized vol)")
    for (name, fc) in models
        bias = 100 * (mean(fc) / mean(rv) - 1)
        println(rpad("mean $name fcast", 18), ": ", fmt(mean(fc)),
                "  (", fmt(100 * ann_vol(mean(fc)); d=4), "%)",
                "  bias ", bias >= 0 ? "+" : "", fmt(bias; d=3), "%")
    end

    return results
end

main()
