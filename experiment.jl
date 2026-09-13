# Walk-forward experiment: EWMA vs GARCH(1,1) on NVDA's 21-day realized variance.
# Run with:  julia --project=. experiment.jl
#
# Reads the frozen CSV directly, not through DataFetch -- that points at a
# different cache path and can hit the network.

include("src/volatility.jl")
include("src/realized_vol.jl")
include("src/ewma.jl")
include("src/garch.jl")
include("src/loss.jl")
include("src/walk_forward.jl")

using .Volatility
using .RealizedVol
using .EWMA
using .Garch
using .Loss
using .WalkForward

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

    println("=== NVDA walk-forward: EWMA vs GARCH(1,1) ===")
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

    # A day either model is missing is dropped for both, so the comparison
    # stays like-for-like.
    has_target = .!ismissing.(results.realized)
    has_both = .!ismissing.(results.ewma_hstep) .& .!ismissing.(results.garch_hstep)
    mask = has_target .& has_both

    n_rows = nrow(results)
    n_no_target = count(.!has_target)
    n_garch_missing = count(ismissing, results.garch_hstep)
    n_scored = count(mask)

    println()
    println("Rows produced          : $n_rows")
    println("Dropped, no target     : $n_no_target  (last $H days have no forward window)")
    println("Dropped, GARCH missing : $n_garch_missing  (non-converged windows)")
    println("Scored on              : $n_scored days, identical set for both models")

    if n_garch_missing > 0
        println()
        println("!! Some GARCH windows did not converge. Open decision 2 (drop the window /")
        println("!! carry the previous fit forward / exclude the day) is now live and unresolved.")
        println("!! The table below excludes those days from BOTH models, which is the")
        println("!! 'exclude the day' branch -- it has not been agreed. Treat as provisional.")
    end

    ewma_s = score_column(results.realized, results.ewma_hstep, mask)
    garch_s = score_column(results.realized, results.garch_hstep, mask)

    rv = collect(skipmissing(results.realized[mask]))

    println()
    println("=== Results: $H-day total variance, out-of-sample ===")
    println(rpad("Model", 10), lpad("n", 7), lpad("QLIKE", 14), lpad("MSE", 16))
    for (name, s) in (("EWMA", ewma_s), ("GARCH", garch_s))
        println(
            rpad(name, 10),
            lpad(string(s.n), 7),
            lpad(fmt(s.qlike), 14),
            lpad(fmt(s.mse), 16),
        )
    end

    println()
    println("Lower is better for both losses.")
    qlike_winner = ewma_s.qlike < garch_s.qlike ? "EWMA" : "GARCH"
    mse_winner = ewma_s.mse < garch_s.mse ? "EWMA" : "GARCH"
    println("QLIKE favours : $qlike_winner")
    println("MSE favours   : $mse_winner")
    if qlike_winner != mse_winner
        println("The two losses DISAGREE. Per CLAUDE.md that is a finding, not an")
        println("inconvenience, and it gets its own paragraph in the README.")
    end
    println()
    println("No significance test has been run yet. With overlapping $H-day windows the")
    println("errors are strongly autocorrelated, so these means are not yet evidence of")
    println("a real difference -- that is the Diebold-Mariano step.")

    println()
    println("--- level sanity check (reporting units only) ---")
    println("mean realized     : ", fmt(mean(rv)), "  (", fmt(100 * ann_vol(mean(rv)); d=4), "% annualized vol)")
    ewma_fc = collect(skipmissing(results.ewma_hstep[mask]))
    garch_fc = collect(skipmissing(results.garch_hstep[mask]))
    println("mean EWMA fcast   : ", fmt(mean(ewma_fc)), "  (", fmt(100 * ann_vol(mean(ewma_fc)); d=4), "%)")
    println("mean GARCH fcast  : ", fmt(mean(garch_fc)), "  (", fmt(100 * ann_vol(mean(garch_fc)); d=4), "%)")

    return results
end

main()
