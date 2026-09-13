module WalkForward

using DataFrames
using CSV

using ..Garch
using ..EWMA
using ..RealizedVol
using ..Features
using EvoTrees

export walkforward_splits, walk_forward

# Fixed in advance, never tuned against the test period.
const GBT_NROUNDS = 200
const GBT_ETA = 0.05
const GBT_MAX_DEPTH = 4
const GBT_ROWSAMPLE = 0.8
const GBT_COLSAMPLE = 0.8
const GBT_SEED = 1234
const GBT_MIN_TRAIN_ROWS = 100

function walkforward_splits(
    n::Int,
    min_train::Int,
    test_size::Int,
    h::Int = 21
)
    splits = NamedTuple[]

    train_end = min_train

    while train_end + h < n

        test_start = train_end + h + 1

        test_end = min(test_start + test_size - 1, n)

        push!(
            splits,
            (
                train_end = train_end,
                test_range = test_start:test_end
            )
        )

        train_end += test_size
    end

    return splits
end

function walk_forward(
    returns;
    min_train::Int,
    test_size::Int,
    h::Int = 21,
    lambda::Float64 = 0.94,
    feature_windows = (5, 21, 63),
    output_path::Union{Nothing,AbstractString} = nothing,
)

    n = length(returns)

    splits = walkforward_splits(
        n,
        min_train,
        test_size,
        h
    )

    realized = forward_realized_variance(returns, h)

    ewma_path = ewma_variance_path(
        returns;
        lambda = lambda
    )

    # Causal row by row -- check (w) -- so building once over the sample is safe.
    gbt_ok = n >= maximum(feature_windows)
    feats = gbt_ok ? build_features(returns; lambda=lambda, windows=feature_windows) : nothing

    results = DataFrame(
        window = Int[],
        day = Int[],
        garch_1step = Union{Missing, Float64}[],
        garch_hstep = Union{Missing, Float64}[],
        ewma_1step = Union{Missing, Float64}[],
        ewma_hstep = Union{Missing, Float64}[],
        realized = Union{Missing, Float64}[],
        omega = Union{Missing, Float64}[],
        alpha = Union{Missing, Float64}[],
        beta = Union{Missing, Float64}[],
        mu = Union{Missing, Float64}[],
        converged = Bool[],
        gbt_hstep = Union{Missing, Float64}[]
    )

    garch_failed_windows = 0
    gbt_skipped_windows = 0

    for (window_id, split) in enumerate(splits)

        train_end = split.train_end
        test_range = split.test_range

        println(
            "Window $window_id: " *
            "train 1:$train_end, " *
            "test $(first(test_range)):$(last(test_range))"
        )

        fit = fit_garch11(
            returns[1:train_end];
            h0_window = train_end
        )

        if !fit.converged
            garch_failed_windows += 1
            println("  GARCH failed to converge — GARCH columns missing for this window; EWMA still scored")
        end

        garch_variances = fit.converged ? variances(garch11_forecast_path(
            returns[1:last(test_range)],
            fit.params;
            h0_window = train_end
        )) : nothing

        # Trains on log(target); exp() brings it back, which biases predictions
        # low. Left uncorrected -- the README reports the bias.
        gbt_pred = Dict{Int,Float64}()
        if gbt_ok
            train_rows = [t for t in 1:train_end
                          if feats.valid[t] && !ismissing(realized[t]) && realized[t] > 0]

            if length(train_rows) >= GBT_MIN_TRAIN_ROWS
                cfg = EvoTreeRegressor(
                    nrounds = GBT_NROUNDS,
                    eta = GBT_ETA,
                    max_depth = GBT_MAX_DEPTH,
                    rowsample = GBT_ROWSAMPLE,
                    colsample = GBT_COLSAMPLE,
                    seed = GBT_SEED,
                )
                model = EvoTrees.fit(
                    cfg;
                    x_train = feats.X[train_rows, :],
                    y_train = [log(Float64(realized[t])) for t in train_rows],
                )

                pred_rows = [t for t in test_range if feats.valid[t]]
                if !isempty(pred_rows)
                    p = Float64.(model(feats.X[pred_rows, :]))
                    for (i, t) in enumerate(pred_rows)
                        gbt_pred[t] = exp(p[i])
                    end
                end
            else
                gbt_skipped_windows += 1
                println("  GBT skipped: only $(length(train_rows)) usable training rows")
            end
        end

        for day in test_range

            if fit.converged
                garch_1step = garch_variances[day]
                garch_hstep = garch11_hstep_variance(
                    garch_1step,
                    fit.params,
                    h
                )
            else
                garch_1step = missing
                garch_hstep = missing
            end

            ewma_1step = ewma_path[day]

            ewma_hstep = ewma_hstep_variance(
                ewma_1step,
                h
            )

            push!(
                results,
                (
                    window_id,
                    day,
                    garch_1step,
                    garch_hstep,
                    ewma_1step,
                    ewma_hstep,
                    realized[day],
                    fit.params.omega,
                    fit.params.alpha,
                    fit.params.beta,
                    fit.params.mu,
                    fit.converged,
                    get(gbt_pred, day, missing)
                )
            )
        end
    end

    println()
    println("Walk-forward complete.")
    println("Windows attempted: $(length(splits))")
    println("Windows where GARCH failed to converge: $garch_failed_windows")
    println("Windows where GBT was skipped: $gbt_skipped_windows" *
            (gbt_ok ? "" : " (sample shorter than the longest feature window)"))
    println("Rows generated: $(nrow(results))")

    if output_path !== nothing
        mkpath(dirname(output_path))
        CSV.write(output_path, results)
        println("Forecasts written to $output_path")
    end

    return results
end

end
