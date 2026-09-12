module WalkForward

using DataFrames
using CSV

using ..Garch
using ..EWMA
using ..RealizedVol

export walkforward_splits, walk_forward

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
        converged = Bool[]
    )

    garch_failed_windows = 0

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
                    fit.converged
                )
            )
        end
    end

    println()
    println("Walk-forward complete.")
    println("Windows attempted: $(length(splits))")
    println("Windows where GARCH failed to converge: $garch_failed_windows")
    println("Rows generated: $(nrow(results))")

    if output_path !== nothing
        mkpath(dirname(output_path))
        CSV.write(output_path, results)
        println("Forecasts written to $output_path")
    end

    return results
end

end
