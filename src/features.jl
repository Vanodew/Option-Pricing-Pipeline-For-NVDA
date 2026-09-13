module Features

using ..RealizedVol
using ..EWMA

export build_features

"""
    build_features(returns; lambda=0.94, windows=(5, 21, 63))

Feature matrix for the GBT, one row per day, every column built from returns up
to and including day `t`. Columns: trv_<w> for each window, ret_1, ret_mean_5,
ewma_1step.

Returns `(X, names, valid)`. `valid[t]` is false while any column is undefined
and those rows hold NaN, so callers must filter on it.
"""
function build_features(
    returns::AbstractVector{<:Real};
    lambda::Float64=0.94,
    windows::NTuple{N,Int}=(5, 21, 63),
) where {N}
    n = length(returns)
    n >= 2 || error("Need at least 2 returns, got $n.")
    all(w -> w >= 1, windows) || error("Windows must be at least 1 day.")

    cols = Vector{Vector{Union{Missing,Float64}}}()
    names = String[]

    for w in windows
        w <= n || error("Window w=$w exceeds the sample length $n.")
        push!(cols, trailing_realized_variance(returns, w))
        push!(names, "trv_$w")
    end

    push!(cols, Vector{Union{Missing,Float64}}(Float64.(returns)))
    push!(names, "ret_1")

    m5 = Vector{Union{Missing,Float64}}(missing, n)
    for t in 5:n
        m5[t] = sum(@view returns[(t - 4):t]) / 5
    end
    push!(cols, m5)
    push!(names, "ret_mean_5")

    push!(cols, Vector{Union{Missing,Float64}}(ewma_variance_path(returns; lambda=lambda)))
    push!(names, "ewma_1step")

    p = length(cols)
    X = Matrix{Float64}(undef, n, p)
    valid = trues(n)
    for j in 1:p, t in 1:n
        v = cols[j][t]
        if ismissing(v)
            valid[t] = false
            X[t, j] = NaN
        else
            X[t, j] = v
        end
    end

    return (X=X, names=names, valid=valid)
end

end # module
