module RealizedVol

export forward_realized_variance, trailing_realized_variance
#understand again the differnece between the forward and trailing realized variance
function forward_realized_variance(
    returns::AbstractVector{<:Real}, h::Int,
)::Vector{Union{Missing,Float64}}
    h >= 1 || error("Horizon h must be at least 1 day.")
    n = length(returns)
    n > h || error("Need more than h=$h returns, got $n.")
    out = Vector{Union{Missing,Float64}}(missing, n)
    for t in 1:(n - h)
        out[t] = sum(abs2, @view returns[(t + 1):(t + h)])
    end
    return out
end

function trailing_realized_variance(
    returns::AbstractVector{<:Real}, w::Int,
)::Vector{Union{Missing,Float64}}
    w >= 1 || error("Window w must be at least 1 day.")
    n = length(returns)
    n >= w || error("Need at least w=$w returns, got $n.")
    out = Vector{Union{Missing,Float64}}(missing, n)
    for t in w:n
        out[t] = sum(abs2, @view returns[(t - w + 1):t])
    end
    return out
end

end # module
