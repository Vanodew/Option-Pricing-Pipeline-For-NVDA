module EWMA

#takes small parts of the main module and exports them in the code
export ewma_variance_path, ewma_hstep_variance
#TRYNA UNDERSTAND EXPORT
function ewma_variance_path(
    returns::AbstractVector{<:Real};
    lambda::Float64=0.94,
    init::Float64=Float64(first(returns))^2, #square because its a variance estimate but over our time span its negligeble
)::Vector{Float64}
    0.0 < lambda < 1.0 || error("lambda must be strictly between 0 and 1.")
    init >= 0.0 || error("init must be a nonnegative variance.")
    n = length(returns)
    v = Vector{Float64}(undef, n)
    prev = init                       # sigma^2_{1|0}
    for t in 1:n
        prev = lambda * prev + (1.0 - lambda) * abs2(Float64(returns[t])) #formula of ewma
        v[t] = prev                   # sigma^2_{t+1|t}
    end
    return v
end

function ewma_hstep_variance(v_next::Real, h::Int)::Float64 #increasing the number of days
    h >= 1 || error("Horizon h must be at least 1 day.")
    v_next >= 0 || error("Variance forecast must be nonnegative.")
    return h * Float64(v_next)
end

end # module
