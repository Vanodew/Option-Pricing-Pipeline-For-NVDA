module Garch

import Optim
using LinearAlgebra: diag, inv, pinv #applying matrix use also need to do that in ewma and data.jl
using Statistics: mean, var

export GARCH11Params, GARCH11Fit, VarianceSeries, ConditionalVariance, OneStepForecast,
       variances, timing,
       garch11_variance_path, garch11_logliks, garch11_loglik, fit_garch11,
       garch11_forecast_path, garch11_hstep_variance,
       unconditional_variance, persistence

struct GARCH11Params
    omega::Float64
    alpha::Float64
    beta::Float64
    mu::Float64
end

persistence(p::GARCH11Params) = p.alpha + p.beta

unconditional_variance(p::GARCH11Params) = p.omega / (1.0 - p.alpha - p.beta)

# =============================================================================
# Timing-tagged variance series
# =============================================================================

struct VarianceSeries{A,T<:Real}
    values::Vector{T}

    function VarianceSeries{A,T}(values::Vector{T}) where {A,T<:Real}
        A === :conditional || A === :forecast ||
            error("Unknown timing tag :$A (expected :conditional or :forecast).")
        return new{A,T}(values)
    end
end

VarianceSeries{A}(values::Vector{T}) where {A,T<:Real} = VarianceSeries{A,T}(values)

const ConditionalVariance = VarianceSeries{:conditional}

const OneStepForecast = VarianceSeries{:forecast}

variances(s::VarianceSeries) = s.values

timing(::VarianceSeries{A}) where {A} = A

Base.length(s::VarianceSeries) = length(s.values)
Base.getindex(s::VarianceSeries, i) = s.values[i]
Base.firstindex(s::VarianceSeries) = firstindex(s.values)
Base.lastindex(s::VarianceSeries) = lastindex(s.values)
Base.eachindex(s::VarianceSeries) = eachindex(s.values)
Base.iterate(s::VarianceSeries, st...) = iterate(s.values, st...)
Base.eltype(::VarianceSeries{A,T}) where {A,T} = T

const _TIMING_DOC = Dict(
    :conditional => "h_t, known at t-1",
    :forecast => "sigma^2_{t+1|t}, known at t",
)

_mismatch(A, B) = ArgumentError(
    "Refusing to compare variance series with different timing conventions: " *
    ":$A ($(_TIMING_DOC[A])) vs :$B ($(_TIMING_DOC[B])). They are offset by one " *
    "index — forecast[t] == conditional[t+1]. Slice explicitly via variances(), " *
    "e.g. variances(fc)[1:end-1] == variances(cv)[2:end]."
)

_bare() = ArgumentError(
    "Refusing to compare a VarianceSeries against a bare vector: the vector " *
    "carries no timing convention, so the comparison could be silently " *
    "misaligned by one day. Unwrap with variances(...) if the alignment is " *
    "known to be right, or wrap the vector in ConditionalVariance/OneStepForecast."
)

function Base.:(==)(a::VarianceSeries{A}, b::VarianceSeries{B}) where {A,B}
    A === B || throw(_mismatch(A, B))
    return a.values == b.values
end

function Base.isapprox(a::VarianceSeries{A}, b::VarianceSeries{B}; kw...) where {A,B}
    A === B || throw(_mismatch(A, B))
    return isapprox(a.values, b.values; kw...)
end

Base.:(==)(::VarianceSeries, ::AbstractVector) = throw(_bare())
Base.:(==)(::AbstractVector, ::VarianceSeries) = throw(_bare())
Base.isapprox(::VarianceSeries, ::AbstractVector; kw...) = throw(_bare())
Base.isapprox(::AbstractVector, ::VarianceSeries; kw...) = throw(_bare())

function Base.show(io::IO, s::VarianceSeries{A}) where {A}
    print(io, "VarianceSeries{:$A}(n=", length(s.values), ", ",
          _TIMING_DOC[A], ")")
end

# =============================================================================
# Seeding
# =============================================================================

function _resolve_seed(returns::AbstractVector{<:Real},
                       h0::Union{Nothing,Real},
                       h0_window::Union{Nothing,Integer})
    if h0 !== nothing && h0_window !== nothing
        error("Pass h0 or h0_window, not both — they are two ways to set the same seed.")
    elseif h0 !== nothing
        h0 >= 0 || error("h0 must be a nonnegative variance, got $h0.")
        return float(h0)
    elseif h0_window !== nothing
        k = Int(h0_window)
        2 <= k <= length(returns) ||
            error("h0_window must be in 2:$(length(returns)), got $k (need >= 2 for a variance).")
        return float(var(@view returns[1:k]))
    else
        return float(var(returns))
    end
end

# =============================================================================
# Recursion and likelihood
# =============================================================================

# Internal, generic in the parameter type so ForwardDiff duals pass through
# during the BFGS polish. Returns a bare vector; the public wrapper tags it.
function _variance_path(returns::AbstractVector{<:Real}, omega::Real, alpha::Real,
                        beta::Real, mu::Real, seed::Real)
    n = length(returns)
    T = promote_type(typeof(omega), typeof(alpha), typeof(beta), typeof(mu),
                     typeof(seed), Float64)
    h = Vector{T}(undef, n)
    h[1] = seed
    for t in 2:n
        a_prev = returns[t - 1] - mu
        h[t] = omega + alpha * abs2(a_prev) + beta * h[t - 1]
    end
    return h
end

function garch11_variance_path(
    returns::AbstractVector{<:Real}, omega::Real, alpha::Real, beta::Real;
    mu::Real=0.0, h0::Union{Nothing,Real}=nothing,
    h0_window::Union{Nothing,Integer}=nothing,
)
    omega >= 0 || error("omega must be nonnegative, got $omega.")
    alpha >= 0 || error("alpha must be nonnegative, got $alpha.")
    beta >= 0 || error("beta must be nonnegative, got $beta.")
    length(returns) >= 1 || error("Need at least one return.")
    seed = _resolve_seed(returns, h0, h0_window)
    return ConditionalVariance(_variance_path(returns, omega, alpha, beta, mu, seed))
end

# Internal: per-observation log-likelihood contributions, generic in T.
function _logliks(returns::AbstractVector{<:Real}, omega::Real, alpha::Real,
                  beta::Real, mu::Real, seed::Real)
    h = _variance_path(returns, omega, alpha, beta, mu, seed)
    T = eltype(h)
    out = Vector{T}(undef, length(h))
    logtwopi = log(2 * pi)
    for t in eachindex(h)
        if h[t] <= 0
            out[t] = T(-Inf)
        else
            a = returns[t] - mu
            out[t] = -0.5 * (logtwopi + log(h[t]) + abs2(a) / h[t])
        end
    end
    return out
end

function garch11_logliks(
    returns::AbstractVector{<:Real}, omega::Real, alpha::Real, beta::Real;
    mu::Real=0.0, h0::Union{Nothing,Real}=nothing,
    h0_window::Union{Nothing,Integer}=nothing,
)
    seed = _resolve_seed(returns, h0, h0_window)
    return _logliks(returns, omega, alpha, beta, mu, seed)
end

function garch11_loglik(
    returns::AbstractVector{<:Real}, omega::Real, alpha::Real, beta::Real;
    mu::Real=0.0, h0::Union{Nothing,Real}=nothing,
    h0_window::Union{Nothing,Integer}=nothing,
)
    return sum(garch11_logliks(returns, omega, alpha, beta;
                               mu=mu, h0=h0, h0_window=h0_window))
end

# =============================================================================
# Estimation
# =============================================================================

# --- unconstrained reparameterization ----------------------------------------
#
# The MLE must satisfy omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1. A
# box-constrained optimizer handles the first three but not the sum, and
# solutions sit close enough to alpha + beta = 1 that an unconstrained
# optimizer walks straight out of the valid region.
#
# So optimize over an unconstrained theta in R^4 that maps *onto* the valid
# region and nowhere else:
#
#   omega = exp(theta_1)                          > 0
#   p     = logistic(theta_2)                     in (0, 1)   persistence
#   s     = logistic(theta_3)                     in (0, 1)   alpha's share
#   alpha = p * s,  beta = p * (1 - s)            alpha + beta = p < 1
#   mu    = theta_4                               unrestricted
#
# Splitting into persistence and share (rather than transforming alpha and
# beta separately) is what makes the sum constraint automatic. The MLE is
# invariant to reparameterization, so the fitted coefficients are the same
# ones a constrained optimizer would find.
#
# Standard errors, however, are NOT invariant, which is why the covariance
# below is computed in the natural (omega, alpha, beta, mu) space rather than
# in theta space.

_logistic(x) = 1 / (1 + exp(-x))
_logit(x) = log(x / (1 - x))

function _unpack(theta::AbstractVector)
    omega = exp(theta[1])
    p = _logistic(theta[2])
    s = _logistic(theta[3])
    return omega, p * s, p * (1 - s), theta[4]
end

function _pack(omega::Real, alpha::Real, beta::Real, mu::Real)
    p = alpha + beta
    return [log(omega), _logit(p), _logit(alpha / p), mu]
end

struct GARCH11Fit
    params::GARCH11Params
    se::NamedTuple{(:omega, :alpha, :beta, :mu),NTuple{4,Float64}}
    se_hessian::NamedTuple{(:omega, :alpha, :beta, :mu),NTuple{4,Float64}}
    tstat::NamedTuple{(:omega, :alpha, :beta, :mu),NTuple{4,Float64}}
    loglik::Float64
    converged::Bool
end

function Base.show(io::IO, ::MIME"text/plain", f::GARCH11Fit)
    p = f.params
    println(io, "GARCH(1,1) with Gaussian errors, hand-coded MLE",
            f.converged ? "" : "  [NOT CONVERGED]")
    println(io, "")
    println(io, "            Estimate     Std.Error (robust)   t")
    for (name, est) in (("omega", p.omega), ("alpha", p.alpha),
                        ("beta ", p.beta), ("mu   ", p.mu))
        key = Symbol(strip(name))
        println(io, "  ", name, "  ",
                rpad(string(round(est, sigdigits=6)), 14), " ",
                rpad(string(round(getproperty(f.se, key), sigdigits=5)), 18), " ",
                round(getproperty(f.tstat, key), digits=3))
    end
    println(io, "")
    println(io, "  log L       ", round(f.loglik, digits=4))
    println(io, "  persistence ", round(persistence(p), digits=5),
            "   (shock half-life ",
            round(log(0.5) / log(persistence(p)), digits=1), " days)")
    print(io, "  long-run vol ",
          round(100 * sqrt(252 * unconditional_variance(p)), digits=2), "% annualized")
end

Base.show(io::IO, f::GARCH11Fit) = show(io, MIME"text/plain"(), f)

# --- numerical derivatives ----------------------------------------------------
#
# Finite differences rather than autodiff, so the covariance is derived from
# the likelihood as written rather than inheriting the same code path twice.
# Step sizes are relative to each parameter, which matters here because the
# four parameters differ by four orders of magnitude (omega ~ 1e-5, beta ~ 1).
#
# The exponents are the standard bias/round-off optima: for a central FIRST
# difference the truncation error is O(h^2) and the round-off O(eps/h), which
# balance at h ~ eps^(1/3); for a central SECOND difference the round-off is
# O(eps/h^2) and they balance at h ~ eps^(1/4).

_step(x, rel) = rel * max(abs(x), 1e-8)

const _SCORE_REL = cbrt(eps(Float64))          # ~6.1e-6
const _HESS_REL = eps(Float64)^0.25            # ~1.2e-4

# Per-observation score matrix S (n x 4): S[t, i] = d l_t / d x_i.
function _score_matrix(f, x::Vector{Float64})
    n = length(f(x))
    S = Matrix{Float64}(undef, n, 4)
    for i in 1:4
        h = _step(x[i], _SCORE_REL)
        xp = copy(x); xp[i] += h
        xm = copy(x); xm[i] -= h
        S[:, i] = (f(xp) .- f(xm)) ./ (2h)
    end
    return S
end

# Hessian of the total log-likelihood by central second differences.
function _hessian(g, x::Vector{Float64})
    H = Matrix{Float64}(undef, 4, 4)
    hs = [_step(x[i], _HESS_REL) for i in 1:4]
    g0 = g(x)
    for i in 1:4
        xp = copy(x); xp[i] += hs[i]
        xm = copy(x); xm[i] -= hs[i]
        H[i, i] = (g(xp) - 2g0 + g(xm)) / hs[i]^2
    end
    for i in 1:4, j in (i + 1):4
        xpp = copy(x); xpp[i] += hs[i]; xpp[j] += hs[j]
        xpm = copy(x); xpm[i] += hs[i]; xpm[j] -= hs[j]
        xmp = copy(x); xmp[i] -= hs[i]; xmp[j] += hs[j]
        xmm = copy(x); xmm[i] -= hs[i]; xmm[j] -= hs[j]
        H[i, j] = H[j, i] =
            (g(xpp) - g(xpm) - g(xmp) + g(xmm)) / (4 * hs[i] * hs[j])
    end
    return H
end

_safe_inv(M) = try
    inv(M)
catch
    @warn "Information matrix is singular; falling back to pseudoinverse. Standard errors are unreliable."
    pinv(M)
end

_nt(v) = (omega=v[1], alpha=v[2], beta=v[3], mu=v[4])

function fit_garch11(
    returns::AbstractVector{<:Real};
    h0::Union{Nothing,Real}=nothing,
    h0_window::Union{Nothing,Integer}=nothing,
    start::Union{Nothing,GARCH11Params}=nothing,
)
    n = length(returns)
    n >= 10 || error("Need at least 10 returns to fit GARCH(1,1), got $n.")
    r = collect(float.(returns))
    seed = _resolve_seed(r, h0, h0_window)

    s = start === nothing ?
        GARCH11Params(var(r) * (1 - 0.05 - 0.90), 0.05, 0.90, mean(r)) : start
    theta0 = _pack(s.omega, s.alpha, s.beta, s.mu)

    nll = function (theta)
        omega, alpha, beta, mu = _unpack(theta)
        ll = sum(_logliks(r, omega, alpha, beta, mu, seed))
        return isfinite(ll) ? -ll : oftype(ll, Inf)
    end

    res = Optim.optimize(nll, theta0, Optim.NelderMead(),
                         Optim.Options(g_tol=1e-12, iterations=10_000))
    theta = Optim.minimizer(res)
    ok = Optim.converged(res)

    polished = Optim.optimize(nll, theta, Optim.BFGS(),
                              Optim.Options(g_tol=1e-10, iterations=2_000))
    if Optim.converged(polished) && Optim.minimum(polished) <= Optim.minimum(res)
        theta = Optim.minimizer(polished)
        ok = true
    end

    omega, alpha, beta, mu = _unpack(theta)
    params = GARCH11Params(omega, alpha, beta, mu)
    x = [omega, alpha, beta, mu]

    # Covariance in the natural parameter space.
    logliks_at = z -> _logliks(r, z[1], z[2], z[3], z[4], seed)
    total_at = z -> sum(logliks_at(z))

    J = -_hessian(total_at, x)              # observed information
    Ji = _safe_inv(J)
    S = _score_matrix(logliks_at, x)
    V = S' * S                              # outer product of scores
    sandwich = Ji * V * Ji

    se_rob = _nt(sqrt.(abs.(diag(sandwich))))
    se_hes = _nt(sqrt.(abs.(diag(Ji))))
    tst = _nt([x[i] / sqrt(abs(diag(sandwich)[i])) for i in 1:4])

    return GARCH11Fit(params, se_rob, se_hes, tst, sum(logliks_at(x)), ok)
end

# =============================================================================
# Forecasting
# =============================================================================

function garch11_forecast_path(
    returns::AbstractVector{<:Real}, p::GARCH11Params;
    h0::Union{Nothing,Real}=nothing,
    h0_window::Union{Nothing,Integer}=nothing,
)
    seed = _resolve_seed(returns, h0, h0_window)
    h = _variance_path(returns, p.omega, p.alpha, p.beta, p.mu, seed)
    out = Vector{Float64}(undef, length(h))
    for t in eachindex(h)
        a = returns[t] - p.mu
        out[t] = p.omega + p.alpha * abs2(a) + p.beta * h[t]
    end
    return OneStepForecast(out)
end

function garch11_hstep_variance(v_next::Real, p::GARCH11Params, h::Int)
    h >= 1 || error("Horizon h must be at least 1 day.")
    v_next >= 0 || error("Variance forecast must be nonnegative.")
    rho = p.alpha + p.beta
    if rho >= 1 - 1e-12
        return h * Float64(v_next)   # IGARCH / EWMA limit: flat forecast
    end
    h_bar = p.omega / (1 - rho)
    return h * h_bar + (Float64(v_next) - h_bar) * (1 - rho^h) / (1 - rho)
end

end # module
