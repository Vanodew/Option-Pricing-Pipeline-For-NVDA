module Loss

using Statistics: mean

export qlike,mse,mean_loss

#first we wanna write about QLIKE and MSE prediction
#second mean_loss must be calculated for each

function qlike(rv::Real, forecast::Real)::Float64
    if forecast <= 0
        error("Forecast variance must be positive")
    end

    x = rv / forecast
    return x - log(x) -1
end

function mse(rv::Real, forecast::Real)::Float64
    return (rv - forecast)^2
end

function mean_loss(realized, forecasts, loss_fn)::Float64
    losses = Float64[]

    for (rv,forecast) in zip(realized,forecasts)
        if !ismissing(rv) && !ismissing(forecast)
            push!(losses, loss_fn(rv,forecast))
        end
    end

    return mean(losses)
end

end

