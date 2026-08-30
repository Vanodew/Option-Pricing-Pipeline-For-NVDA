module DataFetch

using HTTP
using JSON3
using CSV
using DataFrames
using Dates

export fetch_price_history

function fetch_price_history(
    symbol::String;
    days::Int=730,
    cache_path::String="data/prices.csv",
    force_refresh::Bool=false,
)::DataFrame
    if !force_refresh && isfile(cache_path)
        return CSV.read(cache_path, DataFrame)
    end

    range_str = days <= 730 ? "2y" : days <= 1825 ? "5y" : "10y"
    url = "https://query1.finance.yahoo.com/v8/finance/chart/$symbol" *
        "?range=$range_str&interval=1d"

    response = HTTP.get(url, ["User-Agent" => "Mozilla/5.0"]) #this line masks the traffic as mozilla firefox 
    body = JSON3.read(response.body)

    result = body.chart.result
    (result === nothing || isempty(result)) && error(
        "Yahoo Finance returned no data for symbol $symbol. Check the symbol is correct.",
    )

    r = result[1]
    timestamps = r.timestamp
    closes = r.indicators.quote[1].close
    #body.chart.result[1].indicators.quote[1].close is simply yahoo's nesting to reach the actual price array
    close_vals = [c === nothing ? missing : Float64(c) for c in closes]

    df = DataFrame(
        date=Date.(unix2datetime.(timestamps)),
        close=close_vals,
    )
    dropmissing!(df) #simply removes the rows that are null
    sort!(df, :date)

    cutoff = Dates.today() - Day(days)
    df = df[df.date .>= cutoff, :]

    mkpath(dirname(cache_path))
    CSV.write(cache_path, df)

    return df
end

end # module
