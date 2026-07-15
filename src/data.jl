module DataFetch

using HTTP
using JSON3
using CSV
using DataFrames
using Dates

export fetch_price_history

"""
    fetch_price_history(symbol; days=730, cache_path="data/prices.csv", force_refresh=false)

Get daily close prices for `symbol` as a DataFrame with columns `:date` and `:close`.

Reads Yahoo Finance's public chart endpoint for daily history over the requested
`days` window. No API key required.

Caches results to `cache_path` as CSV. On subsequent calls, loads from the cache
instead of hitting the API again, unless `force_refresh=true`.
"""
function fetch_price_history(
    symbol::String;
    days::Int=730,
    cache_path::String="data/prices.csv",
    force_refresh::Bool=false,
)::DataFrame
    if !force_refresh && isfile(cache_path)
        return CSV.read(cache_path, DataFrame)
    end

    range_str = days <= 730 ? "2y" : "5y"
    url = "https://query1.finance.yahoo.com/v8/finance/chart/$symbol" *
        "?range=$range_str&interval=1d"

    response = HTTP.get(url, ["User-Agent" => "Mozilla/5.0"])
    body = JSON3.read(response.body)

    result = body.chart.result
    (result === nothing || isempty(result)) && error(
        "Yahoo Finance returned no data for symbol $symbol. Check the symbol is correct.",
    )

    r = result[1]
    timestamps = r.timestamp
    closes = r.indicators.quote[1].close

    close_vals = [c === nothing ? missing : Float64(c) for c in closes]

    df = DataFrame(
        date=Date.(unix2datetime.(timestamps)),
        close=close_vals,
    )
    dropmissing!(df)
    sort!(df, :date)

    cutoff = Dates.today() - Day(days)
    df = df[df.date .>= cutoff, :]

    mkpath(dirname(cache_path))
    CSV.write(cache_path, df)

    return df
end

end # module
