using DataFrames
include("src/data.jl")
using .DataFetch

df = fetch_price_history("NVDA")
println(nrow(df), " rows")
println(first(df, 5))
println(last(df, 5))
