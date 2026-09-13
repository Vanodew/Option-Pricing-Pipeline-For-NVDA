# Forecast vs realized variance over the walk-forward test period.
# Run with:  julia --project=. chart.jl
#
# Reads results/walkforward.csv, so it re-renders without refitting anything.
# Raw SVG rather than a plotting package -- not worth the dependency tree for one
# static chart. Log y-axis because realized variance spans ~20x here.

using CSV
using DataFrames
using Dates
using Statistics

const RESULTS_PATH = "results/walkforward.csv"
const PRICES_PATH = "data/prices_10y.csv"
const OUT_PATH = "results/forecast_vs_realized.svg"

const H = 21

const W, HT = 1100, 560
const ML, MR, MT, MB = 72, 168, 58, 56   # margins: left, right, top, bottom
const PW, PH = W - ML - MR, HT - MT - MB # plot area

const C_REALIZED = "#333333"
const C_EWMA = "#4C72B0"
const C_GARCH = "#DD8452"
const C_GBT = "#55A868"
const C_GRID = "#E2E2E2"
const C_AXIS = "#9A9A9A"
const C_TEXT = "#222222"
const C_MUTED = "#666666"

# Labels only -- axis, data and scoring are all in variance.
ann_vol(v) = sqrt(v / H * 252)

esc(s) = replace(string(s), "&" => "&amp;", "<" => "&lt;", ">" => "&gt;")

function main()
    res = CSV.read(RESULTS_PATH, DataFrame)
    prices = CSV.read(PRICES_PATH, DataFrame)

    keep = .!ismissing.(res.realized) .&
           .!ismissing.(res.ewma_hstep) .&
           .!ismissing.(res.garch_hstep) .&
           .!ismissing.(res.gbt_hstep)
    d = res[keep, :]
    isempty(d) && error("nothing to plot: no rows with a target and both forecasts")

    days = Vector{Int}(d.day)
    # returns[i] is the return earned on prices.date[i+1] -- diff() drops one row.
    dates = prices.date[days .+ 1]

    realized = Vector{Float64}(d.realized)
    ewma = Vector{Float64}(d.ewma_hstep)
    garch = Vector{Float64}(d.garch_hstep)
    gbt = Vector{Float64}(d.gbt_hstep)

    lo = minimum(min.(realized, ewma, garch, gbt)) * 0.85
    hi = maximum(max.(realized, ewma, garch, gbt)) * 1.15

    x0, x1 = float(first(days)), float(last(days))
    px(day) = ML + (day - x0) / (x1 - x0) * PW
    py(v) = MT + PH - (log10(v) - log10(lo)) / (log10(hi) - log10(lo)) * PH

    polyline(xs, ys, colour, width, opacity) =
        "<polyline fill=\"none\" stroke=\"$colour\" stroke-width=\"$width\" " *
        "stroke-opacity=\"$opacity\" stroke-linejoin=\"round\" points=\"" *
        join(("$(round(px(xs[i]), digits=2)),$(round(py(ys[i]), digits=2))"
              for i in eachindex(xs)), " ") * "\"/>"

    io = IOBuffer()
    println(io, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"$W\" height=\"$HT\" ",
                "viewBox=\"0 0 $W $HT\" font-family=\"-apple-system,BlinkMacSystemFont,",
                "'Segoe UI',Helvetica,Arial,sans-serif\">")
    println(io, "<rect width=\"$W\" height=\"$HT\" fill=\"#FFFFFF\"/>")

    # --- title -------------------------------------------------------------
    println(io, "<text x=\"$ML\" y=\"28\" font-size=\"17\" font-weight=\"600\" ",
                "fill=\"$C_TEXT\">NVDA: $H-day forecast vs realized variance, out-of-sample</text>")
    println(io, "<text x=\"$ML\" y=\"46\" font-size=\"12\" fill=\"$C_MUTED\">",
                "Walk-forward, expanding window, $H-day embargo &#183; ",
                esc(first(dates)), " to ", esc(last(dates)),
                " &#183; $(length(days)) scored days &#183; log scale</text>")

    # --- y gridlines -------------------------------------------------------
    for v in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1)
        (v < lo || v > hi) && continue
        y = round(py(v), digits=2)
        println(io, "<line x1=\"$ML\" y1=\"$y\" x2=\"$(ML + PW)\" y2=\"$y\" ",
                    "stroke=\"$C_GRID\" stroke-width=\"1\"/>")
        println(io, "<text x=\"$(ML - 9)\" y=\"$(y + 4)\" font-size=\"11\" ",
                    "text-anchor=\"end\" fill=\"$C_MUTED\">$v</text>")
        println(io, "<text x=\"$(ML - 9)\" y=\"$(y + 15)\" font-size=\"9\" ",
                    "text-anchor=\"end\" fill=\"#B0B0B0\">",
                    "$(round(Int, 100 * ann_vol(v)))% vol</text>")
    end

    # --- x gridlines at year boundaries ------------------------------------
    for yr in (year(first(dates)) + 1):year(last(dates))
        idx = findfirst(>=(Date(yr, 1, 1)), dates)
        idx === nothing && continue
        x = round(px(days[idx]), digits=2)
        println(io, "<line x1=\"$x\" y1=\"$MT\" x2=\"$x\" y2=\"$(MT + PH)\" ",
                    "stroke=\"$C_GRID\" stroke-width=\"1\"/>")
        println(io, "<text x=\"$x\" y=\"$(MT + PH + 20)\" font-size=\"11\" ",
                    "text-anchor=\"middle\" fill=\"$C_MUTED\">$yr</text>")
    end

    # --- series ------------------------------------------------------------
    println(io, polyline(days, realized, C_REALIZED, 1.1, 0.55))
    println(io, polyline(days, ewma, C_EWMA, 1.5, 0.9))
    println(io, polyline(days, garch, C_GARCH, 1.5, 0.9))
    println(io, polyline(days, gbt, C_GBT, 1.5, 0.9))

    # --- axis lines --------------------------------------------------------
    println(io, "<line x1=\"$ML\" y1=\"$(MT + PH)\" x2=\"$(ML + PW)\" y2=\"$(MT + PH)\" ",
                "stroke=\"$C_AXIS\" stroke-width=\"1\"/>")
    println(io, "<text x=\"$(ML - 9)\" y=\"$(MT - 12)\" font-size=\"11\" ",
                "text-anchor=\"end\" fill=\"$C_MUTED\">variance</text>")

    # --- legend ------------------------------------------------------------
    lx = ML + PW + 22
    entries = (
        ("Realized", C_REALIZED, "the target: sum of squared\nreturns over t+1..t+$H"),
        ("EWMA", C_EWMA, "RiskMetrics &#955;=0.94,\nbias +0.01%"),
        ("GARCH(1,1)", C_GARCH, "refit every window,\nbias +7.5%"),
        ("GBT", C_GBT, "refit every window,\nbias -15.7%"),
    )
    ly = MT + 4
    for (label, colour, note) in entries
        println(io, "<line x1=\"$lx\" y1=\"$ly\" x2=\"$(lx + 22)\" y2=\"$ly\" ",
                    "stroke=\"$colour\" stroke-width=\"2.4\"/>")
        println(io, "<text x=\"$(lx + 29)\" y=\"$(ly + 4)\" font-size=\"12\" ",
                    "font-weight=\"600\" fill=\"$C_TEXT\">$label</text>")
        for (k, line) in enumerate(split(note, "\n"))
            println(io, "<text x=\"$lx\" y=\"$(ly + 20 + 12 * k)\" font-size=\"10\" ",
                        "fill=\"$C_MUTED\">$line</text>")
        end
        ly += 68
    end

    # --- caption -----------------------------------------------------------
    println(io, "<text x=\"$ML\" y=\"$(HT - 16)\" font-size=\"11\" fill=\"$C_MUTED\">",
                "No model beats EWMA significantly. GARCH scores best by sitting high and ",
                "flat (",
                "$(round(maximum(garch) / minimum(garch), digits=1))&#215; range against ",
                "realized's $(round(maximum(realized) / minimum(realized), digits=1))&#215;), ",
                "never badly caught low; GBT runs low and loses to GARCH on QLIKE.</text>")

    println(io, "</svg>")

    mkpath(dirname(OUT_PATH))
    write(OUT_PATH, String(take!(io)))

    println("Wrote $OUT_PATH")
    println("  scored days : $(length(days))  ($(first(dates)) to $(last(dates)))")
    println("  y-range     : $(round(lo, sigdigits=3)) to $(round(hi, sigdigits=3)) (log)")
    println("  realized    : median $(round(median(realized), sigdigits=4)), " *
            "span $(round(maximum(realized) / minimum(realized), digits=1))x")
    println("  garch       : median $(round(median(garch), sigdigits=4)), " *
            "span $(round(maximum(garch) / minimum(garch), digits=1))x")
    println("  ewma        : median $(round(median(ewma), sigdigits=4)), " *
            "span $(round(maximum(ewma) / minimum(ewma), digits=1))x")
    println("  gbt         : median $(round(median(gbt), sigdigits=4)), " *
            "span $(round(maximum(gbt) / minimum(gbt), digits=1))x")
end

main()
