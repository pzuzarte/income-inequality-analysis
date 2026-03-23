#!/usr/bin/env python3
"""
Income & Wealth Inequality Analysis
USA · Canada · Norway and OECD Peer Nations

Data Sources:
  • World Bank Open Data API       — GDP per capita, Gini index, income quintile/decile shares
  • World Inequality Database      — Pre-tax income & wealth top shares (2022) [wid.world]
  • Credit Suisse Global Wealth Report 2023 — Mean/median wealth per adult, wealth shares
  • OECD Income Distribution Database       — Relative poverty rates (~2021)

Output: report.html
"""

import warnings
import json
import os
from datetime import datetime

import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import special, stats as sp_stats

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

COUNTRIES = [
    "United States", "Canada", "Norway",
    "Sweden", "Denmark", "Germany", "United Kingdom", "France",
]
FOCAL = {"United States", "Canada", "Norway"}

WB_ISO3 = {
    "United States":  "USA", "Canada": "CAN", "Norway": "NOR",
    "Sweden": "SWE",  "Denmark": "DNK", "Germany": "DEU",
    "United Kingdom": "GBR", "France": "FRA",
}

COLORS = {
    "United States":  "#FF4D6D",
    "Canada":         "#FF9F1C",
    "Norway":         "#00F5D4",
    "Sweden":         "#7B2FBE",
    "Denmark":        "#F72585",
    "Germany":        "#8D99AE",
    "United Kingdom": "#4CC9F0",
    "France":         "#4361EE",
}

SHORT = {
    "United States": "USA", "Canada": "CAN", "Norway": "NOR",
    "Sweden": "SWE", "Denmark": "DNK", "Germany": "DEU",
    "United Kingdom": "UK", "France": "FRA",
}

# ════════════════════════════════════════════════════════════════════════════
# CURATED DATA
# ════════════════════════════════════════════════════════════════════════════

# WID.world 2022 — Pre-tax national income shares (%)
# Source: World Inequality Database (wid.world), adult equal-split, 2022
# Convention: top10 + mid40 + bot50 = 100%; top1 is a subset of top10
WID_INCOME = {
    # country: (top1%, top10%, mid40%, bot50%)
    "United States":  (19.0, 45.4, 40.4, 14.2),
    "Canada":         (13.6, 41.0, 42.8, 16.2),
    "Norway":         ( 9.5, 34.0, 45.3, 20.7),
    "Sweden":         ( 9.3, 33.2, 45.0, 21.8),
    "Denmark":        (10.0, 35.6, 44.5, 19.9),
    "Germany":        (13.0, 40.0, 44.5, 15.5),
    "United Kingdom": (13.5, 40.0, 43.5, 16.5),
    "France":         (10.5, 35.0, 46.5, 18.5),
}

# Credit Suisse Global Wealth Report 2023 — Net personal wealth shares (%)
# Convention: top10 + mid40 + bot50 = 100%; top1 is a subset of top10
WID_WEALTH = {
    # country: (top1%, top10%, mid40%, bot50%)
    "United States":  (35.1, 70.7, 26.9,  2.4),
    "Canada":         (25.6, 57.5, 36.8,  5.7),
    "Norway":         (20.0, 52.0, 39.5,  8.5),
    "Sweden":         (21.4, 54.4, 37.8,  7.8),
    "Denmark":        (24.2, 64.4, 31.8,  3.8),
    "Germany":        (29.4, 63.3, 32.5,  4.2),
    "United Kingdom": (22.8, 57.0, 37.8,  5.2),
    "France":         (23.6, 55.0, 38.5,  6.5),
}

# OECD — Relative poverty (income < 50% of national median), %
POVERTY = {
    "United States": 17.8, "Canada": 12.1, "Norway":  8.4,
    "Sweden":         8.7,  "Denmark": 6.7, "Germany": 10.1,
    "United Kingdom": 11.7, "France":  8.5,
}

# Credit Suisse 2023 — Mean and median wealth per adult (USD)
WEALTH_MEAN = {
    "United States": 551350, "Canada": 369580, "Norway": 385230,
    "Sweden": 333750, "Denmark": 422440, "Germany": 256180,
    "United Kingdom": 302040, "France": 299750,
}
WEALTH_MEDIAN = {
    "United States": 107740, "Canada": 125690, "Norway": 168590,
    "Sweden": 118640, "Denmark": 155440, "Germany": 65374,
    "United Kingdom": 130048, "France": 133560,
}

# Gini fallbacks (Luxembourg Income Study, ca. 2020) — used if WB API unavailable
GINI_FALLBACK = {
    "United States": 39.6, "Canada": 32.8, "Norway": 26.2,
    "Sweden": 27.3, "Denmark": 28.7, "Germany": 31.5,
    "United Kingdom": 35.7, "France": 29.2,
}

# ════════════════════════════════════════════════════════════════════════════
# WORLD BANK DATA FETCH
# ════════════════════════════════════════════════════════════════════════════

WB_BASE = "https://api.worldbank.org/v2"
WB_INDICATORS = {
    "gdp_per_capita": "NY.GDP.PCAP.CD",
    "gini":           "SI.POV.GINI",
    "income_top20":   "SI.DST.05TH.20",
    "income_4th20":   "SI.DST.04TH.20",
    "income_3rd20":   "SI.DST.03RD.20",
    "income_2nd20":   "SI.DST.02ND.20",
    "income_bot20":   "SI.DST.FRST.20",
    "income_top10":   "SI.DST.10TH.10",
    "income_bot10":   "SI.DST.FRST.10",
}


def _wb_fetch(indicator: str) -> dict:
    """Fetch most recent non-null value per country from World Bank API."""
    codes = ";".join(WB_ISO3.values())
    url = f"{WB_BASE}/country/{codes}/indicator/{indicator}"
    params = {"format": "json", "per_page": 1000, "date": "2010:2024", "mrv": 5}
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        pages = r.json()
        if len(pages) < 2 or not pages[1]:
            return {}
        rev = {v: k for k, v in WB_ISO3.items()}
        best = {}
        for item in pages[1]:
            if item["value"] is None:
                continue
            name = rev.get(item["countryiso3code"])
            if not name:
                continue
            yr = int(item["date"])
            if name not in best or yr > best[name][1]:
                best[name] = (float(item["value"]), yr)
        return best
    except Exception as e:
        print(f"  [WB warn] {indicator}: {e}")
        return {}


def fetch_worldbank() -> pd.DataFrame:
    print("Fetching World Bank data…")
    data = {c: {"country": c} for c in COUNTRIES}
    for col, ind in WB_INDICATORS.items():
        print(f"  • {col}")
        series = _wb_fetch(ind)
        for c in COUNTRIES:
            if c in series:
                data[c][col] = series[c][0]
                data[c][f"{col}_year"] = series[c][1]
            else:
                data[c][col] = np.nan

    df = pd.DataFrame(list(data.values())).set_index("country")

    # Apply Gini fallbacks where needed
    for c in COUNTRIES:
        if pd.isna(df.loc[c, "gini"]) and c in GINI_FALLBACK:
            df.loc[c, "gini"] = GINI_FALLBACK[c]
            df.loc[c, "gini_year"] = "est."
            print(f"  [fallback] Gini for {c}: {GINI_FALLBACK[c]}")
    return df


def build_master(df: pd.DataFrame) -> pd.DataFrame:
    """Merge curated WID/CS/OECD data into the master DataFrame."""
    for c in COUNTRIES:
        if c in WID_INCOME:
            t1, t10, m40, b50 = WID_INCOME[c]
            df.loc[c, "wid_inc_top1"]  = t1
            df.loc[c, "wid_inc_top10"] = t10
            df.loc[c, "wid_inc_mid40"] = m40
            df.loc[c, "wid_inc_bot50"] = b50
        if c in WID_WEALTH:
            t1, t10, m40, b50 = WID_WEALTH[c]
            df.loc[c, "wid_wlth_top1"]  = t1
            df.loc[c, "wid_wlth_top10"] = t10
            df.loc[c, "wid_wlth_mid40"] = m40
            df.loc[c, "wid_wlth_bot50"] = b50
        df.loc[c, "poverty"]     = POVERTY.get(c, np.nan)
        df.loc[c, "mean_wealth"] = WEALTH_MEAN.get(c, np.nan)
        df.loc[c, "med_wealth"]  = WEALTH_MEDIAN.get(c, np.nan)

    df["wealth_mm_ratio"] = df["mean_wealth"] / df["med_wealth"]
    return df


# ════════════════════════════════════════════════════════════════════════════
# DISTRIBUTION MODELLING
# ════════════════════════════════════════════════════════════════════════════

def lognormal_sigma(gini_0_100: float) -> float:
    """Convert Gini (0–100 scale) to log-normal sigma.
    G = 2·Φ(σ/√2) − 1  →  σ = √2·Φ⁻¹((G+1)/2)
    """
    g = np.clip(gini_0_100 / 100, 0.01, 0.99)
    return np.sqrt(2) * sp_stats.norm.ppf((g + 1) / 2)


def lorenz_curve(sigma: float, n: int = 300):
    """Parametric Lorenz curve for log-normal distribution."""
    p = np.linspace(0, 1, n)
    L = sp_stats.norm.cdf(sp_stats.norm.ppf(np.clip(p, 1e-6, 1 - 1e-6)) - sigma)
    return p, L


def income_density(gini: float, mean_income: float, n: int = 600):
    """Log-normal income density (x, pdf)."""
    sigma = lognormal_sigma(gini)
    mu = np.log(mean_income) - sigma ** 2 / 2
    dist = sp_stats.lognorm(s=sigma, scale=np.exp(mu))
    x_max = dist.ppf(0.97)
    x = np.linspace(dist.ppf(0.002), x_max, n)
    pdf = dist.pdf(x)
    return x, pdf


# ════════════════════════════════════════════════════════════════════════════
# PLOTLY DARK THEME
# ════════════════════════════════════════════════════════════════════════════

BG       = "#0D1117"
BG_PAPER = "#161B22"
GRID     = "#21262D"
TEXT     = "#E6EDF3"
SUBTEXT  = "#8B949E"

BASE_LAYOUT = dict(
    paper_bgcolor=BG_PAPER,
    plot_bgcolor=BG,
    font=dict(family="Inter, Helvetica Neue, Arial, sans-serif", color=TEXT, size=13),
    title_font=dict(size=17, color=TEXT),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
               tickfont=dict(color=SUBTEXT)),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
               tickfont=dict(color=SUBTEXT)),
    legend=dict(bgcolor="rgba(22,27,34,0.9)", bordercolor=GRID, borderwidth=1,
                font=dict(color=TEXT)),
    margin=dict(t=70, b=55, l=65, r=30),
    hoverlabel=dict(bgcolor="#2D333B", bordercolor=GRID,
                    font=dict(color=TEXT, size=13)),
)


def lay(**kwargs):
    """Merge kwargs into a copy of BASE_LAYOUT."""
    out = dict(BASE_LAYOUT)
    for k, v in kwargs.items():
        if k in ("xaxis", "yaxis") and k in out:
            merged = dict(out[k])
            merged.update(v)
            out[k] = merged
        else:
            out[k] = v
    return out


def to_html(fig: go.Figure) -> str:
    return fig.to_html(
        full_html=False, include_plotlyjs=False,
        config={"displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "responsive": True},
    )


H = 480   # default chart height


def hex_rgba(hex_color: str, alpha: float) -> str:
    """Convert a #RRGGBB hex color to rgba(r,g,b,alpha) string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ════════════════════════════════════════════════════════════════════════════
# CHARTS
# ════════════════════════════════════════════════════════════════════════════

def chart_gdp(df):
    sub = df["gdp_per_capita"].dropna().sort_values(ascending=True)
    fig = go.Figure()
    for c, val in sub.items():
        focal = c in FOCAL
        fig.add_trace(go.Bar(
            x=[val], y=[SHORT[c]], orientation="h",
            marker=dict(color=COLORS[c], opacity=1.0 if focal else 0.5,
                        line=dict(color="white", width=1.5) if focal else dict(width=0)),
            showlegend=False,
            hovertemplate=f"<b>{c}</b><br>GDP/capita: $%{{x:,.0f}}<extra></extra>",
        ))
        fig.add_annotation(x=val, y=SHORT[c], text=f"${val:,.0f}",
                           showarrow=False, xanchor="left", xshift=8,
                           font=dict(color=TEXT, size=12))
    fig.update_layout(**lay(
        title="<b>GDP per Capita</b> — Current USD (World Bank)",
        height=H,
        xaxis=dict(tickformat="$,.0f", title="USD"),
        yaxis=dict(categoryorder="total ascending"),
    ))
    return to_html(fig)


def chart_gini(df):
    sub = df["gini"].dropna().sort_values(ascending=True)
    fig = go.Figure()
    for y0, y1, color, label in [
        (0, 27,  "rgba(0,245,212,0.07)",  "Low"),
        (27, 36, "rgba(255,159,28,0.07)", "Moderate"),
        (36, 60, "rgba(255,77,109,0.07)", "High"),
    ]:
        fig.add_vrect(x0=y0, x1=y1, fillcolor=color, line_width=0,
                      annotation_text=label,
                      annotation_position="top",
                      annotation=dict(font=dict(color=SUBTEXT, size=11)))
    for c, val in sub.items():
        focal = c in FOCAL
        yr = df.loc[c, "gini_year"] if "gini_year" in df.columns else ""
        yr_str = f" ({yr})" if yr and yr != "est." else (" (est.)" if yr == "est." else "")
        fig.add_trace(go.Bar(
            x=[val], y=[SHORT[c]], orientation="h",
            marker=dict(color=COLORS[c], opacity=1.0 if focal else 0.5,
                        line=dict(color="white", width=1.5) if focal else dict(width=0)),
            showlegend=False,
            hovertemplate=f"<b>{c}</b><br>Gini: %{{x:.1f}}{yr_str}<extra></extra>",
        ))
        fig.add_annotation(x=val, y=SHORT[c], text=f"{val:.1f}{yr_str}",
                           showarrow=False, xanchor="left", xshift=6,
                           font=dict(color=TEXT, size=12))
    fig.update_layout(**lay(
        title="<b>Gini Coefficient</b> — Post-Transfer Household Income (World Bank)",
        height=H,
        xaxis=dict(title="Gini Index  (0 = perfect equality · 100 = maximum inequality)"),
        yaxis=dict(categoryorder="total ascending"),
    ))
    return to_html(fig)


def chart_quintiles(df):
    cols = ["income_bot20", "income_2nd20", "income_3rd20", "income_4th20", "income_top20"]
    labels = ["Bottom 20%", "2nd Quintile", "Middle 20%", "4th Quintile", "Top 20%"]
    q_colors = ["#00F5D4", "#4CC9F0", "#7B2FBE", "#FF9F1C", "#FF4D6D"]

    sub = df[cols].dropna()
    if sub.empty:
        return "<p style='color:#8B949E;padding:24px'>No quintile data available from World Bank API.</p>"

    sub = sub.sort_values("income_bot20", ascending=True)   # lowest bottom-quintile share → bottom of chart; most equal → top
    ys = [SHORT[c] for c in sub.index]

    fig = go.Figure()
    for col, lbl, color in zip(cols, labels, q_colors):
        fig.add_trace(go.Bar(
            x=sub[col], y=ys, orientation="h", name=lbl,
            marker_color=color,
            hovertemplate=f"<b>{lbl}</b>: %{{x:.1f}}%<extra></extra>",
        ))
    fig.update_layout(**lay(
        title="<b>Income Share by Quintile</b> — Post-Transfer (World Bank)",
        height=H, barmode="stack",
        xaxis=dict(title="Share of Total Income (%)", ticksuffix="%"),
        legend=dict(**BASE_LAYOUT["legend"], traceorder="normal"),
    ))
    return to_html(fig)


def chart_wid_income(df):
    metrics = [
        ("wid_inc_top1",  "Top 1%",     "#FF4D6D"),
        ("wid_inc_top10", "Top 10%",    "#FF9F1C"),
        ("wid_inc_mid40", "Middle 40%", "#4CC9F0"),
        ("wid_inc_bot50", "Bottom 50%", "#00F5D4"),
    ]
    sub = df[[m[0] for m in metrics]].dropna()
    sub = sub.sort_values("wid_inc_bot50", ascending=False)
    xs = [SHORT[c] for c in sub.index]

    fig = go.Figure()
    for col, lbl, color in metrics:
        fig.add_trace(go.Bar(
            x=xs, y=sub[col], name=lbl, marker_color=color,
            hovertemplate=f"<b>{lbl}</b>: %{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(**lay(
        title="<b>Pre-Tax National Income Shares</b> — World Inequality Database, 2022",
        height=H, barmode="group",
        yaxis=dict(title="Share of National Income (%)", ticksuffix="%"),
    ))
    return to_html(fig)


def chart_lorenz(df):
    sub = df["gini"].dropna()
    fig = go.Figure()

    # Perfect equality reference
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dot", color="rgba(255,255,255,0.2)", width=1.5),
        name="Perfect Equality", hoverinfo="skip",
    ))

    for c in sub.index:
        sigma = lognormal_sigma(sub[c])
        p, L = lorenz_curve(sigma)
        focal = c in FOCAL
        fig.add_trace(go.Scatter(
            x=p, y=L, mode="lines", name=SHORT[c],
            line=dict(color=COLORS[c], width=3.5 if focal else 1.5),
            opacity=1.0 if focal else 0.55,
            hovertemplate=f"<b>{c}</b><br>Bottom %{{x:.0%}} earn %{{y:.1%}} of income<extra></extra>",
        ))

    # Shaded inequality area for focal countries
    for c in FOCAL:
        if c not in sub.index:
            continue
        sigma = lognormal_sigma(sub[c])
        p, L = lorenz_curve(sigma)
        hex_c = COLORS[c]
        fig.add_trace(go.Scatter(
            x=list(p) + list(p[::-1]),
            y=list(L) + list(p[::-1]),
            fill="toself",
            fillcolor=hex_rgba(COLORS[c], 0.1),
            line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))

    fig.update_layout(**lay(
        title="<b>Lorenz Curves</b> — Log-Normal Model Fitted to Gini (focal countries shaded)",
        height=H,
        xaxis=dict(title="Cumulative Population Share", tickformat=".0%"),
        yaxis=dict(title="Cumulative Income Share", tickformat=".0%"),
    ))
    return to_html(fig)


def chart_density(df):
    sub = df[["gini", "gdp_per_capita"]].dropna()
    fig = go.Figure()

    for c in sub.index:
        gini = sub.loc[c, "gini"]
        mean = sub.loc[c, "gdp_per_capita"]
        x, pdf = income_density(gini, mean)
        focal = c in FOCAL
        fig.add_trace(go.Scatter(
            x=x, y=pdf, mode="lines", name=SHORT[c],
            line=dict(color=COLORS[c], width=3.5 if focal else 1.5,
                      dash="solid" if focal else "dot"),
            fill="tozeroy" if focal else None,
            fillcolor=hex_rgba(COLORS[c], 0.1) if focal else None,
            opacity=1.0 if focal else 0.5,
            hovertemplate=f"<b>{c}</b><br>Income: $%{{x:,.0f}}<extra></extra>",
        ))

    fig.add_annotation(
        text="Heavier right tail → more skewed distribution",
        xref="paper", yref="paper", x=0.98, y=0.95,
        showarrow=False, font=dict(color=SUBTEXT, size=11), xanchor="right",
    )
    fig.update_layout(**lay(
        title="<b>Income Distribution Density</b> — Log-Normal Model (Gini + GDP/Capita)",
        height=H,
        xaxis=dict(title="Annual Income (USD, approximate)", tickformat="$,.0f"),
        yaxis=dict(title="Probability Density", showticklabels=False),
    ))
    return to_html(fig)


def chart_wealth_conc(df):
    metrics = [
        ("wid_wlth_top1",  "Top 1% Wealth",    "#FF4D6D"),
        ("wid_wlth_top10", "Top 10% Wealth",   "#FF9F1C"),
        ("wid_wlth_bot50", "Bottom 50% Wealth","#00F5D4"),
    ]
    sub = df[[m[0] for m in metrics]].dropna()
    sub = sub.sort_values("wid_wlth_top1", ascending=False)
    xs = [SHORT[c] for c in sub.index]

    fig = go.Figure()
    for col, lbl, color in metrics:
        fig.add_trace(go.Bar(
            x=xs, y=sub[col], name=lbl, marker_color=color,
            hovertemplate=f"<b>{lbl}</b>: %{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(**lay(
        title="<b>Wealth Concentration</b> — Credit Suisse Global Wealth Report 2023",
        height=H, barmode="group",
        yaxis=dict(title="Share of Total Net Wealth (%)", ticksuffix="%"),
    ))
    return to_html(fig)


def chart_wealth_skew(df):
    sub = df[["mean_wealth", "med_wealth", "wealth_mm_ratio"]].dropna()
    sub = sub.sort_values("wealth_mm_ratio", ascending=False)
    xs = [SHORT[c] for c in sub.index]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Mean vs Median Wealth per Adult (USD)", "Mean / Median Ratio"],
        column_widths=[0.65, 0.35],
        horizontal_spacing=0.1,
    )
    fig.add_trace(go.Bar(x=xs, y=sub["mean_wealth"], name="Mean Wealth",
                         marker_color="#4CC9F0",
                         hovertemplate="<b>Mean</b>: $%{y:,.0f}<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=xs, y=sub["med_wealth"], name="Median Wealth",
                         marker_color="#00F5D4",
                         hovertemplate="<b>Median</b>: $%{y:,.0f}<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Bar(
        x=xs, y=sub["wealth_mm_ratio"],
        marker_color=[COLORS[c] for c in sub.index],
        showlegend=False,
        hovertemplate="<b>Mean/Median</b>: %{y:.2f}×<extra></extra>",
    ), row=1, col=2)

    fig.update_layout(**lay(
        title="<b>Wealth Skewness</b> — Mean vs Median per Adult (Credit Suisse 2023)",
        height=H, barmode="group",
    ))
    fig.update_yaxes(tickformat="$,.0f", gridcolor=GRID, row=1, col=1)
    fig.update_yaxes(title_text="Ratio (↑ = more right-skewed)", gridcolor=GRID, row=1, col=2)
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID)
    for ann in fig.layout.annotations:
        ann.font.color = SUBTEXT
        ann.font.size = 13
    return to_html(fig)


def chart_scatter(df):
    sub = df[["gini", "gdp_per_capita", "poverty"]].dropna()
    fig = go.Figure()

    for c in sub.index:
        row = sub.loc[c]
        focal = c in FOCAL
        fig.add_trace(go.Scatter(
            x=[row["gdp_per_capita"]], y=[row["gini"]],
            mode="markers+text",
            marker=dict(
                size=max(row["poverty"] * 2.4, 12),
                color=COLORS[c],
                opacity=1.0 if focal else 0.7,
                line=dict(color="white", width=2) if focal else dict(width=0),
            ),
            text=[SHORT[c]], textposition="top center",
            textfont=dict(color=TEXT, size=11),
            name=c,
            hovertemplate=(
                f"<b>{c}</b><br>"
                f"GDP/capita: $%{{x:,.0f}}<br>"
                f"Gini: %{{y:.1f}}<br>"
                f"Poverty rate: {row['poverty']:.1f}%"
                "<extra></extra>"
            ),
        ))

    # Median reference lines
    for val, axis in [(sub["gdp_per_capita"].median(), "x"),
                      (sub["gini"].median(), "y")]:
        fig.add_shape(
            type="line",
            x0=val if axis == "x" else sub["gdp_per_capita"].min() * 0.88,
            x1=val if axis == "x" else sub["gdp_per_capita"].max() * 1.06,
            y0=val if axis == "y" else sub["gini"].min() * 0.88,
            y1=val if axis == "y" else sub["gini"].max() * 1.06,
            line=dict(color=GRID, width=1, dash="dot"),
        )

    fig.update_layout(**lay(
        title="<b>Gini vs GDP per Capita</b> — Bubble size = relative poverty rate",
        height=H + 40, showlegend=False,
        xaxis=dict(title="GDP per Capita (USD)", tickformat="$,.0f"),
        yaxis=dict(title="Gini Coefficient"),
    ))
    return to_html(fig)


def chart_poverty(df):
    sub = df["poverty"].dropna().sort_values(ascending=True)
    avg = sub.mean()
    fig = go.Figure()
    for c, val in sub.items():
        focal = c in FOCAL
        fig.add_trace(go.Bar(
            x=[val], y=[SHORT[c]], orientation="h",
            marker=dict(color=COLORS[c], opacity=1.0 if focal else 0.5,
                        line=dict(color="white", width=1.5) if focal else dict(width=0)),
            showlegend=False,
            hovertemplate=f"<b>{c}</b><br>Poverty: %{{x:.1f}}%<extra></extra>",
        ))
        fig.add_annotation(x=val, y=SHORT[c], text=f"{val:.1f}%",
                           showarrow=False, xanchor="left", xshift=6,
                           font=dict(color=TEXT, size=12))
    fig.add_vline(x=avg, line_dash="dot",
                  line_color="rgba(255,255,255,0.35)",
                  annotation_text=f"Group avg {avg:.1f}%",
                  annotation_position="top right",
                  annotation_font=dict(color=SUBTEXT, size=11))
    fig.update_layout(**lay(
        title="<b>Relative Poverty Rate</b> — Income < 50% of National Median (OECD, ~2021)",
        height=H,
        xaxis=dict(title="% of Population", ticksuffix="%"),
        yaxis=dict(categoryorder="total ascending"),
    ))
    return to_html(fig)


def chart_radar(df):
    metric_defs = [
        ("gdp_per_capita",  False, "GDP per Capita"),
        ("gini",            True,  "Income Equality"),
        ("poverty",         True,  "Low Poverty"),
        ("wid_inc_bot50",   False, "Bottom 50%<br>Income Share"),
        ("wid_wlth_bot50",  False, "Bottom 50%<br>Wealth Share"),
        ("wealth_mm_ratio", True,  "Wealth Equity<br>(low Mean÷Med)"),
    ]

    # Normalize across all 8 countries (0–100, 100 = best)
    norm = {}
    for col, invert, lbl in metric_defs:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        mn, mx = vals.min(), vals.max()
        for c, v in vals.items():
            score = (v - mn) / (mx - mn) * 100 if mx != mn else 50.0
            if invert:
                score = 100 - score
            norm.setdefault(c, {})[lbl] = score

    labels = [m[2] for m in metric_defs if m[0] in df.columns]

    fig = go.Figure()
    for country in ["United States", "Canada", "Norway"]:
        if country not in norm:
            continue
        scores = [norm[country].get(lbl, 50) for lbl in labels]
        fig.add_trace(go.Scatterpolar(
            r=scores + [scores[0]],
            theta=labels + [labels[0]],
            name=country,
            line=dict(color=COLORS[country], width=3),
            fill="toself",
            fillcolor=hex_rgba(COLORS[country], 0.13),
            hovertemplate="<b>" + country + "</b><br>%{theta}: %{r:.0f}/100<extra></extra>",
        ))

    fig.update_layout(
        **{k: v for k, v in BASE_LAYOUT.items() if k not in ("xaxis", "yaxis")},
        polar=dict(
            bgcolor=BG,
            radialaxis=dict(visible=True, range=[0, 100],
                            gridcolor=GRID, linecolor=GRID,
                            tickfont=dict(color=SUBTEXT, size=10)),
            angularaxis=dict(gridcolor=GRID, linecolor=GRID,
                             tickfont=dict(color=TEXT, size=12)),
        ),
        title=(
            "<b>Equality & Prosperity Scorecard</b> — USA · Canada · Norway<br>"
            "<sup>100 = best among all 8 countries on each dimension</sup>"
        ),
        height=520,
    )
    return to_html(fig)


def chart_top1_income_vs_wealth(df):
    """Dot-plot comparing top 1% income share vs wealth share."""
    sub = df[["wid_inc_top1", "wid_wlth_top1"]].dropna()
    fig = go.Figure()

    for c in sub.index:
        inc = sub.loc[c, "wid_inc_top1"]
        wlth = sub.loc[c, "wid_wlth_top1"]
        focal = c in FOCAL
        # Line connecting income to wealth
        fig.add_trace(go.Scatter(
            x=[inc, wlth], y=[SHORT[c], SHORT[c]],
            mode="lines",
            line=dict(color=COLORS[c], width=2 if focal else 1),
            opacity=1.0 if focal else 0.5,
            showlegend=False, hoverinfo="skip",
        ))
        # Income dot
        fig.add_trace(go.Scatter(
            x=[inc], y=[SHORT[c]], mode="markers",
            marker=dict(color=COLORS[c], size=12 if focal else 9,
                        symbol="circle",
                        line=dict(color="white", width=1.5) if focal else dict(width=0)),
            name=f"{SHORT[c]} Income" if focal else None,
            showlegend=False,
            hovertemplate=f"<b>{c}</b><br>Top 1% Income: %{{x:.1f}}%<extra></extra>",
        ))
        # Wealth dot
        fig.add_trace(go.Scatter(
            x=[wlth], y=[SHORT[c]], mode="markers",
            marker=dict(color=COLORS[c], size=12 if focal else 9,
                        symbol="diamond",
                        line=dict(color="white", width=1.5) if focal else dict(width=0)),
            showlegend=False,
            hovertemplate=f"<b>{c}</b><br>Top 1% Wealth: %{{x:.1f}}%<extra></extra>",
        ))

    # Legend items
    for sym, lbl in [("circle", "Income Share"), ("diamond", "Wealth Share")]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color="#8B949E", size=10, symbol=sym),
            name=lbl,
        ))

    fig.update_layout(**lay(
        title="<b>Top 1% — Income Share vs Wealth Share</b><br>"
              "<sup>Circle = pre-tax income · Diamond = net wealth · WID.world / Credit Suisse 2023</sup>",
        height=H,
        xaxis=dict(title="Share (%)", ticksuffix="%"),
        yaxis=dict(categoryorder="array",
                   categoryarray=[SHORT[c] for c in
                                  sub.sort_values("wid_wlth_top1").index]),
    ))
    return to_html(fig)


# ════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ════════════════════════════════════════════════════════════════════════════

def build_summary_table(df):
    col_defs = [
        ("gdp_per_capita",  "GDP/Capita",            "${:,.0f}"),
        ("gini",            "Gini",                  "{:.1f}"),
        ("wid_inc_top1",    "Top 1%<br>Income",      "{:.1f}%"),
        ("wid_inc_bot50",   "Bottom 50%<br>Income",  "{:.1f}%"),
        ("wid_wlth_top1",   "Top 1%<br>Wealth",      "{:.1f}%"),
        ("med_wealth",      "Median<br>Wealth",      "${:,.0f}"),
        ("poverty",         "Poverty<br>Rate",       "{:.1f}%"),
    ]
    sub = df.copy()
    if "gini" in sub.columns:
        sub = sub.sort_values("gini", na_position="last")

    header = "<tr>" + "".join(f"<th>{h}</th>" for _, h, _ in
                               [("", "Country", "")] + col_defs) + "</tr>"
    rows = ""
    for c in sub.index:
        focal = c in FOCAL
        cls = "row-focal" if focal else "row-peer"
        cells = f"<td><strong>{c}</strong></td>"
        for col, _, fmt in col_defs:
            val = sub.loc[c, col] if col in sub.columns else np.nan
            cells += "<td>N/A</td>" if pd.isna(val) else f"<td>{fmt.format(val)}</td>"
        rows += f'<tr class="{cls}">{cells}</tr>\n'

    return f"""
    <div class="table-responsive">
      <table class="summary-table">
        <thead>{header}</thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""


# ════════════════════════════════════════════════════════════════════════════
# HERO STAT CARDS
# ════════════════════════════════════════════════════════════════════════════

def stat_card(country, df):
    color = COLORS[country]

    def fmt(col, tmpl):
        try:
            v = df.loc[country, col]
            return tmpl.format(v) if not pd.isna(v) else "N/A"
        except Exception:
            return "N/A"

    rows = [
        ("GDP per Capita",      fmt("gdp_per_capita", "${:,.0f}")),
        ("Gini Index",          fmt("gini",            "{:.1f}")),
        ("Top 1% Income",       fmt("wid_inc_top1",   "{:.1f}%")),
        ("Top 1% Wealth",       fmt("wid_wlth_top1",  "{:.1f}%")),
        ("Poverty Rate",        fmt("poverty",         "{:.1f}%")),
        ("Median Wealth",       fmt("med_wealth",      "${:,.0f}")),
    ]
    row_html = "".join(
        f'<div class="sr"><span class="sl">{lbl}</span>'
        f'<span class="sv">{val}</span></div>'
        for lbl, val in rows
    )
    return f"""
    <div class="stat-card" style="border-top:3px solid {color}">
      <div class="sc-head" style="color:{color}">{country}</div>
      {row_html}
    </div>"""


# ════════════════════════════════════════════════════════════════════════════
# HTML ASSEMBLY
# ════════════════════════════════════════════════════════════════════════════

def make_section(sid, title, desc, chart_html, source=""):
    badge = (f'<span class="src-badge">Source: {source}</span>' if source else "")
    return f"""
<section id="{sid}" class="rpt-section">
  <div class="sec-hdr">
    <h2 class="sec-title">{title}</h2>
    {badge}
  </div>
  <p class="sec-desc">{desc}</p>
  <div class="chart-card">{chart_html}</div>
</section>"""


def build_html(df, charts, table_html):
    cards_html = "".join(stat_card(c, df)
                         for c in ["United States", "Canada", "Norway"])

    nav_items = [
        ("overview",       "Overview"),
        ("gdp",            "GDP per Capita"),
        ("gini",           "Gini Coefficient"),
        ("quintiles",      "Income Quintiles"),
        ("wid-income",     "Top Income Shares"),
        ("income-wealth",  "Income vs Wealth Top 1%"),
        ("lorenz",         "Lorenz Curves"),
        ("density",        "Income Density"),
        ("wealth-conc",    "Wealth Concentration"),
        ("wealth-skew",    "Wealth Skewness"),
        ("scatter",        "Gini vs GDP"),
        ("poverty",        "Poverty Rates"),
        ("radar",          "Radar Scorecard"),
        ("summary",        "Summary Table"),
        ("methodology",    "Methodology"),
    ]
    nav_html = "\n".join(
        f'<a href="#{sid}" class="nav-link">{lbl}</a>'
        for sid, lbl in nav_items
    )

    today = datetime.now().strftime("%B %d, %Y")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Income & Wealth Inequality — USA · Canada · Norway</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js" charset="utf-8"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#0D1117;--bg-card:#161B22;--bg-card2:#1C2128;
  --border:#30363D;--text:#E6EDF3;--muted:#8B949E;
  --usa:#FF4D6D;--can:#FF9F1C;--nor:#00F5D4;
  --font:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;
}}
html{{scroll-behavior:smooth}}
body{{background:var(--bg);color:var(--text);font-family:var(--font);
  font-size:15px;line-height:1.6;display:flex;min-height:100vh}}

/* ── Sidebar ── */
#sidebar{{width:220px;min-width:220px;background:var(--bg-card);
  border-right:1px solid var(--border);height:100vh;position:sticky;
  top:0;overflow-y:auto;padding:24px 0;flex-shrink:0;
  scrollbar-width:thin;scrollbar-color:var(--border) transparent}}
.sb-title{{font-size:11px;font-weight:600;letter-spacing:.1em;
  text-transform:uppercase;color:var(--muted);padding:0 20px 14px}}
.nav-link{{display:block;padding:7px 20px;color:var(--muted);
  text-decoration:none;font-size:13px;
  border-left:2px solid transparent;transition:all .15s}}
.nav-link:hover{{color:var(--text);background:rgba(255,255,255,.04);
  border-left-color:var(--nor)}}
.nav-link.active{{color:var(--nor);border-left-color:var(--nor);
  background:rgba(0,245,212,.06);font-weight:500}}

/* ── Main ── */
#main{{flex:1;overflow-y:auto;height:100vh}}

/* ── Hero ── */
.hero{{background:linear-gradient(135deg,#0D1117 0%,#161B22 60%,#0D1117 100%);
  border-bottom:1px solid var(--border);padding:60px 48px 44px;position:relative;overflow:hidden}}
.hero::before{{content:'';position:absolute;inset:0;
  background:radial-gradient(ellipse 900px 500px at 75% 50%,rgba(0,245,212,.05) 0%,transparent 70%),
             radial-gradient(ellipse 600px 400px at 15% 90%,rgba(255,77,109,.04) 0%,transparent 70%);
  pointer-events:none}}
.hero-eyebrow{{font-size:12px;font-weight:600;letter-spacing:.15em;
  text-transform:uppercase;color:var(--nor);margin-bottom:14px}}
.hero h1{{font-size:44px;font-weight:700;letter-spacing:-.5px;
  line-height:1.15;margin-bottom:16px}}
.hero h1 span{{background:linear-gradient(90deg,var(--usa),var(--can),var(--nor));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}}
.hero-sub{{font-size:16px;color:var(--muted);max-width:700px;margin-bottom:40px;line-height:1.75}}
.hero-sub strong{{color:var(--text)}}

/* ── Stat Cards ── */
.stat-cards{{display:flex;gap:18px;flex-wrap:wrap}}
.stat-card{{background:var(--bg-card2);border:1px solid var(--border);
  border-radius:10px;padding:20px 22px;min-width:195px;flex:1}}
.sc-head{{font-size:12px;font-weight:700;letter-spacing:.06em;
  text-transform:uppercase;margin-bottom:14px}}
.sr{{display:flex;justify-content:space-between;align-items:center;
  padding:5px 0;border-bottom:1px solid rgba(48,54,61,.5);font-size:13px}}
.sr:last-child{{border-bottom:none}}
.sl{{color:var(--muted)}}
.sv{{font-weight:600;font-variant-numeric:tabular-nums}}

/* ── Content ── */
.content{{padding:0 48px 80px;max-width:1320px}}

/* ── Sections ── */
.rpt-section{{padding:52px 0 0}}
.sec-hdr{{display:flex;align-items:baseline;gap:14px;margin-bottom:8px}}
.sec-title{{font-size:24px;font-weight:700;letter-spacing:-.3px}}
.src-badge{{font-size:11px;font-weight:500;color:var(--muted);
  background:var(--bg-card2);border:1px solid var(--border);
  border-radius:20px;padding:3px 10px;white-space:nowrap}}
.sec-desc{{color:var(--muted);font-size:14px;max-width:820px;
  margin-bottom:20px;line-height:1.75}}
.chart-card{{background:var(--bg-card);border:1px solid var(--border);
  border-radius:12px;padding:8px;overflow:hidden}}
.divider{{height:1px;background:var(--border);margin-top:52px}}

/* ── Summary Table ── */
.table-responsive{{background:var(--bg-card);border:1px solid var(--border);
  border-radius:12px;overflow-x:auto}}
.summary-table{{width:100%;border-collapse:collapse;font-size:13px}}
.summary-table th{{background:var(--bg-card2);color:var(--muted);
  font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;
  padding:12px 16px;text-align:left;border-bottom:1px solid var(--border)}}
.summary-table td{{padding:11px 16px;border-bottom:1px solid rgba(48,54,61,.4);
  font-variant-numeric:tabular-nums}}
.summary-table tr:hover td{{background:rgba(255,255,255,.02)}}
.row-focal td{{color:var(--text)}}
.row-focal td:first-child{{font-weight:600}}
.row-peer td{{color:var(--muted)}}

/* ── Methodology ── */
.meth-box{{background:var(--bg-card);border:1px solid var(--border);
  border-radius:12px;padding:32px}}
.meth-box h3{{font-size:15px;font-weight:600;margin-bottom:8px;color:var(--text)}}
.meth-box p,.meth-box li{{color:var(--muted);font-size:13px;line-height:1.85}}
.meth-box ul{{padding-left:20px;margin-top:6px}}
.meth-box p+h3,.meth-box ul+h3{{margin-top:24px}}
.src-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));
  gap:14px;margin-top:14px}}
.src-item{{background:var(--bg-card2);border:1px solid var(--border);
  border-radius:8px;padding:14px 16px}}
.src-name{{font-weight:600;font-size:13px;color:var(--text);margin-bottom:4px}}
.src-detail{{font-size:12px;color:var(--muted);line-height:1.7}}

/* ── Footer ── */
.rpt-footer{{border-top:1px solid var(--border);margin-top:64px;
  padding:22px 48px;color:var(--muted);font-size:12px;
  display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px}}

/* ── Scrollbar ── */
::-webkit-scrollbar{{width:6px;height:6px}}
::-webkit-scrollbar-track{{background:transparent}}
::-webkit-scrollbar-thumb{{background:var(--border);border-radius:3px}}

/* ── Mobile / collapsible nav ── */
#nav-toggle{{display:none;position:fixed;top:14px;left:14px;z-index:1001;
  background:var(--bg-card);border:1px solid var(--border);border-radius:8px;
  padding:8px 13px;color:var(--text);font-size:18px;line-height:1;
  cursor:pointer;transition:background .15s}}
#nav-toggle:hover{{background:var(--border)}}
#sidebar-overlay{{display:none;position:fixed;inset:0;
  background:rgba(0,0,0,.55);z-index:999;backdrop-filter:blur(2px)}}
#sidebar-overlay.open{{display:block}}

@media(max-width:768px){{
  body{{display:block}}
  #nav-toggle{{display:block}}
  #sidebar{{position:fixed;top:0;left:-240px;height:100vh;z-index:1000;
    width:220px;transition:left .25s ease;box-shadow:none}}
  #sidebar.open{{left:0;box-shadow:4px 0 24px rgba(0,0,0,.6)}}
  #main{{height:auto;overflow-y:unset}}
  .hero{{padding:64px 20px 32px}}
  .stat-cards{{flex-direction:column}}
  .content{{padding:0 16px 60px}}
  .section-title{{font-size:18px}}
  .chart-card{{padding:4px}}
  .report-footer{{padding:20px;flex-direction:column;gap:4px}}
}}
</style>
</head>
<body>
<button id="nav-toggle" aria-label="Toggle navigation">&#9776;</button>
<div id="sidebar-overlay"></div>

<!-- SIDEBAR -->
<nav id="sidebar">
  <div class="sb-title">Navigation</div>
  {nav_html}
</nav>

<!-- MAIN -->
<div id="main">

  <!-- HERO -->
  <section id="overview" class="hero">
    <div class="hero-eyebrow">Global Inequality Analysis · {datetime.now().year}</div>
    <h1>Income &amp; Wealth<br><span>Inequality</span></h1>
    <p class="hero-sub">
      A comparative deep dive into the structure of income and wealth distribution across
      <strong>USA</strong>, <strong>Canada</strong>, and <strong>Norway</strong>,
      benchmarked against five peer OECD nations — Sweden, Denmark, Germany,
      United Kingdom, and France.
    </p>
    <div class="stat-cards">{cards_html}</div>
  </section>

  <!-- CONTENT -->
  <div class="content">

    {make_section("gdp","GDP per Capita",
      "Gross Domestic Product per capita in current US dollars measures average material "
      "living standards. GDP per capita is a <em>mean</em> — it does not reveal how income "
      "is distributed across the population.",
      charts["gdp"], "World Bank — NY.GDP.PCAP.CD")}
    <div class="divider"></div>

    {make_section("gini","Gini Coefficient",
      "The Gini coefficient measures income inequality within a country on a scale of 0 "
      "(perfect equality) to 100 (all income to one person). These figures derive from "
      "household surveys on post-transfer disposable income. Background shading marks "
      "Low (&lt;27), Moderate (27–36), and High (&gt;36) inequality bands.",
      charts["gini"], "World Bank — SI.POV.GINI")}
    <div class="divider"></div>

    {make_section("quintiles","Income Share by Quintile",
      "Income distributed across five equally-sized population groups. The wider the gap "
      "between the top and bottom quintile shares, the greater the income spread. Countries "
      "are sorted by bottom-quintile share (most equal at top).",
      charts["quintiles"], "World Bank — SI.DST.* series")}
    <div class="divider"></div>

    {make_section("wid-income","Pre-Tax National Income Shares",
      "Top income concentration from the World Inequality Database, built on tax records "
      "and national accounts — capturing market income before redistribution. These figures "
      "are typically higher than post-transfer survey Gini measures, revealing the raw "
      "concentration of economic output.",
      charts["wid_income"], "World Inequality Database — wid.world, 2022")}
    <div class="divider"></div>

    {make_section("income-wealth","Top 1% — Income Share vs Wealth Share",
      "Wealth is almost always more concentrated than income. This chart shows — for each "
      "country — how the top 1%'s share of pre-tax income (circle) compares to their share "
      "of total net wealth (diamond). The gap between the two reveals how capital accumulation "
      "amplifies inequality beyond what income alone suggests.",
      charts["top1_dot"], "WID.world (income) · Credit Suisse GWR 2023 (wealth)")}
    <div class="divider"></div>

    {make_section("lorenz","Lorenz Curves",
      "The Lorenz curve plots the cumulative income share earned by the bottom x% of the "
      "population. The further the curve bows below the 45° equality line, the greater the "
      "inequality — and the larger the Gini coefficient. Curves are derived from a log-normal "
      "model fitted to each country's Gini; focal countries are shaded.",
      charts["lorenz"], "Parametric log-normal · World Bank Gini")}
    <div class="divider"></div>

    {make_section("density","Income Distribution Density",
      "Estimated probability density of income using a log-normal model calibrated to each "
      "country's Gini and GDP per capita. A narrow, tall peak signals a compressed "
      "distribution; a long right tail signals high inequality. Focal countries shown with "
      "filled areas; comparison countries as dashed lines.",
      charts["density"], "Parametric log-normal model")}
    <div class="divider"></div>

    {make_section("wealth-conc","Wealth Concentration",
      "Net personal wealth is significantly more concentrated than income. These figures "
      "show what share of total household wealth is held by the wealthiest 1%, the top 10%, "
      "and the bottom half of the population. Countries sorted by top 1% wealth share.",
      charts["wealth_conc"], "Credit Suisse Global Wealth Report 2023 · WID.world")}
    <div class="divider"></div>

    {make_section("wealth-skew","Wealth Skewness — Mean vs Median",
      "When mean wealth greatly exceeds median wealth, the distribution is right-skewed — "
      "a small number of ultra-wealthy individuals pull the average far above the typical "
      "person's holdings. The mean/median ratio (right panel) directly quantifies this skew.",
      charts["wealth_skew"], "Credit Suisse Global Wealth Report 2023")}
    <div class="divider"></div>

    {make_section("scatter","Gini vs GDP per Capita",
      "Does higher income mean lower inequality? Not automatically. Bubble size represents "
      "the relative poverty rate. The Nordic countries demonstrate that high prosperity and "
      "low inequality can coexist — challenging the notion of an inevitable trade-off.",
      charts["scatter"], "World Bank · OECD")}
    <div class="divider"></div>

    {make_section("poverty","Relative Poverty Rates",
      "The share of the population with income below 50% of the national median — the OECD's "
      "standard relative poverty threshold. This captures how many people fall significantly "
      "below the living standards of their own society.",
      charts["poverty"], "OECD Income Distribution Database, ~2021")}
    <div class="divider"></div>

    {make_section("radar","Equality &amp; Prosperity Scorecard",
      "A normalized radar chart comparing the three focal countries across six dimensions. "
      "Each axis is scored 0–100 relative to all eight countries in the dataset, where "
      "100 = best performing on that dimension. Higher overall area = more equal and prosperous.",
      charts["radar"], "Multi-source composite")}
    <div class="divider"></div>

    <section id="summary" class="rpt-section">
      <div class="sec-hdr"><h2 class="sec-title">Summary Table</h2></div>
      <p class="sec-desc">
        Key inequality metrics for all eight countries, sorted by Gini coefficient (most equal first).
        Focal countries — USA, Canada, Norway — are highlighted.
      </p>
      {table_html}
    </section>
    <div class="divider"></div>

    <section id="methodology" class="rpt-section">
      <div class="sec-hdr"><h2 class="sec-title">Methodology &amp; Data Sources</h2></div>
      <div class="meth-box">
        <h3>Data Sources</h3>
        <div class="src-grid">
          <div class="src-item">
            <div class="src-name">World Bank Open Data</div>
            <div class="src-detail">GDP per capita (NY.GDP.PCAP.CD), Gini index (SI.POV.GINI),
            income quintile and decile shares. Fetched live via REST API; most recent available
            year per country (search window 2010–2024, up to 5 most-recent values).</div>
          </div>
          <div class="src-item">
            <div class="src-name">World Inequality Database (WID.world)</div>
            <div class="src-detail">Pre-tax national income shares (Top 1%, Top 10%, Middle 40%,
            Bottom 50%). Constructed from tax records, national accounts, and household surveys.
            Reference year: 2022. Data embedded from published WID figures.</div>
          </div>
          <div class="src-item">
            <div class="src-name">Credit Suisse Global Wealth Report 2023</div>
            <div class="src-detail">Net personal wealth shares (Top 1%, Top 10%, Middle 40%,
            Bottom 50%), mean and median wealth per adult in USD. Reference period: mid-2022.
            Data embedded from published CS Research Institute figures.</div>
          </div>
          <div class="src-item">
            <div class="src-name">OECD Income Distribution Database</div>
            <div class="src-detail">Relative poverty rates defined as income below 50% of national
            median. Most recent available year, approximately 2021–2022. Data embedded from
            OECD.Stat published figures.</div>
          </div>
        </div>
        <h3>Income Distribution Model</h3>
        <p>The Lorenz curves and income density charts use a parametric log-normal model.
        For a log-normal distribution, the Gini G satisfies <em>G = 2Φ(σ/√2) − 1</em>,
        where Φ is the standard normal CDF. Given observed Gini G and mean income M:
        <em>σ = √2 · Φ⁻¹((G+1)/2)</em> and <em>μ = ln(M) − σ²/2</em>.
        <strong>Note:</strong> log-normal distributions underestimate the extreme upper tail of
        real income distributions, which empirically follow a Pareto distribution above the
        ~90th percentile. These charts are illustrative of distributional shape, not precise
        estimates of tail income.</p>
        <h3>Country Selection</h3>
        <p>The United States, Canada, and Norway are the primary focus. Five additional OECD
        peer nations (Sweden, Denmark, Germany, United Kingdom, France) are included to
        provide distributional context and anchor the relative position of focal countries
        within the high-income world.</p>
        <h3>Limitations</h3>
        <ul>
          <li>World Bank Gini figures rely on household surveys, which systematically
          under-capture top incomes. WID figures using tax records typically show higher
          concentration.</li>
          <li>Wealth data comparability is limited by differences in national measurement
          methodologies, pension treatment, and survey coverage.</li>
          <li>Data vintages vary across indicators and countries; all comparisons are
          approximate.</li>
          <li>GDP per capita is used as a proxy for mean household income in the log-normal
          model; this slightly overstates true mean household income.</li>
        </ul>
      </div>
    </section>

  </div><!-- /content -->

  <footer class="rpt-footer">
    <span>Income &amp; Wealth Inequality Report &mdash; Generated {today}</span>
    <span>Data: World Bank &middot; WID.world &middot; Credit Suisse &middot; OECD</span>
  </footer>

</div><!-- /main -->

<script>
// ── Scrollspy ──────────────────────────────────────────────
const sections = document.querySelectorAll('section[id]');
const navLinks  = document.querySelectorAll('#sidebar .nav-link');
const observer  = new IntersectionObserver(entries => {{
  entries.forEach(e => {{
    if (e.isIntersecting) {{
      const id = e.target.getAttribute('id');
      navLinks.forEach(l =>
        l.classList.toggle('active', l.getAttribute('href') === '#' + id)
      );
    }}
  }});
}}, {{ rootMargin: '-15% 0px -65% 0px' }});
sections.forEach(s => observer.observe(s));

// ── Collapsible sidebar ────────────────────────────────────
const toggle  = document.getElementById('nav-toggle');
const sidebar = document.getElementById('sidebar');
const overlay = document.getElementById('sidebar-overlay');

function openNav() {{
  sidebar.classList.add('open');
  overlay.classList.add('open');
  toggle.innerHTML = '&#x2715;';
}}
function closeNav() {{
  sidebar.classList.remove('open');
  overlay.classList.remove('open');
  toggle.innerHTML = '&#9776;';
}}

toggle.addEventListener('click', () =>
  sidebar.classList.contains('open') ? closeNav() : openNav()
);
overlay.addEventListener('click', closeNav);
navLinks.forEach(l => l.addEventListener('click', () => {{
  if (window.innerWidth <= 768) closeNav();
}}));
</script>
</body>
</html>"""


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Income & Wealth Inequality Analysis")
    print("  USA · Canada · Norway and OECD Peers")
    print("=" * 60)

    # 1. Fetch & assemble data
    df = fetch_worldbank()
    df = build_master(df)

    print(f"\nData assembled for {len(df)} countries.")
    print(df[["gdp_per_capita", "gini", "wid_inc_top1", "wid_wlth_top1", "poverty"]].to_string())

    # 2. Build all charts
    print("\nBuilding charts…")
    chart_builders = {
        "gdp":         (chart_gdp,             "GDP per Capita"),
        "gini":        (chart_gini,            "Gini Coefficient"),
        "quintiles":   (chart_quintiles,       "Income Quintiles"),
        "wid_income":  (chart_wid_income,      "WID Income Shares"),
        "top1_dot":    (chart_top1_income_vs_wealth, "Top 1% Income vs Wealth"),
        "lorenz":      (chart_lorenz,          "Lorenz Curves"),
        "density":     (chart_density,         "Income Density"),
        "wealth_conc": (chart_wealth_conc,     "Wealth Concentration"),
        "wealth_skew": (chart_wealth_skew,     "Wealth Skewness"),
        "scatter":     (chart_scatter,         "Gini vs GDP"),
        "poverty":     (chart_poverty,         "Poverty Rates"),
        "radar":       (chart_radar,           "Radar Scorecard"),
    }
    charts = {}
    for key, (fn, label) in chart_builders.items():
        print(f"  • {label}")
        try:
            charts[key] = fn(df)
        except Exception as e:
            print(f"    [warn] {label} failed: {e}")
            charts[key] = f"<p style='color:#FF4D6D;padding:24px'>Chart unavailable: {e}</p>"

    # 3. Summary table & HTML
    print("\nAssembling HTML report…")
    table_html = build_summary_table(df)
    html = build_html(df, charts, table_html)

    output = "report.html"
    with open(output, "w", encoding="utf-8") as f:
        f.write(html)

    abs_path = os.path.abspath(output)
    print(f"\n✓  Report written → {output}")
    print(f"   Open: file://{abs_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
