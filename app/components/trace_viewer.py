"""SpO2 trace visualization component using Plotly."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_gen.synthetic import NightTrace


def plot_trace(
    trace: NightTrace,
    show_accel: bool = False,
    title: str | None = None,
) -> go.Figure:
    """Plot an overnight SpO2 trace with clinical threshold lines.

    Args:
        trace: NightTrace to visualize
        show_accel: whether to show accelerometer on secondary y-axis
        title: custom title (auto-generated if None)
    """
    spo2 = trace.spo2
    n = len(spo2)
    hours = np.arange(n) / 3600.0  # convert seconds to hours

    if title is None:
        baby = trace.baby
        title = (
            f"Baby {baby.baby_id} | GA {baby.gestational_age_weeks}w "
            f"({baby.ga_category}) | Night {trace.night_number} | "
            f"Label: {trace.ground_truth_label}"
        )

    if show_accel:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
    else:
        fig = go.Figure()

    # Owlet palette (matched from owletcare.com)
    TEAL_DARK = "#2C5F5B"
    TEAL_PRIMARY = "#5BA69E"
    TEAL_LIGHT = "#6BACA4"
    URGENT_RED = "#C1565B"
    AMBER = "#D4A054"
    WARM_WHITE = "#FEFCFA"
    BORDER = "#E2DDD8"

    # Main SpO2 line
    fig.add_trace(go.Scatter(
        x=hours, y=spo2,
        mode="lines",
        name="SpO2",
        line=dict(color=TEAL_PRIMARY, width=1.2),
    ))

    # Shade urgent regions (SpO2 < 90%)
    urgent_mask = spo2 < 90
    if np.any(urgent_mask):
        fig.add_trace(go.Scatter(
            x=hours, y=np.where(urgent_mask, spo2, np.nan),
            mode="lines", name="< 90% (urgent)",
            line=dict(color=URGENT_RED, width=2),
            fill="tozeroy", fillcolor="rgba(193, 86, 91, 0.08)",
        ))

    # Shade borderline regions (90-94%)
    borderline_mask = (spo2 >= 90) & (spo2 <= 94)
    if np.any(borderline_mask):
        fig.add_trace(go.Scatter(
            x=hours, y=np.where(borderline_mask, spo2, np.nan),
            mode="lines", name="90-94% (borderline)",
            line=dict(color=AMBER, width=1.5),
        ))

    # Threshold lines
    fig.add_hline(y=90, line_dash="dash", line_color=URGENT_RED,
                  annotation_text="90% urgent", annotation_position="top left")
    fig.add_hline(y=94, line_dash="dash", line_color=AMBER,
                  annotation_text="94% borderline", annotation_position="top left")
    fig.add_hline(y=95, line_dash="dot", line_color=TEAL_LIGHT,
                  annotation_text="95% normal", annotation_position="top left")

    # Accelerometer on secondary y-axis
    if show_accel:
        fig.add_trace(go.Scatter(
            x=hours, y=trace.accel_magnitude,
            mode="lines", name="Accel (g)",
            line=dict(color=TEAL_DARK, width=0.5),
            opacity=0.3,
        ), secondary_y=True)
        fig.update_yaxes(title_text="Accelerometer (g)", secondary_y=True)

    fig.update_layout(
        title=dict(text=title, font=dict(color=TEAL_DARK, size=14,
                   family="Playfair Display, Georgia, serif")),
        xaxis_title="Hours into night",
        yaxis_title="SpO2 (%)",
        yaxis=dict(range=[60, 102], gridcolor=BORDER),
        xaxis=dict(gridcolor=BORDER),
        height=400,
        font=dict(family="DM Sans, system-ui, sans-serif", color=TEAL_DARK),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=WARM_WHITE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return fig
