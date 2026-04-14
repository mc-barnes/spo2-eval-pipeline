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

    # Main SpO2 line
    fig.add_trace(go.Scatter(
        x=hours, y=spo2,
        mode="lines",
        name="SpO2",
        line=dict(color="#2563eb", width=1),
    ))

    # Shade urgent regions (SpO2 < 90%)
    urgent_mask = spo2 < 90
    if np.any(urgent_mask):
        fig.add_trace(go.Scatter(
            x=hours, y=np.where(urgent_mask, spo2, np.nan),
            mode="lines", name="< 90% (urgent)",
            line=dict(color="#dc2626", width=2),
            fill="tozeroy", fillcolor="rgba(220, 38, 38, 0.1)",
        ))

    # Shade borderline regions (90-94%)
    borderline_mask = (spo2 >= 90) & (spo2 <= 94)
    if np.any(borderline_mask):
        fig.add_trace(go.Scatter(
            x=hours, y=np.where(borderline_mask, spo2, np.nan),
            mode="lines", name="90-94% (borderline)",
            line=dict(color="#f59e0b", width=1.5),
        ))

    # Threshold lines
    fig.add_hline(y=90, line_dash="dash", line_color="#dc2626",
                  annotation_text="90% urgent", annotation_position="top left")
    fig.add_hline(y=94, line_dash="dash", line_color="#f59e0b",
                  annotation_text="94% borderline", annotation_position="top left")
    fig.add_hline(y=95, line_dash="dot", line_color="#16a34a",
                  annotation_text="95% normal", annotation_position="top left")

    # Accelerometer on secondary y-axis
    if show_accel:
        fig.add_trace(go.Scatter(
            x=hours, y=trace.accel_magnitude,
            mode="lines", name="Accel (g)",
            line=dict(color="#9333ea", width=0.5),
            opacity=0.5,
        ), secondary_y=True)
        fig.update_yaxes(title_text="Accelerometer (g)", secondary_y=True)

    fig.update_layout(
        title=title,
        xaxis_title="Hours into night",
        yaxis_title="SpO2 (%)",
        yaxis=dict(range=[60, 102]),
        height=400,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return fig
