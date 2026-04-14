"""SpO2 Eval Pipeline — Design tokens.

Single source of truth for colors, typography, spacing, and shared Plotly layout.
Light/warm palette for clinical dashboard readability.
"""

# ---------------------------------------------------------------------------
# Colors — Clinical dashboard palette
# ---------------------------------------------------------------------------

# Primary / Brand teal family
TEAL_DARK = "#2C5F5B"       # Headings, dark text, sidebar labels, footer bg
TEAL_PRIMARY = "#5BA69E"    # Buttons, accents, chart primary, links
TEAL_LIGHT = "#6BACA4"      # Icon backgrounds, secondary labels, detail keys
SAGE = "#8CBDB7"            # Secondary chart bars, muted teal elements

# Backgrounds & Surfaces
CREAM_BG = "#F7F0EA"        # Page background (warm cream)
WARM_WHITE = "#FEFCFA"      # Cards, sidebar, plot backgrounds
SAGE_BG = "#E8F1EF"         # Table headers, subtle highlights, hover states
BORDER = "#E2DDD8"          # Card borders, dividers, chart grid lines

# Text
HEADING_TEXT = TEAL_DARK     # H1-H3, metric values
BODY_TEXT = "#3D4F5F"        # Body paragraphs, table cells (darker, blue-toned)
MUTED_TEXT = "#7A8B87"       # Captions, labels, secondary info

# Clinical status
URGENT_RED = "#C1565B"      # Urgent alerts, SpO2 < 90%
AMBER = "#D4A054"           # Borderline/monitor, 90-94% SpO2
NEUTRAL_GRAY = "#9CA3AF"    # Artifact, disabled states

# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------

FONT_HEADING = "'Playfair Display', Georgia, serif"
FONT_BODY = "'DM Sans', system-ui, sans-serif"
GOOGLE_FONTS_URL = "https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,500;0,600;0,700;1,500;1,600&family=DM+Sans:wght@400;500;600&display=swap"

# ---------------------------------------------------------------------------
# Spacing & Radii
# ---------------------------------------------------------------------------

RADIUS_BUTTON = "24px"      # Pill-shaped buttons
RADIUS_CARD = "14px"        # Metric cards, content panels
RADIUS_TABLE = "12px"       # Tables, small cards
RADIUS_BADGE = "20px"       # Urgency badges, pills
RADIUS_INPUT = "8px"        # Inputs, selects

# ---------------------------------------------------------------------------
# Chart colors — ordered sequences for Plotly
# ---------------------------------------------------------------------------

# Tier bars / funnel (Tier 1 → Tier 2 → Expert)
TIER_COLORS = [TEAL_PRIMARY, SAGE, AMBER]

# Full funnel including "All Traces"
FUNNEL_COLORS = [TEAL_DARK, TEAL_PRIMARY, SAGE, AMBER]

# Clinical label colors (for pie charts, GT distribution)
LABEL_COLORS = {
    "normal": TEAL_LIGHT,
    "borderline": AMBER,
    "urgent": URGENT_RED,
    "emergency": "#8B2020",
    "artifact": NEUTRAL_GRAY,
}

# Eval bar colors
EVAL_COLORS = [TEAL_PRIMARY, SAGE, AMBER]

# Urgency badge colors
URGENCY_COLORS = {
    "EMERGENCY": "#8B2020",
    "URGENT": URGENT_RED,
    "MONITOR": AMBER,
    "ROUTINE": TEAL_PRIMARY,
    "ARTIFACT REVIEW": NEUTRAL_GRAY,
}

# ---------------------------------------------------------------------------
# Shared Plotly layout
# ---------------------------------------------------------------------------

PLOTLY_LAYOUT = dict(
    font=dict(family=FONT_BODY, color=TEAL_DARK, size=13),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=WARM_WHITE,
    margin=dict(l=40, r=20, t=40, b=40),
)

# ---------------------------------------------------------------------------
# Global CSS — injected into Streamlit via st.markdown
# ---------------------------------------------------------------------------

GLOBAL_CSS = f"""
<style>
@import url('{GOOGLE_FONTS_URL}');

/* --- Global --- */
.stApp {{
    background-color: {CREAM_BG};
}}

/* --- Sidebar: warm cream --- */
section[data-testid="stSidebar"] {{
    background-color: {WARM_WHITE};
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] * {{
    color: {TEAL_DARK} !important;
}}
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label {{
    color: {TEAL_LIGHT} !important;
    font-family: {FONT_BODY} !important;
    font-size: 0.85rem !important;
}}
section[data-testid="stSidebar"] hr {{
    border-color: {BORDER} !important;
}}

/* --- Typography: serif headings --- */
h1 {{
    font-family: {FONT_HEADING} !important;
    color: {TEAL_DARK} !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
}}
h2, h3 {{
    font-family: {FONT_HEADING} !important;
    color: {TEAL_DARK} !important;
    font-weight: 500 !important;
}}
p, span, div, label, li {{
    font-family: {FONT_BODY};
    color: {BODY_TEXT};
}}
.stCaption, [data-testid="stCaptionContainer"] {{
    color: {MUTED_TEXT} !important;
    font-family: {FONT_BODY} !important;
}}

/* --- Metric cards: clean white, warm border --- */
[data-testid="stMetric"] {{
    background: {WARM_WHITE};
    border: 1px solid {BORDER};
    border-radius: {RADIUS_CARD};
    padding: 18px 22px;
    box-shadow: 0 1px 4px rgba(44, 95, 91, 0.04);
}}
[data-testid="stMetricLabel"] {{
    color: {MUTED_TEXT} !important;
    font-family: {FONT_BODY} !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
[data-testid="stMetricValue"] {{
    color: {TEAL_DARK} !important;
    font-family: {FONT_BODY} !important;
    font-weight: 600 !important;
}}

/* --- Dataframes --- */
[data-testid="stDataFrame"] {{
    border-radius: {RADIUS_TABLE};
    overflow: hidden;
    border: 1px solid {BORDER};
}}

/* --- Buttons: pill shape --- */
.stButton > button[kind="primary"] {{
    background-color: {TEAL_PRIMARY} !important;
    border-color: {TEAL_PRIMARY} !important;
    border-radius: {RADIUS_BUTTON} !important;
    font-family: {FONT_BODY} !important;
    font-weight: 500 !important;
    padding: 8px 28px !important;
}}
.stButton > button[kind="primary"]:hover {{
    background-color: {TEAL_DARK} !important;
    border-color: {TEAL_DARK} !important;
}}
.stButton > button {{
    border-radius: {RADIUS_BUTTON} !important;
    font-family: {FONT_BODY} !important;
}}

/* --- Alerts --- */
[data-testid="stAlert"] {{
    border-radius: 10px !important;
}}

/* --- Selectbox --- */
.stSelectbox > div > div {{
    border-radius: {RADIUS_INPUT} !important;
}}

/* --- Expander --- */
.streamlit-expanderHeader {{
    font-family: {FONT_BODY} !important;
    color: {TEAL_DARK} !important;
}}

/* --- Slider: teal accent --- */
.stSlider [data-baseweb="slider"] div[role="slider"] {{
    background-color: {TEAL_PRIMARY} !important;
    border-color: {TEAL_PRIMARY} !important;
}}
.stSlider [data-baseweb="slider"] [data-testid="stTickBarMin"],
.stSlider [data-baseweb="slider"] [data-testid="stTickBarMax"] {{
    color: {MUTED_TEXT} !important;
}}
div[data-baseweb="slider"] div[role="progressbar"] > div {{
    background-color: {TEAL_PRIMARY} !important;
}}
div[data-baseweb="slider"] div[data-testid="stThumbValue"] {{
    color: {TEAL_PRIMARY} !important;
}}

/* --- Radio buttons: teal accent --- */
.stRadio [data-testid="stMarkdownContainer"] {{
    font-family: {FONT_BODY} !important;
}}
div[data-baseweb="radio"] label span[data-testid] {{
    color: {TEAL_PRIMARY} !important;
}}
:root {{
    --primary-color: {TEAL_PRIMARY};
}}
.st-emotion-cache-1gulkj5 {{
    background-color: {TEAL_PRIMARY} !important;
}}
div[role="radiogroup"] label div[data-checked="true"] {{
    background-color: {TEAL_PRIMARY} !important;
    border-color: {TEAL_PRIMARY} !important;
}}
[data-testid="stWidgetLabel"] {{
    font-family: {FONT_BODY} !important;
}}

/* --- Bordered containers (st.container(border=True)) --- */
[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {WARM_WHITE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: {RADIUS_CARD} !important;
    box-shadow: 0 1px 6px rgba(44,95,91,0.05) !important;
    padding: 8px !important;
}}
[data-testid="stVerticalBlockBorderWrapper"] > div {{
    background: {WARM_WHITE} !important;
}}


/* --- Tabs: teal underline --- */
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
    color: {TEAL_DARK} !important;
    border-bottom-color: {TEAL_PRIMARY} !important;
}}
.stTabs [data-baseweb="tab-list"] button {{
    font-family: {FONT_BODY} !important;
    color: {MUTED_TEXT} !important;
}}
</style>
"""


# ---------------------------------------------------------------------------
# Reusable HTML component builders
# ---------------------------------------------------------------------------


def section_card(title: str, content: str, subtitle: str = "") -> str:
    """Wrap content in a styled card panel with optional title."""
    sub_html = (
        f'<div style="color:{MUTED_TEXT}; font-size:0.8rem; margin-top:2px; '
        f'font-family:{FONT_BODY};">{subtitle}</div>'
    ) if subtitle else ""
    title_html = (
        f'<div style="font-family:{FONT_HEADING}; color:{TEAL_DARK}; '
        f'font-weight:500; font-size:1.15rem; margin-bottom:4px;">{title}</div>'
        f'{sub_html}'
        f'<div style="border-bottom:1px solid {BORDER}; margin:12px 0;"></div>'
    ) if title else ""
    return (
        f'<div style="background:{WARM_WHITE}; border:1px solid {BORDER}; '
        f'border-radius:{RADIUS_CARD}; padding:24px; margin-bottom:16px; '
        f'box-shadow:0 1px 6px rgba(44,95,91,0.05);">'
        f'{title_html}{content}</div>'
    )


def metric_card_html(label: str, value: str, accent_color: str = TEAL_PRIMARY,
                     delta: str = "", delta_color: str = "") -> str:
    """Rich metric card with colored accent bar on top. Fixed height for alignment."""
    delta_html = (
        f'<div style="color:{delta_color or MUTED_TEXT}; font-size:0.8rem; margin-top:4px; '
        f'font-family:{FONT_BODY};">{delta}</div>'
    ) if delta else (
        f'<div style="font-size:0.8rem; margin-top:4px; visibility:hidden;">&nbsp;</div>'
    )
    return (
        f'<div style="background:{WARM_WHITE}; border:1px solid {BORDER}; '
        f'border-radius:{RADIUS_CARD}; padding:0; overflow:hidden; '
        f'box-shadow:0 1px 6px rgba(44,95,91,0.05); height:120px;">'
        f'<div style="height:4px; background:{accent_color};"></div>'
        f'<div style="padding:18px 22px;">'
        f'<div style="color:{MUTED_TEXT}; font-size:0.75rem; text-transform:uppercase; '
        f'letter-spacing:0.06em; font-family:{FONT_BODY}; margin-bottom:6px;">{label}</div>'
        f'<div style="color:{TEAL_DARK}; font-family:{FONT_BODY}; '
        f'font-weight:600; font-size:1.7rem; line-height:1.1;">{value}</div>'
        f'{delta_html}'
        f'</div></div>'
    )


def page_intro_html(text: str) -> str:
    """Styled intro callout for the top of a page."""
    return (
        f'<div style="background:{SAGE_BG}; border-left:3px solid {TEAL_PRIMARY}; '
        f'border-radius:0 {RADIUS_TABLE} {RADIUS_TABLE} 0; padding:14px 20px; '
        f'margin-bottom:24px; font-family:{FONT_BODY}; color:{BODY_TEXT}; '
        f'font-size:0.9rem; line-height:1.5;">{text}</div>'
    )


def accuracy_rows_html(tiers: list[tuple[str, float, str]]) -> str:
    """Flat accuracy bars matching the funnel aesthetic. Each tier is (name, pct, color)."""
    rows = ""
    for name, pct, color in tiers:
        bar_w = max(pct, 3)
        rows += (
            f'<div style="margin-bottom:14px;">'
            f'<div style="display:flex; justify-content:space-between; '
            f'align-items:baseline; margin-bottom:5px;">'
            f'<span style="color:{TEAL_DARK}; font-family:{FONT_BODY}; font-size:0.82rem; '
            f'font-weight:500;">{name}</span>'
            f'<span style="color:{TEAL_DARK}; font-family:{FONT_BODY}; font-weight:600; '
            f'font-size:0.9rem;">{pct:.1f}%</span></div>'
            f'<div style="background:{SAGE_BG}; border-radius:6px; height:10px; '
            f'overflow:hidden;">'
            f'<div style="background:{color}; width:{bar_w}%; height:100%; '
            f'border-radius:6px;"></div>'
            f'</div></div>'
        )
    return rows


def segmented_bar_html(segments: list[tuple[str, int, str]], total: int,
                       title: str = "", subtitle: str = "") -> str:
    """Segmented horizontal bar with labels underneath. Each segment is (name, count, color)."""
    # Build the bar segments
    bar_parts = ""
    for i, (name, count, color) in enumerate(segments):
        pct = count / total * 100 if total else 0
        # First segment gets left radius, last gets right radius
        radius = "0"
        if i == 0 and i == len(segments) - 1:
            radius = "8px"
        elif i == 0:
            radius = "8px 0 0 8px"
        elif i == len(segments) - 1:
            radius = "0 8px 8px 0"
        bar_parts += (
            f'<div style="width:{pct}%; height:100%; background:{color}; '
            f'border-radius:{radius}; display:flex; align-items:center; '
            f'justify-content:center; min-width:32px;">'
            f'<span style="color:white; font-family:{FONT_BODY}; font-weight:600; '
            f'font-size:0.8rem;">{count}</span></div>'
        )

    # Build the legend labels underneath
    legend_parts = ""
    for name, count, color in segments:
        pct = count / total * 100 if total else 0
        legend_parts += (
            f'<div style="display:flex; align-items:center;">'
            f'<div style="width:10px; height:10px; border-radius:3px; '
            f'background:{color}; margin-right:8px; flex-shrink:0;"></div>'
            f'<div>'
            f'<div style="font-family:{FONT_BODY}; font-size:0.82rem; '
            f'color:{TEAL_DARK}; font-weight:500;">{name}</div>'
            f'<div style="font-family:{FONT_BODY}; font-size:0.75rem; '
            f'color:{MUTED_TEXT};">{count} traces &middot; {pct:.0f}%</div>'
            f'</div></div>'
        )

    # Title block
    title_html = ""
    if title:
        sub = (f'<div style="color:{MUTED_TEXT}; font-size:0.8rem; margin-top:2px; '
               f'font-family:{FONT_BODY};">{subtitle}</div>') if subtitle else ""
        title_html = (
            f'<div style="font-family:{FONT_HEADING}; color:{TEAL_DARK}; '
            f'font-weight:500; font-size:1.15rem;">{title}</div>'
            f'{sub}'
            f'<div style="border-bottom:1px solid {BORDER}; margin:12px 0 16px;"></div>'
        )

    return (
        f'<div style="background:{WARM_WHITE}; border:1px solid {BORDER}; '
        f'border-radius:{RADIUS_CARD}; padding:24px; '
        f'box-shadow:0 1px 6px rgba(44,95,91,0.05);">'
        f'{title_html}'
        f'<div style="display:flex; height:40px; border-radius:8px; overflow:hidden; '
        f'margin-bottom:16px;">{bar_parts}</div>'
        f'<div style="display:flex; justify-content:space-evenly;">{legend_parts}</div>'
        f'</div>'
    )


def urgency_badge_html(urgency_level: str) -> str:
    """Return HTML for a colored urgency badge pill."""
    color = URGENCY_COLORS.get(urgency_level, TEAL_LIGHT)
    return (
        f'<div style="display:inline-block; background:{color}; '
        f'color:white; padding:6px 18px; border-radius:{RADIUS_BADGE}; '
        f'font-weight:600; font-size:0.85rem; letter-spacing:0.05em; '
        f'font-family:{FONT_BODY};">'
        f'{urgency_level}</div>'
    )


def detail_row_html(label: str, value: str) -> str:
    """Return HTML for a key-value detail row (patient details panel)."""
    return (
        f'<div style="display:flex; justify-content:space-between; '
        f'padding:6px 0; border-bottom:1px solid {BORDER};">'
        f'<span style="color:{TEAL_LIGHT}; font-size:0.85rem; '
        f'font-family:{FONT_BODY};">{label}</span>'
        f'<span style="color:{TEAL_DARK}; font-weight:500; font-size:0.85rem; '
        f'font-family:{FONT_BODY};">{value}</span>'
        f'</div>'
    )
