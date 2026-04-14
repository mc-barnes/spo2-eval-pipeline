# Design System — SpO2 Eval Pipeline

Design tokens for the Streamlit dashboard. All values are defined in `app/theme.py`.

## Colors

### Primary / Brand
| Token | Hex | Usage |
|-------|-----|-------|
| `TEAL_DARK` | `#2C5F5B` | Headings, dark text, sidebar labels |
| `TEAL_PRIMARY` | `#5BA69E` | Buttons, accents, chart primary, links |
| `TEAL_LIGHT` | `#6BACA4` | Icon backgrounds, secondary labels |
| `SAGE` | `#8CBDB7` | Secondary chart bars, muted elements |

### Backgrounds & Surfaces
| Token | Hex | Usage |
|-------|-----|-------|
| `CREAM_BG` | `#F7F0EA` | Page background |
| `WARM_WHITE` | `#FEFCFA` | Cards, sidebar, plot backgrounds |
| `SAGE_BG` | `#E8F1EF` | Table headers, subtle highlights |
| `BORDER` | `#E2DDD8` | Card borders, dividers, grid lines |

### Text
| Token | Hex | Usage |
|-------|-----|-------|
| `HEADING_TEXT` | `#2C5F5B` | H1-H3, metric values |
| `BODY_TEXT` | `#3D4F5F` | Paragraphs, table cells |
| `MUTED_TEXT` | `#7A8B87` | Captions, labels, secondary info |

### Clinical Status
| Token | Hex | Usage |
|-------|-----|-------|
| `URGENT_RED` | `#C1565B` | SpO2 < 90%, urgent alerts |
| `AMBER` | `#D4A054` | 90-94% SpO2, borderline/monitor |
| `NEUTRAL_GRAY` | `#9CA3AF` | Artifact, disabled states |

## Typography

| Role | Font | Weight | Size |
|------|------|--------|------|
| H1 | Playfair Display | 600 | 2rem |
| H2 | Playfair Display | 500 | 1.4rem |
| H3 | Playfair Display | 500 | 1.15rem |
| Body | DM Sans | 400 | 0.95rem |
| Label | DM Sans | 400 | 0.78rem (uppercase) |
| Caption | DM Sans | 400 | 0.8rem |

Italic Playfair Display is used for emphasis in headings.

## Spacing

8px grid. Key values: `4px` (xs), `8px` (sm), `16px` (md), `24px` (lg), `32px` (xl), `40px` (section).

## Border Radius

| Usage | Value |
|-------|-------|
| Buttons | 24px (pill) |
| Metric cards | 14px |
| Tables, small cards | 12px |
| Badges, pills | 20px |
| Inputs, selects | 8px |

## Chart Color Sequences

**Tier bars/funnel**: `TEAL_PRIMARY` → `SAGE` → `AMBER`
**Clinical labels**: Normal (`TEAL_LIGHT`), Borderline (`AMBER`), Urgent (`URGENT_RED`), Artifact (`NEUTRAL_GRAY`)
**Eval bars**: `TEAL_PRIMARY` → `SAGE` → `AMBER`

## Source of Truth

All tokens live in `app/theme.py`. Dashboard (`app/dashboard.py`) and trace viewer (`app/components/trace_viewer.py`) import from there. Do not hardcode color values in page code.
