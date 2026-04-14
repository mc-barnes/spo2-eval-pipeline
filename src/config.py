"""Central configuration for the SpO2 AI Eval Pipeline.

Single source of truth for clinical parameters, file paths, and pipeline constants.
Every module imports from here — no magic numbers elsewhere.
"""
from pathlib import Path
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# -- Paths --
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "output"
EXAMPLES_DIR = DATA_DIR / "examples"

# -- Gestational Age Categories (weeks) --
GA_EXTREMELY_PRETERM = (24, 28)
GA_VERY_PRETERM = (28, 32)
GA_MODERATE_PRETERM = (32, 37)
GA_TERM = (37, 42)

GA_CATEGORIES = {
    "extremely_preterm": GA_EXTREMELY_PRETERM,
    "very_preterm": GA_VERY_PRETERM,
    "moderate_preterm": GA_MODERATE_PRETERM,
    "term": GA_TERM,
}

# -- SpO2 Baselines by GA Category (mean, std) --
SPO2_BASELINES = {
    "extremely_preterm": (91.0, 2.0),
    "very_preterm": (93.0, 1.5),
    "moderate_preterm": (95.0, 1.0),
    "term": (98.0, 0.8),
}

# -- Birth Weight Ranges by GA Category (min_g, max_g) --
BIRTH_WEIGHT_RANGES = {
    "extremely_preterm": (500, 1000),
    "very_preterm": (1000, 1800),
    "moderate_preterm": (1800, 2500),
    "term": (2500, 4500),
}

# -- Clinical Thresholds (fixed, used as fallback when GA unknown) --
# NOTE: The rule engine uses GA-adjusted thresholds below when GA is available.
# These fixed values are kept as fallbacks and documentation of the baseline.
SPO2_URGENT_THRESHOLD = 90        # % — below this is urgent (term default)
SPO2_URGENT_DURATION_S = 10       # seconds — must sustain this long
SPO2_EMERGENCY_THRESHOLD = 80     # % — below this is emergency (call 911)
SPO2_BORDERLINE_LOW = 90
SPO2_BORDERLINE_HIGH = 94
SPO2_BORDERLINE_DURATION_S = 60   # seconds sustained for borderline
SPO2_NORMAL_THRESHOLD = 95        # above this all night = normal
SPO2_NORMAL_PCT = 0.95            # fraction of night that must be above threshold
ACCEL_ARTIFACT_THRESHOLD_G = 2.5  # g — accelerometer spike threshold
SPO2_ARTIFACT_RATE = 15.0         # %/3s — implausible rate of SpO2 change
SPO2_ARTIFACT_WINDOW_S = 3        # seconds for rate-of-change check

# -- GA-Adjusted Thresholds (from published neonatal reference ranges) --
# Sources: Castillo et al. 2008, Hay et al. 2002, BOOST II / COT trials
GA_URGENT_THRESHOLDS = {
    "extremely_preterm": 85,   # baseline ~91%, urgent well below baseline
    "very_preterm": 88,        # baseline ~93%
    "moderate_preterm": 89,    # baseline ~95%
    "term": 90,                # baseline ~98%, standard threshold
}

GA_BORDERLINE_RANGES = {
    "extremely_preterm": (85, 90),  # between urgent and expected baseline
    "very_preterm": (88, 92),
    "moderate_preterm": (89, 93),
    "term": (90, 94),
}

# -- Signal Parameters --
SAMPLING_RATE_HZ = 1
NIGHT_DURATION_HOURS = 8
NIGHT_DURATION_S = NIGHT_DURATION_HOURS * 3600  # 28,800

# -- Neonatal Physiology --
BREATHING_RATE_BPM_RANGE = (30, 60)  # neonatal respiratory rate
BREATHING_RATE_BPM_DEFAULT = 40

# -- Pattern Distribution for Synthetic Data --
# Weights for assigning pattern types (normal, urgent, borderline, artifact)
PATTERN_WEIGHTS = {
    "normal": 0.40,
    "urgent": 0.15,
    "borderline": 0.30,
    "artifact": 0.15,
}

# -- Pipeline Coverage Targets --
TIER1_COVERAGE_TARGET = 0.65
TIER2_COVERAGE_TARGET = 0.25
EXPERT_QUEUE_TARGET = 0.10

# -- Classifier Confidence Thresholds --
TIER2_CONFIDENCE_NORMAL = 0.80
TIER2_CONFIDENCE_URGENT = 0.70
TIER2_CONFIDENCE_DEFAULT = 0.75

# -- LLM Settings --
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-6"

# -- Dataset Defaults --
DEFAULT_N_BABIES = 100
DEFAULT_NIGHTS_PER_BABY = 3
DEFAULT_SEED = 42
