# ── Standard library ─────────────────────────────────────────────────────────
from __future__ import annotations

# ── Third-party ───────────────────────────────────────────────────────────────
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# ═════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be the very first Streamlit call)
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Life Expectancy Predictor",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ═════════════════════════════════════════════════════════════════════════════
# GLOBAL CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════
FEATURE_NAMES: list[str] = [
    "Adult Mortality",
    "BMI",
    "Schooling",
    "GDP",
    "HIV/AIDS",
]

FEATURE_PLACEHOLDERS: dict[str, str] = {
    "Adult Mortality": "e.g. 150",
    "BMI": "e.g. 38.5",
    "Schooling": "e.g. 12",
    "GDP": "e.g. 5000",
    "HIV/AIDS": "e.g. 0.5",
}

FEATURE_UNITS: dict[str, str] = {
    "Adult Mortality": "per 1,000 adults",
    "BMI": "body-mass index",
    "Schooling": "years",
    "GDP": "USD per capita",
    "HIV/AIDS": "deaths per 1,000 live births",
}

MODEL_PATH: str = "life_expectancy_model.pkl"
SCALER_PATH: str = "life_expectancy_scaler.pkl"

# Colour palette (consistent throughout)
CLR_GREEN = "#2ecc71"
CLR_RED = "#e74c3c"
CLR_BG = "#0e1117"
CLR_CARD = "#1c1f26"
CLR_ACCENT = "#4f8ef7"
CLR_TEXT = "#e0e0e0"
CLR_MUTED = "#8b8fa8"

# ═════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS  — injected once at startup
# ═════════════════════════════════════════════════════════════════════════════

def inject_custom_css() -> None:
    """Inject global CSS overrides for a polished dark-theme look."""
    st.markdown(
        """
        <style>
        /* ── Base ── */
        html, body, [class*="css"] {
            font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
        }

        /* ── Metric card ── */
        .metric-card {
            background: linear-gradient(135deg, #1a2540 0%, #1c2d4a 100%);
            border: 1px solid #2d4070;
            border-radius: 16px;
            padding: 28px 32px;
            text-align: center;
            margin: 16px 0;
        }
        .metric-card .label {
            font-size: 0.85rem;
            font-weight: 600;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #8b9dc3;
            margin-bottom: 6px;
        }
        .metric-card .value {
            font-size: 3.4rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            line-height: 1.1;
        }
        .metric-card .unit {
            font-size: 1.1rem;
            color: #8b9dc3;
            font-weight: 400;
        }
        .metric-card .badge {
            display: inline-block;
            margin-top: 10px;
            padding: 4px 14px;
            border-radius: 99px;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.06em;
        }
        .badge-high   { background:#1a4d2e; color:#2ecc71; }
        .badge-mid    { background:#4d3d0f; color:#f1c40f; }
        .badge-low    { background:#4d1a1a; color:#e74c3c; }

        /* ── Section header ── */
        .section-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 28px 0 10px;
        }
        .section-header h3 {
            font-size: 1.05rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #c0cfe8;
            margin: 0;
        }
        .section-header .icon {
            font-size: 1.2rem;
        }

        /* ── Contribution table ── */
        .contrib-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88rem;
        }
        .contrib-table th {
            background: #1a2035;
            color: #8b9dc3;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-size: 0.75rem;
            padding: 10px 14px;
            text-align: left;
        }
        .contrib-table td {
            padding: 9px 14px;
            border-bottom: 1px solid #1e2535;
            color: #d0daf0;
        }
        .contrib-table tr:last-child td { border-bottom: none; }
        .contrib-table tr:hover td { background: #141824; }
        .pos-chip { color: #2ecc71; font-weight: 700; }
        .neg-chip { color: #e74c3c; font-weight: 700; }
        .neutral-chip { color: #8b9dc3; font-weight: 600; }

        /* ── Interpretation card ── */
        .interp-card {
            background: #111827;
            border-left: 3px solid #4f8ef7;
            border-radius: 0 10px 10px 0;
            padding: 16px 20px;
            margin: 14px 0;
            font-size: 0.9rem;
            color: #c8d6f0;
            line-height: 1.7;
        }

        /* ── Reliability card ── */
        .reliability-row {
            display: flex;
            gap: 12px;
            margin: 14px 0;
        }
        .rel-chip {
            flex: 1;
            background: #131826;
            border: 1px solid #222c44;
            border-radius: 10px;
            padding: 14px 16px;
            text-align: center;
        }
        .rel-chip .rl { font-size: 0.72rem; color: #6b7a9d; text-transform: uppercase; letter-spacing: 0.07em; }
        .rel-chip .rv { font-size: 1.3rem; font-weight: 700; color: #d0daf0; margin-top: 4px; }

        /* ── Input labels ── */
        label { font-weight: 500 !important; }

        /* ── Divider spacing ── */
        hr { margin: 18px 0 !important; }

        /* ── Warning / error boxes ── */
        .warn-box {
            background: #2d2208;
            border: 1px solid #f39c1240;
            border-radius: 10px;
            padding: 12px 16px;
            color: #f0c060;
            font-size: 0.88rem;
        }
        .err-box {
            background: #2d0808;
            border: 1px solid #e74c3c40;
            border-radius: 10px;
            padding: 12px 16px;
            color: #f07070;
            font-size: 0.88rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# MODEL & EXPLAINER LOADING  (cached across re-runs)
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading model …")
def load_artifacts() -> tuple:
    """
    Load the trained Random Forest model and fitted StandardScaler from disk.
    Returns (model, scaler) on success, raises FileNotFoundError on failure.
    """
    rf_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return rf_model, scaler


@st.cache_resource(show_spinner="Initialising SHAP explainer …")
def build_shap_explainer(_model) -> shap.TreeExplainer:
    """
    Build a SHAP TreeExplainer for the Random Forest.
    Using `feature_perturbation='interventional'` with `check_additivity=False`
    avoids the slow multi-output path and keeps explanations fast.
    """
    return shap.TreeExplainer(_model)


# ═════════════════════════════════════════════════════════════════════════════
# PREDICTION HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def parse_inputs(raw: dict[str, str]) -> np.ndarray | None:
    """
    Validate and parse raw string inputs into a (1, n_features) float array.
    Returns None and emits an error message if any input is invalid.
    """
    values: list[float] = []
    for name, raw_val in raw.items():
        raw_val = raw_val.strip()
        if not raw_val:
            st.markdown(
                f'<div class="warn-box">⚠️ <b>{name}</b> is empty. Please fill in all fields.</div>',
                unsafe_allow_html=True,
            )
            return None
        try:
            values.append(float(raw_val))
        except ValueError:
            st.markdown(
                f'<div class="err-box">❌ <b>{name}</b> — "<code>{raw_val}</code>" is not a valid number.</div>',
                unsafe_allow_html=True,
            )
            return None
    return np.array([values])  # shape (1, 5)


def categorise_prediction(years: float) -> tuple[str, str, str]:
    """
    Map a predicted life-expectancy value to a (label, badge_class, emoji).
    """
    if years >= 75:
        return "High", "badge-high", "🟢"
    elif years >= 65:
        return "Moderate", "badge-mid", "🟡"
    else:
        return "Low", "badge-low", "🔴"


# ═════════════════════════════════════════════════════════════════════════════
# SHAP  —  local explanation
# ═════════════════════════════════════════════════════════════════════════════

def compute_shap_values(
    explainer: shap.TreeExplainer,
    input_scaled: np.ndarray,
) -> np.ndarray:
    """
    Compute SHAP values for a single scaled input row.
    Returns a 1-D array of shape (n_features,).
    """
    shap_values = explainer.shap_values(input_scaled, check_additivity=False)

    # For Random Forest Regressor the output is (1, n_features)
    if isinstance(shap_values, list):
        # Multi-output or classifier wrapper — take first output
        shap_values = shap_values[0]

    return np.array(shap_values).flatten()


# ═════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ═════════════════════════════════════════════════════════════════════════════

def build_shap_bar_chart(shap_vals: np.ndarray) -> plt.Figure:
    """
    Build a professional horizontal bar chart of SHAP values.
    Green = positive contribution (raises life expectancy).
    Red   = negative contribution (lowers life expectancy).
    """
    df = pd.DataFrame(
        {"Feature": FEATURE_NAMES, "SHAP": shap_vals}
    ).sort_values("SHAP", ascending=True)

    bar_colours = [CLR_GREEN if v >= 0 else CLR_RED for v in df["SHAP"]]

    fig, ax = plt.subplots(figsize=(8, 3.8))
    fig.patch.set_facecolor(CLR_BG)
    ax.set_facecolor(CLR_BG)

    bars = ax.barh(
        df["Feature"],
        df["SHAP"],
        color=bar_colours,
        edgecolor="none",
        height=0.52,
    )

    # Value labels beside each bar
    x_max = df["SHAP"].abs().max()
    pad = x_max * 0.04

    for bar, val in zip(bars, df["SHAP"]):
        label = f"{val:+.2f}"
        x_pos = val + pad if val >= 0 else val - pad
        ha = "left" if val >= 0 else "right"
        ax.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha=ha,
            color="white",
            fontsize=9,
            fontweight="600",
        )

    # Zero baseline
    ax.axvline(0, color="#4a5068", linewidth=1.0, linestyle="--", alpha=0.8)

    # Axis styling
    ax.set_xlabel("SHAP Value  (years of life expectancy)", color=CLR_MUTED, fontsize=9)
    ax.tick_params(axis="x", colors=CLR_MUTED, labelsize=8.5)
    ax.tick_params(axis="y", colors=CLR_TEXT, labelsize=9.5)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Y-tick label colours matching bar colour
    for tick_label, val in zip(ax.get_yticklabels(), df["SHAP"]):
        tick_label.set_color(CLR_GREEN if val >= 0 else CLR_RED)
        tick_label.set_fontweight("600")

    plt.tight_layout(pad=0.6)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# CONTRIBUTION TABLE
# ═════════════════════════════════════════════════════════════════════════════

def render_contribution_table(shap_vals: np.ndarray) -> None:
    """
    Render a styled dataframe showing Feature | Contribution | Direction.
    Uses st.dataframe with a pandas Styler for colour-coded cells —
    avoids Streamlit's HTML sandboxing that blocks raw <table> injection.
    """
    sorted_idx = np.argsort(np.abs(shap_vals))[::-1]  # descending by magnitude

    rows: list[dict] = []
    for idx in sorted_idx:
        feat = FEATURE_NAMES[idx]
        val = float(shap_vals[idx])
        contrib_str = f"+{val:.2f}" if val > 0 else f"{val:.2f}"
        direction = "▲ Positive" if val > 0 else ("▼ Negative" if val < 0 else "── Neutral")
        rows.append({"Feature": feat, "Contribution (yrs)": contrib_str, "Direction": direction})

    df_table = pd.DataFrame(rows)

    def _style_row(row: pd.Series) -> list[str]:
        """Apply green/red background tint based on Direction column."""
        direction = row["Direction"]
        if "Positive" in direction:
            bg = "background-color: #0d2b1a; color: #2ecc71;"
        elif "Negative" in direction:
            bg = "background-color: #2b0d0d; color: #e74c3c;"
        else:
            bg = "color: #8b9dc3;"
        return [bg] * len(row)

    styled = (
        df_table.style
        .apply(_style_row, axis=1)
        .set_properties(**{
            "font-size": "0.88rem",
            "border": "1px solid #1e2535",
            "padding": "8px 14px",
        })
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", "#1a2035"),
                ("color", "#8b9dc3"),
                ("font-size", "0.75rem"),
                ("letter-spacing", "0.07em"),
                ("text-transform", "uppercase"),
                ("padding", "10px 14px"),
            ]},
            {"selector": "table", "props": [
                ("width", "100%"),
                ("border-collapse", "collapse"),
            ]},
        ])
        .hide(axis="index")
    )

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# AI INTERPRETATION
# ═════════════════════════════════════════════════════════════════════════════

def render_interpretation(shap_vals: np.ndarray) -> None:
    """
    Generate plain-English interpretation from SHAP values.
    Identifies the top positive and top negative contributor.
    """
    named = list(zip(FEATURE_NAMES, shap_vals))
    pos_sorted = sorted(
        [(n, v) for n, v in named if v > 0], key=lambda x: x[1], reverse=True
    )
    neg_sorted = sorted(
        [(n, v) for n, v in named if v < 0], key=lambda x: x[1]
    )

    lines: list[str] = []

    if pos_sorted:
        top_pos_name, top_pos_val = pos_sorted[0]
        lines.append(
            f"📈 <b>{top_pos_name}</b> contributed most toward "
            f"<b>increasing</b> predicted life expectancy "
            f"(+{top_pos_val:.2f} years)."
        )

    if neg_sorted:
        top_neg_name, top_neg_val = neg_sorted[0]
        lines.append(
            f"📉 <b>{top_neg_name}</b> contributed most toward "
            f"<b>decreasing</b> predicted life expectancy "
            f"({top_neg_val:.2f} years)."
        )

    if not pos_sorted and not neg_sorted:
        lines.append("All features had near-zero SHAP contributions for this input.")

    # Overall narrative
    total_positive = sum(v for _, v in pos_sorted)
    total_negative = sum(v for _, v in neg_sorted)
    if total_positive > abs(total_negative):
        lines.append(
            "🌐 Overall, <b>favourable indicators</b> outweighed risk factors "
            "in this prediction."
        )
    elif abs(total_negative) > total_positive:
        lines.append(
            "🌐 Overall, <b>risk factors</b> outweighed positive indicators "
            "in this prediction."
        )
    else:
        lines.append(
            "🌐 Positive and negative factors are roughly <b>balanced</b> "
            "in this prediction."
        )

    body = "<br>".join(lines)
    st.markdown(
        f'<div class="interp-card">{body}</div>',
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# MODEL RELIABILITY PANEL
# ═════════════════════════════════════════════════════════════════════════════

def _get_model_type_label(model) -> str:
    """Return a human-readable model family name."""
    class_name = type(model).__name__
    module = type(model).__module__

    if "xgboost" in module:
        return "XGBoost"
    if "lightgbm" in module:
        return "LightGBM"
    if "catboost" in module:
        return "CatBoost"
    if "RandomForest" in class_name:
        return "Random Forest"
    return class_name


def _tree_preds_from_sklearn(model, input_scaled: np.ndarray) -> np.ndarray | None:
    """
    Extract per-tree predictions from a scikit-learn ensemble (e.g. RandomForest).
    Returns None if the model doesn't expose `.estimators_`.
    """
    if not hasattr(model, "estimators_"):
        return None
    return np.array([tree.predict(input_scaled)[0] for tree in model.estimators_])


def _tree_preds_from_xgboost(model, input_scaled: np.ndarray) -> np.ndarray | None:
    """
    Extract per-tree predictions from an XGBoost model using staged prediction.
    Works for XGBRegressor.  Returns None if unavailable.
    """
    try:
        import xgboost as xgb  # local import — only runs if model is XGB

        dmatrix = xgb.DMatrix(input_scaled)
        # predict with ntree_limit=1..n_estimators gives cumulative predictions;
        # differences between consecutive steps give individual tree contributions.
        n_trees = model.n_estimators
        cumulative = np.array([
            model.get_booster().predict(dmatrix, iteration_range=(0, i + 1))[0]
            for i in range(n_trees)
        ])
        # Each "tree pred" = cumulative[i] - cumulative[i-1] + base_score/n_trees
        # For reliability purposes the cumulative spread is most meaningful.
        return cumulative
    except Exception:
        return None


def render_reliability(model, input_scaled: np.ndarray) -> None:
    """
    Display prediction reliability metrics in a model-agnostic way.

    Strategy:
      1. Try sklearn `.estimators_`  (Random Forest, ExtraTrees, …)
      2. Try XGBoost staged prediction
      3. Fall back to a SHAP-variance-based reliability note
    """
    model_label = _get_model_type_label(model)

    # ── Attempt to collect per-tree / per-stage predictions ──────────────────
    tree_preds = _tree_preds_from_sklearn(model, input_scaled)

    if tree_preds is None:
        tree_preds = _tree_preds_from_xgboost(model, input_scaled)

    if tree_preds is not None and len(tree_preds) > 1:
        # ── Full reliability panel ────────────────────────────────────────────
        mean_pred = tree_preds.mean()
        std_pred  = tree_preds.std()
        p5, p95   = np.percentile(tree_preds, [5, 95])
        agreement_pct = np.mean(np.abs(tree_preds - mean_pred) <= 2.0) * 100

        if std_pred <= 1.5:
            rel_label  = "🟢 High"
            rel_colour = CLR_GREEN
        elif std_pred <= 3.0:
            rel_label  = "🟡 Moderate"
            rel_colour = "#f1c40f"
        else:
            rel_label  = "🔴 Low"
            rel_colour = CLR_RED

        chips_html = f"""
        <div class="reliability-row">
            <div class="rel-chip">
                <div class="rl">Reliability</div>
                <div class="rv" style="color:{rel_colour};">{rel_label}</div>
            </div>
            <div class="rel-chip">
                <div class="rl">Std Dev</div>
                <div class="rv">± {std_pred:.2f} yrs</div>
            </div>
            <div class="rel-chip">
                <div class="rl">90 % Interval</div>
                <div class="rv">{p5:.1f} – {p95:.1f} yrs</div>
            </div>
            <div class="rel-chip">
                <div class="rl">Tree Agreement</div>
                <div class="rv">{agreement_pct:.0f} %</div>
            </div>
        </div>
        """
        st.markdown(chips_html, unsafe_allow_html=True)
        st.caption(
            f"Model: {model_label}  ·  "
            "Tree Agreement = % of individual estimators within ±2 years of the ensemble mean."
        )

    else:
        # ── Graceful fallback for unsupported model types ─────────────────────
        st.info(
            f"ℹ️ **{model_label}** does not expose individual tree predictions "
            "in a way that allows direct reliability estimation. "
            "SHAP values above still provide mathematically exact local explanations. "
            "For ensemble spread, consider wrapping the model in a scikit-learn "
            "`BaggingRegressor`."
        )


# ═════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def section_header(icon: str, title: str) -> None:
    """Render a styled section heading."""
    st.markdown(
        f"""
        <div class="section-header">
            <span class="icon">{icon}</span>
            <h3>{title}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_card(prediction: float) -> None:
    """Render the large metric card showing predicted life expectancy."""
    label, badge_cls, emoji = categorise_prediction(prediction)
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">Predicted Life Expectancy</div>
            <div class="value" style="color:#4f8ef7;">
                {prediction:.1f}
                <span class="unit">years</span>
            </div>
            <div>
                <span class="badge {badge_cls}">{emoji} {label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    inject_custom_css()

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center; padding: 10px 0 4px;">
            <span style="font-size:2.8rem;">🫀</span>
            <h1 style="font-size:2rem; font-weight:800; margin:6px 0 0; color:#e8eeff;">
                Life Expectancy Predictor
            </h1>
            <p style="color:#7a88a8; font-size:0.92rem; margin-top:6px;">
                Machine Learning · WHO Dataset · Random Forest + SHAP Explanations
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Load artifacts ────────────────────────────────────────────────────────
    try:
        model, scaler = load_artifacts()
        explainer = build_shap_explainer(model)
        model_ready = True
    except FileNotFoundError as exc:
        st.markdown(
            f'<div class="err-box">❌ <b>Model files not found.</b> '
            f"Ensure <code>life_expectancy_model.pkl</code> and "
            f"<code>life_expectancy_scaler.pkl</code> are in the same "
            f"directory as <code>app.py</code>.<br><small>{exc}</small></div>",
            unsafe_allow_html=True,
        )
        return

    # ── Input section ─────────────────────────────────────────────────────────
    section_header("📋", "Health & Socioeconomic Indicators")
    st.markdown(
        "<p style='color:#7a88a8; font-size:0.88rem; margin-bottom:18px;'>"
        "Enter values for the five predictive features below. "
        "All fields are required.</p>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="medium")
    raw_inputs: dict[str, str] = {}

    input_keys = list(FEATURE_PLACEHOLDERS.keys())
    left_features = input_keys[:3]   # Adult Mortality, BMI, Schooling
    right_features = input_keys[3:]  # GDP, HIV/AIDS

    with col1:
        for feat in left_features:
            raw_inputs[feat] = st.text_input(
                f"{feat}  ·  *{FEATURE_UNITS[feat]}*",
                placeholder=FEATURE_PLACEHOLDERS[feat],
                key=f"input_{feat}",
            )

    with col2:
        for feat in right_features:
            raw_inputs[feat] = st.text_input(
                f"{feat}  ·  *{FEATURE_UNITS[feat]}*",
                placeholder=FEATURE_PLACEHOLDERS[feat],
                key=f"input_{feat}",
            )

    st.divider()

    # ── Predict button ────────────────────────────────────────────────────────
    predict_clicked = st.button(
        "🔍  Predict Life Expectancy",
        use_container_width=True,
        type="primary",
    )

    if not predict_clicked:
        st.markdown(
            "<p style='color:#5a6680; font-size:0.82rem; text-align:center; "
            "margin-top:8px;'>"
            "Fill in the indicators above and click Predict.</p>",
            unsafe_allow_html=True,
        )
        return

    if not model_ready:
        return

    # ── Validate & parse inputs ───────────────────────────────────────────────
    input_array = parse_inputs(raw_inputs)
    if input_array is None:
        return

    # ── Scale & predict ───────────────────────────────────────────────────────
    try:
        input_scaled = scaler.transform(input_array)        # (1, 5)
        prediction = model.predict(input_scaled)[0]         # scalar float
    except Exception as exc:
        st.markdown(
            f'<div class="err-box">❌ Prediction failed: <code>{exc}</code></div>',
            unsafe_allow_html=True,
        )
        return

    # ── Result card ───────────────────────────────────────────────────────────
    st.divider()
    section_header("🎯", "Prediction Result")
    render_prediction_card(prediction)

    # ── SHAP computation ──────────────────────────────────────────────────────
    with st.spinner("Computing SHAP explanations …"):
        try:
            shap_vals = compute_shap_values(explainer, input_scaled)
        except Exception as exc:
            st.markdown(
                f'<div class="err-box">⚠️ SHAP computation failed: '
                f"<code>{exc}</code></div>",
                unsafe_allow_html=True,
            )
            return

    # ── SHAP bar chart ────────────────────────────────────────────────────────
    st.divider()
    section_header("📊", "Feature Contribution  (SHAP Values)")
    st.markdown(
        "<p style='color:#7a88a8; font-size:0.86rem; margin-bottom:14px;'>"
        "SHAP (SHapley Additive exPlanations) quantifies the exact contribution "
        "of each feature to <em>this specific prediction</em>. "
        "Values are in <b>years of life expectancy</b>.</p>",
        unsafe_allow_html=True,
    )

    fig = build_shap_bar_chart(shap_vals)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.caption(
        "🟢 Green bar = positive SHAP value (raises predicted life expectancy)   "
        "🔴 Red bar = negative SHAP value (lowers predicted life expectancy)"
    )

    # ── AI Interpretation ─────────────────────────────────────────────────────
    section_header("🤖", "AI Interpretation")
    render_interpretation(shap_vals)

    # ── Contribution table ────────────────────────────────────────────────────
    st.divider()
    section_header("📋", "Feature Contribution Table")
    render_contribution_table(shap_vals)

    # ── Model reliability ─────────────────────────────────────────────────────
    st.divider()
    section_header("📐", "Prediction Reliability")
    st.markdown(
        "<p style='color:#7a88a8; font-size:0.86rem; margin-bottom:4px;'>"
        "Reliability is estimated from the spread of predictions across all "
        "individual decision trees inside the ensemble.</p>",
        unsafe_allow_html=True,
    )
    render_reliability(model, input_scaled)

  


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()