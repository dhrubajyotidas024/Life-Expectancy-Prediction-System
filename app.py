import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Life Expectancy Predictor",
    page_icon="🫀",
    layout="centered"
)

# ── Load model & scaler ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load("life_expectancy_model.pkl")
    scaler = joblib.load("life_expectancy_scaler.pkl")
    return model, scaler

try:
    model, scaler = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🫀 Life Expectancy Predictor")
st.markdown("Enter the health and socioeconomic indicators below to predict life expectancy.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    adult_mortality = st.text_input("Adult Mortality (per 1000)", placeholder="e.g. 150")
    bmi             = st.text_input("BMI", placeholder="e.g. 38.5")
    schooling       = st.text_input("Schooling (years)", placeholder="e.g. 12")

with col2:
    gdp      = st.text_input("GDP per capita (USD)", placeholder="e.g. 5000")
    hiv_aids = st.text_input("HIV/AIDS (deaths per 1000)", placeholder="e.g. 0.5")

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Life Expectancy", use_container_width=True, type="primary"):
    if not model_loaded:
        st.error("❌ Model files not found. Make sure `life_expectancy_model.pkl` and `life_expectancy_scaler.pkl` are in the same folder as app.py.")
    elif not all([adult_mortality, bmi, schooling, gdp, hiv_aids]):
        st.warning("⚠️ Please fill in all the fields before predicting.")
    else:
        try:
            input_values = [
                float(adult_mortality),
                float(bmi),
                float(schooling),
                float(gdp),
                float(hiv_aids)
            ]
            input_data   = np.array([input_values])
            input_scaled = scaler.transform(input_data)
            prediction   = model.predict(input_scaled)[0]

            # ── Result ────────────────────────────────────────────────────────
            st.success(f"### Predicted Life Expectancy: **{prediction:.1f} years**")

            if prediction >= 75:
                st.info("✅ This indicates a **high** life expectancy — comparable to developed nations.")
            elif prediction >= 65:
                st.info("🟡 This indicates a **moderate** life expectancy.")
            else:
                st.warning("🔴 This indicates a **low** life expectancy — key risk factors may need attention.")

            # ── Feature Impact Graph ──────────────────────────────────────────
            st.divider()
            st.subheader("📊 Feature Impact on Prediction")
            st.markdown("How much each factor is **contributing** to the predicted life expectancy:")

            features = ['Adult Mortality', 'BMI', 'Schooling', 'GDP', 'HIV/AIDS']

            # Use feature importances from model weighted by scaled input values
            importances   = model.feature_importances_
            scaled_values = input_scaled[0]

            # Impact = importance * scaled value (positive = helps, negative = hurts)
            impacts = importances * scaled_values

            # Build dataframe
            impact_df = pd.DataFrame({
                'Feature': features,
                'Impact':  impacts
            }).sort_values('Impact', ascending=True)

            colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in impact_df['Impact']]

            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')

            bars = ax.barh(impact_df['Feature'], impact_df['Impact'], color=colors, edgecolor='none', height=0.5)

            # Labels on bars
            for bar, val in zip(bars, impact_df['Impact']):
                ax.text(
                    val + (0.001 if val >= 0 else -0.001),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}",
                    va='center',
                    ha='left' if val >= 0 else 'right',
                    color='white', fontsize=9
                )

            ax.axvline(0, color='white', linewidth=0.8, linestyle='--', alpha=0.5)
            ax.set_xlabel("Impact Score", color='white')
            ax.tick_params(colors='white')
            ax.spines[['top','right','bottom','left']].set_visible(False)
            for label in ax.get_yticklabels():
                label.set_color('white')

            plt.tight_layout()
            st.pyplot(fig)

            st.caption("🟢 Green = positive impact (increases life expectancy)   🔴 Red = negative impact (decreases life expectancy)")

        except ValueError:
            st.error("❌ Please enter valid numbers in all fields.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Model: Random Forest Regressor trained on WHO Life Expectancy dataset")
