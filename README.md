# 🫀 Life Expectancy Predictor

A Machine Learning web app that predicts **Life Expectancy** based on health and socioeconomic indicators using a Random Forest Regressor trained on the WHO Life Expectancy dataset.

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| R2 Score | 0.96 |
| MAE | 1.26 years |
| RMSE | 1.85 years |

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/your-username/life-expectancy-predictor.git
cd life-expectancy-predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
life-expectancy-predictor/
│
├── app.py                        # Streamlit web app
├── Life_Expectancy_Fixed.ipynb   # Jupyter notebook (data analysis + training)
├── life_expectancy_model.pkl     # Trained Random Forest model
├── life_expectancy_scaler.pkl    # StandardScaler for input normalization
├── Life Expectancy Data.csv      # Dataset (WHO)
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## 🔍 Input Features

| Feature | Description |
|---------|-------------|
| Adult Mortality | Deaths per 1000 population (age 15–60) |
| BMI | Average Body Mass Index |
| Schooling | Average years of schooling |
| GDP | GDP per capita (USD) |
| HIV/AIDS | Deaths per 1000 live births (0–4 years) |

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit
- Joblib

---

## 📌 Sample Predictions

| Adult Mortality | BMI | Schooling | GDP | HIV/AIDS | Predicted |
|----------------|-----|-----------|-----|----------|-----------|
| 89 | 57.7 | 18 | 36000 | 0.1 | ~78 years |
| 163 | 38.3 | 11 | 2500 | 1.2 | ~68 years |
| 320 | 22.0 | 5 | 400 | 12.0 | ~54 years |
| 450 | 21.0 | 4 | 300 | 28.0 | ~45 years |
