# EV Battery Capacity Forecasting

A machine learning system to forecast EV battery capacity degradation
over charge cycles using the NASA Battery Degradation Dataset.

---

Overview

Battery capacity degrades over charge cycles, affecting EV performance
and lifespan. This project builds and compares multiple ML models to
predict capacity using historical sensor data, with an interactive
Streamlit dashboard for visualization.

---

Dataset

NASA Battery Degradation Dataset (Cycle-Level CSV)
- 1,415 records | 46 batteries | 7 features
- Source: Kaggle (yashxss/nasa-battery-cycle-level-dataset)
- Features: battery_id, cycle, voltage, temperature, capacity, soh, rul

---

Methodology

Data Cleaning
- Removed batteries with fewer than 50 cycles
- Applied IQR-based outlier removal
- Final dataset: 1,170 records across 14 batteries

Feature Engineering
- Lag features: voltage_lag1, temp_lag1
- Rolling average: voltage_roll3
- Difference features: capacity_diff, voltage_diff

Models
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

---

Results

| Model               | MAE    | RMSE   |
|---------------------|--------|--------|
| Linear Regression   | 0.0679 | 0.0820 |
| Gradient Boosting   | 0.2451 | 0.2692 |
| Random Forest       | 0.2587 | 0.2918 |

Linear Regression achieved the best performance. Battery capacity
degradation follows a near-linear trend over cycles, making complex
ensemble models prone to overfitting on this dataset size.

---

Dashboard

Interactive Streamlit dashboard includes battery selector, model
selector, degradation chart, actual vs predicted plot, model
comparison table, and feature importance chart.

    pip install -r requirements.txt
    streamlit run app.py

---

Project Structure

    ev-battery-forecasting/
    ├── app.py
    ├── ev_battery_forecasting.ipynb
    ├── battery_cycle_level_dataset_CLEAN_FINAL.csv
    ├── requirements.txt
    └── README.md

---

Tech Stack

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit

---

Author

Palak Adsul
B.Tech Computer Engineering (AI/ML Specialization)
Vidyalankar Institute of Technology
GitHub: palakadsul | LinkedIn: palak-adsul
