import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="EV Battery Forecasting", layout="wide")
st.title("⚡ EV Battery Capacity Forecasting")
st.markdown("**NASA Battery Degradation Dataset | ML-powered predictions**")

@st.cache_data
def load_data():
    df = pd.read_csv("battery_cycle_level_dataset_CLEAN_FINAL.csv")
    cycle_counts = df.groupby("battery_id")["cycle"].max()
    valid = cycle_counts[cycle_counts >= 50].index
    df = df[df["battery_id"].isin(valid)].copy()
    Q1, Q3 = df["capacity"].quantile(0.25), df["capacity"].quantile(0.75)
    df = df[(df["capacity"] >= Q1 - 1.5*(Q3-Q1)) & 
            (df["capacity"] <= Q3 + 1.5*(Q3-Q1))]
    return df

df = load_data()


st.sidebar.header("⚙️ Controls")
battery   = st.sidebar.selectbox("Select Battery", df["battery_id"].unique())
model_choice = st.sidebar.selectbox("Select Model",
    ["Linear Regression", "Random Forest", "Gradient Boosting"])


col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Batteries", df["battery_id"].nunique())
col2.metric("Total Records",   len(df))
col3.metric("Max Cycles",      int(df["cycle"].max()))
col4.metric("Avg Capacity",    f"{df['capacity'].mean():.2f} Ah")

st.markdown("---")


st.subheader("🔋 Battery Capacity Degradation Over Cycles")
fig1, ax1 = plt.subplots(figsize=(12, 4))
for bid in df["battery_id"].unique():
    sub  = df[df["battery_id"] == bid]
    alpha = 1.0 if bid == battery else 0.25
    lw    = 2.5 if bid == battery else 0.8
    ax1.plot(sub["cycle"], sub["capacity"],
             label=bid if bid == battery else "",
             alpha=alpha, linewidth=lw)
ax1.set_xlabel("Cycle Number")
ax1.set_ylabel("Capacity (Ah)")
ax1.set_title(f"Highlighted: {battery}")
ax1.legend()
st.pyplot(fig1)

st.markdown("---")

def make_features(group):
    group = group.sort_values("cycle").copy()
    group["voltage_lag1"]  = group["voltage"].shift(1)
    group["voltage_roll3"] = group["voltage"].rolling(3).mean()
    group["temp_lag1"]     = group["temperature"].shift(1)
    group["capacity_diff"] = group["capacity"].diff()
    group["voltage_diff"]  = group["voltage"].diff()
    return group

df_feat = df.groupby("battery_id", group_keys=False).apply(make_features).dropna()

feature_cols = ["cycle", "voltage", "temperature",
                "voltage_lag1", "voltage_roll3",
                "temp_lag1", "capacity_diff", "voltage_diff"]

X        = df_feat[feature_cols]
y        = df_feat["capacity"]
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False)

models = {
    "Linear Regression"  : LinearRegression(),
    "Random Forest"      : RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting"  : GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    preds = m.predict(X_test)
    results[name] = {
        "preds" : preds,
        "mae"   : mean_absolute_error(y_test, preds),
        "rmse"  : np.sqrt(mean_squared_error(y_test, preds))
    }

st.subheader("Actual vs Predicted Capacity")
colors = {
    "Linear Regression" : "green",
    "Random Forest"     : "red",
    "Gradient Boosting" : "purple"
}
fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(y_test.values, label="Actual", color="blue", linewidth=2)
ax2.plot(results[model_choice]["preds"],
         label=f"{model_choice} (MAE={results[model_choice]['mae']:.4f})",
         color=colors[model_choice], linestyle="--", linewidth=2)
ax2.set_xlabel("Test Samples")
ax2.set_ylabel("Capacity (Ah)")
ax2.legend()
st.pyplot(fig2)

st.markdown("---")


st.subheader("Model Comparison")
comp = pd.DataFrame({
    "Model" : list(results.keys()),
    "MAE"   : [round(r["mae"],  4) for r in results.values()],
    "RMSE"  : [round(r["rmse"], 4) for r in results.values()]
}).set_index("Model")
st.dataframe(comp.style.highlight_min(color="lightgreen"))

# ── Feature importance ────────────────────────────────────────
st.subheader("🔍 Feature Importance (Random Forest)")
imp = pd.Series(
    models["Random Forest"].feature_importances_,
    index=feature_cols
).sort_values()
fig3, ax3 = plt.subplots(figsize=(10, 4))
imp.plot(kind="barh", color="steelblue", ax=ax3)
ax3.set_xlabel("Importance Score")
st.pyplot(fig3)

st.markdown("---")
st.caption("Built by Palak Adsul | NASA Battery Degradation Dataset | Scikit-learn")