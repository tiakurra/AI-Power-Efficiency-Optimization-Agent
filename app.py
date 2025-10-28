import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os
from glob import glob
from datetime import datetime

st.set_page_config(page_title="AI Power Efficiency Optimization Agent", layout="wide")
st.title("⚡ AI Power Efficiency Optimization Agent")
st.write("Analyze device-level power usage and detect inefficiencies using statistical and regression analysis.")

# -------------------------
# Button to generate new random data
# -------------------------
if st.button("Generate New Power Data"):
    os.makedirs("data", exist_ok=True)
    np.random.seed()
    time = np.arange(0, 100)
    power = 50 + 10 * np.sin(time / 5) + np.random.normal(0, 2, len(time))
    df = pd.DataFrame({"Time": time, "Power": power})
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = f"data/sample_power_data_{timestamp}.csv"
    df.to_csv(file_path, index=False)
    st.success(f"Generated new CSV: {file_path}")

# -------------------------
# Load latest CSV from data folder
# -------------------------
csv_files = glob("data/sample_power_data_*.csv")
if not csv_files:
    st.warning("No CSV files found in the data folder. Please generate one first.")
    st.stop()

latest_csv = max(csv_files, key=os.path.getctime)
st.write(f"Loading data from: **{latest_csv}**")
data = pd.read_csv(latest_csv)

# -------------------------
# Train regression model
# -------------------------
X = data[["Time"]]
y = data["Power"]
model = LinearRegression()
model.fit(X, y)
data["Predicted Power"] = model.predict(X)

# -------------------------
# Visualization
# -------------------------
st.subheader("Power Usage Trend")
fig, ax = plt.subplots()
ax.plot(data["Time"], data["Power"], label="Actual Power", linewidth=2)
ax.plot(data["Time"], data["Predicted Power"], label="Predicted Power", linestyle="--")
ax.set_xlabel("Time")
ax.set_ylabel("Power (W)")
ax.legend()
st.pyplot(fig)

# -------------------------
# Statistics
# -------------------------
mean_power = data["Power"].mean()
max_power = data["Power"].max()
efficiency_score = 100 - ((max_power - mean_power) / mean_power * 100)

st.subheader("📊 Power Statistics")
st.write(f"**Average Power:** {mean_power:.2f} W")
st.write(f"**Max Power:** {max_power:.2f} W")
st.write(f"**Efficiency Score:** {efficiency_score:.2f}%")

# -------------------------
# Recommendations
# -------------------------
st.subheader("💡 Recommendations")
if efficiency_score < 85:
    st.warning("⚠️ Significant power spikes detected. Consider adjusting device scheduling or load balancing.")
else:
    st.success("✅ Power usage is efficient and stable across monitored intervals.")
