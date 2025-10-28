import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Power Efficiency Optimization Agent", layout="wide")

# ---- Title ----
st.title("⚡ AI Power Efficiency Optimization Agent")
st.write("Analyze device-level power usage and detect inefficiencies using machine learning and statistical analysis.")

# ---- Simulate IoT power data ----
np.random.seed(42)
time = np.arange(0, 100)
power_usage = 50 + 10 * np.sin(time / 5) + np.random.normal(0, 2, len(time))

data = pd.DataFrame({
    "Time (s)": time,
    "Power (W)": power_usage
})

# ---- Train regression model ----
X = data[["Time (s)"]]
y = data["Power (W)"]
model = LinearRegression()
model.fit(X, y)

# Predict power
data["Predicted Power (W)"] = model.predict(X)

# ---- Visualization ----
st.subheader("Power Usage Trend")
fig, ax = plt.subplots()
ax.plot(data["Time (s)"], data["Power (W)"], label="Actual Power", linewidth=2)
ax.plot(data["Time (s)"], data["Predicted Power (W)"], label="Predicted Power", linestyle="--")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Power (W)")
ax.legend()
st.pyplot(fig)

# ---- Statistical insights ----
mean_power = np.mean(power_usage)
max_power = np.max(power_usage)
min_power = np.min(power_usage)
efficiency_score = 100 - ((max_power - mean_power) / mean_power * 100)

st.subheader("📊 Power Statistics")
st.write(f"**Average Power:** {mean_power:.2f} W")
st.write(f"**Max Power:** {max_power:.2f} W")
st.write(f"**Efficiency Score:** {efficiency_score:.2f}%")

# ---- Recommendations ----
st.subheader("💡 Recommendations")
if efficiency_score < 85:
    st.warning("⚠️ Significant power spikes detected. Consider adjusting device scheduling or load balancing.")
else:
    st.success("✅ Power usage is efficient and stable across monitored intervals.")

