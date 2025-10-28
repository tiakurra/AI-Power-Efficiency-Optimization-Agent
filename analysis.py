import pandas as pd
import numpy as np

def generate_power_data(n=200):
    """Simulate IoT sensor readings (voltage, current, power)."""
    time = np.arange(n)
    voltage = 120 + 2 * np.sin(time / 10) + np.random.normal(0, 0.5, n)
    current = 0.8 + 0.3 * np.sin(time / 15) + np.random.normal(0, 0.05, n)
    power = voltage * current
    df = pd.DataFrame({"timestamp": time, "voltage": voltage, "current": current, "power": power})
    return df

def compute_statistics(df):
    """Return basic summary statistics."""
    stats = df.describe()
    corr = df.corr(numeric_only=True)
    return stats, corr

if __name__ == "__main__":
    df = generate_power_data()
    stats, corr = compute_statistics(df)
    print(stats)
    print(corr)
