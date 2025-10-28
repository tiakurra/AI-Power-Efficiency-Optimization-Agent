import numpy as np
import pandas as pd
import os
from datetime import datetime

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Generate random power data
np.random.seed()
time = np.arange(0, 100)
power = 50 + 10 * np.sin(time / 5) + np.random.normal(0, 2, len(time))

df = pd.DataFrame({"Time": time, "Power": power})

# Save with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
file_path = f"data/sample_power_data_{timestamp}.csv"
df.to_csv(file_path, index=False)

print(f"Generated new CSV: {file_path}")
