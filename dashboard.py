from analysis import generate_power_data, compute_statistics
from model import train_power_model
import pandas as pd

# optional OpenAI or Hugging Face integration
try:
    from openai import OpenAI
    client = OpenAI()
except:
    client = None

def generate_efficiency_insights(df, r2):
    """Use simple logic or an LLM to create insights."""
    avg_power = df["power"].mean()
    msg = f"Average power usage is {avg_power:.2f} W with model accuracy R² = {r2:.2f}. "
    if client:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Suggest 3 power efficiency improvements based on this data: {msg}"}],
        )
        return completion.choices[0].message.content
    else:
        return msg + "Consider reducing voltage fluctuations or improving current stability."

def run_agent():
    df = generate_power_data()
    stats, corr = compute_statistics(df)
    model, r2, mse = train_power_model(df)
    insights = generate_efficiency_insights(df, r2)
    return df, stats, corr, r2, mse, insights

if __name__ == "__main__":
    df, stats, corr, r2, mse, insights = run_agent()
    print(insights)
