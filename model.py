import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def train_power_model(df):
    """Train regression model to predict power from voltage & current."""
    X = df[["voltage", "current"]]
    y = df["power"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)

    return model, r2, mse

if __name__ == "__main__":
    from analysis import generate_power_data
    df = generate_power_data()
    model, r2, mse = train_power_model(df)
    print(f"R²: {r2:.3f}, MSE: {mse:.3f}")
