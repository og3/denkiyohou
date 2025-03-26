import pandas as pd
import matplotlib.pyplot as plt

# データ読み込み
df = pd.read_csv("datas/daily_weather_and_power_consumption.csv")

# 数値変換（"date"以外）
for col in df.columns:
    if col != "date":
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 欠損値補完
df_filled = df.fillna(0)

# 相関係数を算出（目的変数との関係）
correlations = df_filled.corr(numeric_only=True)["power_consumption(kWh)"].drop("power_consumption(kWh)")

# 可視化
plt.figure(figsize=(10, 5))
correlations.sort_values().plot(kind="barh")
plt.title("各特徴量と電力消費量の相関係数")
plt.xlabel("相関係数")
plt.tight_layout()
plt.show()

correlations