import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルの再読み込み
df = pd.read_csv('daily_weather_and_power_consumption.csv')

# 列名の整形
df.columns = [col.strip().replace('\n', '') for col in df.columns]

# "年月日"以外を数値に変換（変換できないものはNaNに）
for col in df.columns:
    if col != "date":
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 数値列の抽出
numeric_cols = df.select_dtypes(include='number').columns

# 箱ひげ図の描画
df[numeric_cols].plot(kind='box', subplots=True, layout=(2, 4), figsize=(14, 6), sharex=False, sharey=False)
plt.tight_layout()
plt.show()
