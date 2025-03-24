import pandas as pd
import matplotlib.pyplot as plt

# データの読み込みと列名の整形
df = pd.read_csv('daily_weather_and_power_consumption.csv')
# 前ステップで整形した列名を使用
df.columns = [col.strip().replace('\n', '') for col in df.columns]

# 数値列だけを抽出して箱ひげ図で可視化
numeric_cols = df.select_dtypes(include='number').columns
df[numeric_cols].plot(kind='box', subplots=True, layout=(2, 4), figsize=(12, 6), sharex=False, sharey=False)
plt.tight_layout()
plt.show()
