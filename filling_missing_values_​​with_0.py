import pandas as pd

# データの読み込みと列名の整形
df = pd.read_csv('daily_weather_and_power_consumption.csv')
# カラムに改行コードがあれば削除する
df.columns = [col.strip().replace('\n', '') for col in df.columns]

# 欠損値を0で穴埋め
df_filled = df.fillna(0)

# nullの数を列ごとに確認
print(df_filled.isnull().sum())  # すべて0であればOK