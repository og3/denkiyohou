from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import joblib

# CSVの読み込み
df = pd.read_csv("datas/daily_weather_and_power_consumption.csv")

# 数値に変換（エラーはNaNとして処理）
for col in df.columns:
    if col != "date":
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 欠損値を0で穴埋め
df_filled = df.fillna(0)

# 特徴量と目的変数に分割
X = df_filled.drop(columns=["date", "power_consumption(kWh)"])
y = df_filled["power_consumption(kWh)"]

# 学習データとテストデータに分割（8:2）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ランダムフォレストによる学習と予測
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# モデル評価
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R²:", r2)

# モデルの保存
joblib.dump(model, "models/power_predictor_model.joblib")
print("modelを保存しました")
