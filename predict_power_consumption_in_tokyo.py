import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# 3. 特徴量と目的変数の分離
feature_cols = [
    "total_recipitation(mm)",
    "average_temperature(℃)",
    "use_aircon(bool)",
    "maximum_temperature(℃)",
    "minimum_temperature(℃)",
    "average_humidity(％)",
    "minimum_humidity(％)",
    "sunshine_hours(h)"
]

# 4. モデルの学習
model = joblib.load("models/power_predictor_model_v4.joblib")

# 6. テストデータ（東京の気象データ）の読み込み
weather_df = pd.read_csv("datas/predict_v4.csv")
for col in weather_df.columns:
    if col != "date":
        weather_df[col] = pd.to_numeric(weather_df[col], errors="coerce")

# 7. 必要な特徴量のみ抽出して予測
X_new = weather_df[feature_cols]
predictions = model.predict(X_new)

# 8. 結果をDataFrameにまとめて表示
result_df = weather_df.copy()
result_df["predicted_power_consumption(kWh)"] = predictions

result_df.to_csv("results/prediction_results_v4.csv", index=False)
