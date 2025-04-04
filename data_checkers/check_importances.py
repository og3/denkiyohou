# アップロードされたファイルを使って再実行
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import joblib

model = joblib.load("models/power_predictor_model_v3.joblib")

# 特徴量と目的変数の分離
feature_cols = [
    "average_temperature(℃)",
    "use_aircon(bool)"
]

# 重要度の抽出と可視化
importances = model.feature_importances_
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": importances
}).sort_values(by="importance", ascending=True)

# 棒グラフで可視化
plt.figure(figsize=(10, 6))
plt.barh(importance_df["feature"], importance_df["importance"])
plt.xlabel("Importance")
plt.title("特徴量の重要度（Random Forest）")
plt.tight_layout()
plt.show()

importance_df.sort_values(by="importance", ascending=False)
