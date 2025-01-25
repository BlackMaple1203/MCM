import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error

# 数据准备
data = {
    'Year': ['2019-2020', '2022-2023', '2023-2024'],
    'summer': [2213000, 2296600, 2648600],
    'winter': [323000, 357000, 398000],
    'total': [2536000, 2653600, 3046600]
}
df = pd.DataFrame(data)

# 将年份转换为时间序列索引
df['Year'] = pd.to_datetime(df['Year'].str[:4])  # 只取前4位作为年份
df.set_index('Year', inplace=True)

# 夏季数据
summer_data = df['summer']

# 冬季数据
winter_data = df['winter']

# 定义 SARIMA 模型参数
# SARIMA(p, d, q)(P, D, Q, s)
# 这里假设季节性周期 s=2（夏季和冬季）
order = (1, 1, 1)  # 非季节性部分 (p, d, q)
seasonal_order = (1, 1, 1, 2)  # 季节性部分 (P, D, Q, s)

# 训练夏季模型
summer_model = SARIMAX(summer_data, order=order, seasonal_order=seasonal_order)
summer_results = summer_model.fit(disp=False)

# 训练冬季模型
winter_model = SARIMAX(winter_data, order=order, seasonal_order=seasonal_order)
winter_results = winter_model.fit(disp=False)

# 预测未来 2 年
future_years = 2
forecast_index = pd.date_range(start=df.index[-1] + pd.DateOffset(years=1), periods=future_years, freq='YS')

# 夏季预测
summer_forecast = summer_results.get_forecast(steps=future_years)
summer_pred = summer_forecast.predicted_mean

# 冬季预测
winter_forecast = winter_results.get_forecast(steps=future_years)
winter_pred = winter_forecast.predicted_mean

# 打印预测结果
print("夏季预测结果:")
print(summer_pred)

print("\n冬季预测结果:")
print(winter_pred)

# 可视化结果
plt.figure(figsize=(12, 6))

# 夏季数据
plt.subplot(2, 1, 1)
plt.plot(summer_data.index, summer_data, label='实际夏季人数', marker='o')
plt.plot(forecast_index, summer_pred, label='预测夏季人数', marker='o', linestyle='--')
plt.title('夏季旅游人数预测')
plt.xlabel('年份')
plt.ylabel('人数')
plt.legend()

# 冬季数据
plt.subplot(2, 1, 2)
plt.plot(winter_data.index, winter_data, label='实际冬季人数', marker='o')
plt.plot(forecast_index, winter_pred, label='预测冬季人数', marker='o', linestyle='--')
plt.title('冬季旅游人数预测')
plt.xlabel('年份')
plt.ylabel('人数')
plt.legend()

plt.tight_layout()
plt.show()