import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 提取人口数据
data = {
    'Year': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'Population': [31.4, 32.2, 32.4, 32.6, 32.5, 32.6, 32.5, 32.1, 32.0, 32.0, 32.2, 32.0, 31.7, 31.6, 31.3]
}

official_data = {
    'Year': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029],
    'Population': [31.4, 32.2, 32.4, 32.6, 32.5, 32.6, 32.5, 32.1, 32.0, 32.0, 32.2, 32.0, 31.7, 31.6, 31.3, 31.1, 30.9, 30.7, 30.5, 30.2]
}

df = pd.DataFrame(data)
df.set_index('Year', inplace=True)

# 使用SARIMAX模型进行建模
model = SARIMAX(df['Population'], order=(2, 1, 1), seasonal_order=(1, 1, 1, 12))

# 拟合模型
fitted_model = model.fit()

# 打印模型摘要
print(fitted_model.summary())

# 打印已知数据
print("Historical Data:")
print(df)

# 进行预测 (假设未来5年)
forecast_years = [2025, 2026, 2027, 2028, 2029]
forecast = fitted_model.get_forecast(steps=5)
forecast_index = pd.Index(forecast_years, name='Year')
forecast_df = forecast.summary_frame(alpha=0.05).set_index(forecast_index)

# 打印预测结果
print("Forecast Data:")
print(forecast_df[['mean', 'mean_ci_lower', 'mean_ci_upper']])

# 获取残差
residuals = fitted_model.resid

# 合并官方数据和预测数据
official_df = pd.DataFrame(official_data).set_index('Year')
combined_df = pd.concat([df, forecast_df[['mean']]], axis=0)

# 创建图形并设置子图
fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # 创建2行3列的子图布局

# 原始数据和预测结果
axes[0, 0].plot(df.index, df['Population'], label='Historical Data')
axes[0, 0].plot(forecast_df.index, forecast_df['mean'], label='Forecast', color='red')
axes[0, 0].fill_between(forecast_df.index, forecast_df['mean_ci_lower'], forecast_df['mean_ci_upper'], color='pink', alpha=0.3)
axes[0, 0].set_title('Population Forecast')
axes[0, 0].set_xlabel('Year')
axes[0, 0].set_ylabel('Population (in thousands)')
axes[0, 0].legend()

# 残差图
axes[0, 1].plot(residuals)
axes[0, 1].axhline(0, color='black', linestyle='--')
axes[0, 1].set_title('Residuals')
axes[0, 1].set_xlabel('Year')
axes[0, 1].set_ylabel('Residuals')

# ACF of residuals
plot_acf(residuals, lags=6, ax=axes[0, 2])
axes[0, 2].set_title('ACF of Residuals')

# PACF of residuals
plot_pacf(residuals, lags=6, ax=axes[1, 0])
axes[1, 0].set_title('PACF of Residuals')

# 原始数据与预测数据和官方数据对比
axes[1, 1].plot(combined_df.index, combined_df['mean'], label='Forecast', color='red')
axes[1, 1].plot(official_df.index, official_df['Population'], label='Official Data', color='green')
axes[1, 1].set_title('Forecast vs Official Data')
axes[1, 1].set_xlabel('Year')
axes[1, 1].set_ylabel('Population (in thousands)')
axes[1, 1].legend()

# 展示预测差异图
diff = official_df['Population'] - combined_df['mean'].reindex(official_df.index)
axes[1, 2].plot(diff, label='Forecast Difference', color='purple')
axes[1, 2].set_title('Difference Between Forecast and Official Data')
axes[1, 2].set_xlabel('Year')
axes[1, 2].set_ylabel('Population Difference (in thousands)')
axes[1, 2].axhline(0, color='black', linestyle='--')
axes[1, 2].legend()

# 调整子图布局
plt.tight_layout()
plt.show()
