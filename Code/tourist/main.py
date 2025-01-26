# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from statsmodels.tsa.statespace.sarimax import SARIMAX

# # 朱诺市每年旅游人数数据
# data = {
#     'Year': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
#     'Tourist_Count': [961000, 983000, 1015000, 1072000, 1151000, 1306000, 0, 117000, 1167000, 1670000]
# }

# df = pd.DataFrame(data)
# df.set_index('Year', inplace=True)

# # 替换异常值，2020, 2021年的数据可能会被视为异常值，使用插值填充
# df['Tourist_Count'].loc[2020] = np.nan
# df['Tourist_Count'].loc[2021] = np.nan
# df['Tourist_Count'] = df['Tourist_Count'].interpolate()

# # 创建疫情影响因子（根据题意）
# df['Pandemic_Impact'] = [0, 0, 0, 0, 0, 0, 1, 1, 0.3, 0]

# # 使用SARIMAX模型进行建模
# model = SARIMAX(df['Tourist_Count'], 
#                 exog=df['Pandemic_Impact'],  # 引入外生变量
#                 order=(2, 1, 1))  # 假设p=2, d=1, q=1

# # 拟合模型
# fitted_model = model.fit()

# # 打印模型摘要
# print(fitted_model.summary())

# # 输出模型参数
# params = fitted_model.params
# print("Model Parameters:")
# print(params)

# # 进行预测 (假设未来几年疫情影响因子为0，表示无疫情)
# forecast_years = [2024, 2025, 2026, 2027, 2028]
# forecast_exog = np.array([0, 0, 0, 0, 0])  # 预测期的疫情影响因子
# forecast = fitted_model.forecast(steps=5, exog=forecast_exog)

# # 创建预测数据的DataFrame
# forecast_df = pd.DataFrame({'Year': forecast_years, 'Tourist_Count': forecast})
# forecast_df.set_index('Year', inplace=True)

# # 将已知数据和预测数据汇总到一张表格中
# combined_df = pd.concat([df[['Tourist_Count']], forecast_df], axis=0)

# # 打印汇总表格
# print(combined_df)

# # 可视化预测结果
# plt.plot(df.index, df['Tourist_Count'], label='Historical Data')
# plt.plot(forecast_years, forecast, label='Forecast with Pandemic Impact', color='red')
# plt.xlabel('Year')
# plt.ylabel('Tourist Count')
# plt.title('Tourist Count Forecast with Pandemic Impact (2024-2028)')
# plt.legend()
# plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 朱诺市每年旅游人数数据
data = {
    'Year': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'Tourist_Count': [961000, 983000, 1015000, 1072000, 1151000, 1306000, 0, 117000, 1167000, 1670000]
}

df = pd.DataFrame(data)
df.set_index('Year', inplace=True)

# 替换异常值，2020, 2021年的数据可能会被视为异常值，使用插值填充
df['Tourist_Count'].loc[2020] = np.nan
df['Tourist_Count'].loc[2021] = np.nan
df['Tourist_Count'] = df['Tourist_Count'].interpolate()

# 创建疫情影响因子（根据题意）
df['Pandemic_Impact'] = [0, 0, 0, 0, 0, 0, 1, 1, 0.3, 0]

# 使用SARIMAX模型进行建模
model = SARIMAX(df['Tourist_Count'], 
                exog=df['Pandemic_Impact'],  # 引入外生变量
                order=(2, 1, 1))  # 假设p=2, d=1, q=1

# 拟合模型
fitted_model = model.fit()

# 进行预测 (假设未来几年疫情影响因子为0，表示无疫情)
forecast_years = [2024, 2025, 2026, 2027, 2028]
forecast_exog = np.array([0, 0, 0, 0, 0])  # 预测期的疫情影响因子
forecast = fitted_model.forecast(steps=5, exog=forecast_exog)

# 获取残差
residuals = fitted_model.resid

# 获取模型的预测结果和置信区间
forecast_results = fitted_model.get_forecast(steps=5, exog=forecast_exog)
conf_int = forecast_results.conf_int()

# 创建图形并设置子图
fig, axes = plt.subplots(1, 4, figsize=(40, 5))

# 原始数据和预测结果
axes[0].plot(df.index, df['Tourist_Count'], label='Historical Data')
axes[0].plot(forecast_years, forecast, label='Forecast with Pandemic Impact', color='red')
axes[0].fill_between(forecast_years, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
axes[0].set_title('Tourist Count Forecast with Confidence Intervals')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Tourist Count')
axes[0].legend()

# 残差图
axes[1].plot(residuals)
axes[1].axhline(0, color='black', linestyle='--')
axes[1].set_title('Residuals of SARIMAX Model')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Residuals')

# ACF of residuals (limiting lags to the number of residuals)
max_lags = min(4, len(residuals)-1)  # Set max lags to 4 or length of residuals - 1
plot_acf(residuals, lags=max_lags, ax=axes[2])
axes[2].set_title('ACF of Residuals')

# PACF of residuals (limiting lags to the number of residuals)
plot_pacf(residuals, lags=max_lags, ax=axes[3])
axes[3].set_title('PACF of Residuals')

# 调整子图布局
# plt.tight_layout()
plt.show()


