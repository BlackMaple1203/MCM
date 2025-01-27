import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

data = {
    "Year": [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    "GDP": [2524869000, 2435418000, 2437578000, 2511152000, 2467392000, 2414435000, 2292657000, 2421944000, 2423728000, 2387533000],
    "Tourism_Numbers": [9.61e5, 9.83e5, 1.015e6, 1.072e6, 1.151e6, 1.306e6, 1.259667e6, 1.213333e6, 1.167e6, 1.67e6],
    "GDP": [2524869000, 2435418000, 2437578000, 2511152000, 2467392000, 2414435000, 2292657000, 2421944000, 2423728000, 2387533000]
}

tourism_income = data["Tourism_Numbers"]
tourism_income = [i * 632 for i in tourism_income]
GDP = data["GDP"]

tourism_income_ratio = [i / j for i, j in zip(tourism_income, GDP) if j != -1]

print(tourism_income_ratio)


df = pd.DataFrame(data)
df.set_index('Year', inplace=True)
df["Pande_Impact"] = [0, 0, 0, 0, 0, 0, 0.2, 1, 0.8, 0]
df["Tourist_Ratio"] = tourism_income_ratio

# 扩展预测年份到2030年
forecast_years = list(range(2024, 2030))
forecast_exog = np.zeros(len(forecast_years))  # 假设未来几年疫情影响因子为0

model0 = SARIMAX(df["Tourist_Ratio"], exog=df['Pande_Impact'], order=(1, 2, 2))
fitted_model0 = model0.fit()

forecast0 = fitted_model0.forecast(steps=len(forecast_years), exog=forecast_exog)

forecast_df0 = pd.DataFrame({

    'Year': forecast_years,
    'Tourist_Ratio': forecast0
})

forecast_df0.set_index('Year', inplace=True)

combined_df0 = pd.concat([df, forecast_df0])

plt.plot(combined_df0.index, combined_df0['Tourist_Ratio'], label='Tourist Count')
plt.axvline(x=2023, color='r', linestyle='--', label='Forecast Start')
plt.xlabel('Year')
plt.ylabel('Tourist Ratio')
plt.title('Tourist Ratio Forecast')
plt.legend()
plt.grid(True)
plt.show()




model = SARIMAX(df['GDP'], exog=df['Pande_Impact'], order=(1, 2, 2))
fitted_model = model.fit()



forecast = fitted_model.forecast(steps=len(forecast_years), exog=forecast_exog)

forecast_df = pd.DataFrame({
    'Year': forecast_years,
    'GDP': forecast
})
forecast_df.set_index('Year', inplace=True)

# 合并已知数据和预测数据
combined_df = pd.concat([df, forecast_df])

print(combined_df)

combined_df.to_csv('combined_df.csv')

# 绘制图像
plt.figure(figsize=(10, 6))
plt.plot(combined_df.index, combined_df['GDP'], label='GDP')
plt.axvline(x=2023, color='r', linestyle='--', label='Forecast Start')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.title('GDP Forecast')
plt.legend()
plt.grid(True)
plt.show()