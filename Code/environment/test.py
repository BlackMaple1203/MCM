import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
# from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

import data
kd = data.Kaya_data

# 已知数据
GDP = kd["GDP"]
MTCO2e = kd["MTCO2e"]

# 计算总体CI (排除无效数据)
CI_values = [mtco2e / gdp for gdp, mtco2e in zip(GDP, MTCO2e) if gdp != -1 and mtco2e != -1]

# 计算均值和标准差
mean_CI = np.mean(CI_values)
std_CI = np.std(CI_values, ddof=1)  # 使用样本标准差

# 计算95%的置信区间
n = len(CI_values)
alpha = 0.05  # 95%置信区间
t_score = stats.t.ppf(1 - alpha / 2, df=n - 1)  # t分布临界值
margin_of_error = t_score * (std_CI / np.sqrt(n))

# 置信区间
CI_lower = mean_CI - margin_of_error
CI_upper = mean_CI + margin_of_error

data = {
    "Year": [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    "GDP": [2524869000, 2435418000, 2437578000, 2511152000, 2467392000, 2414435000, 2292657000, 2421944000, 2423728000, 2387533000],
    "Tourism_Numbers": [9.61e5, 9.83e5, 1.015e6, 1.072e6, 1.151e6, 1.306e6, 1.259667e6, 1.213333e6, 1.167e6, 1.67e6],
    "GDP": [2524869000, 2435418000, 2437578000, 2511152000, 2467392000, 2414435000, 2292657000, 2421944000, 2423728000, 2387533000]
}

tourism_number = data["Tourism_Numbers"]
# 人均旅游以2%的增长率增长，初始为632
tourism_spend = [632 * (1 + 0.02) ** i for i in range(20)]
tmp = tourism_spend[0:10]
assert len(tourism_number) == len(tmp)
tourism_income = [i * j for i, j in zip(tourism_number, tmp)]
GDP = data["GDP"]

tourism_income_ratio = [i / j for i, j in zip(tourism_income, GDP) if j != -1]

print(tourism_income_ratio)


df = pd.DataFrame(data)
df.set_index('Year', inplace=True)
df["Pande_Impact"] = [0, 0, 0, 0, 0, 0, 0.2, 1, 0.8, 0]
df["Tourist_Ratio"] = tourism_income_ratio

# 扩展预测年份到2030年
forecast_years = list(range(2024, 2029))
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



def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c

def quad_func(x, a, b, c):
    return a * x ** 2 + b * x + c

CI_pred = [mean_CI * i for i in combined_df0['Tourist_Ratio']]

CI_pred = [float(ci) for ci in CI_pred]

X = np.array(kd["Tourism_Numbers"]).reshape(-1, 1)
y = np.array(CI_pred).reshape(-1, 1)

print(X.shape, y.shape)
assert X.shape == y.shape

model_CI_num = LinearRegression()
model_CI_num.fit(X, y)

# 方程
print("方程：")
print("CI = ", model_CI_num.coef_, "* n + ", model_CI_num.intercept_)

# 相关系数
print("相关系数：")
r = model_CI_num.score(X, y)
print(r)

# 预测并画图
y_pred = model_CI_num.predict(X)

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


tourism = np.array(kd["Tourism_Numbers"])

C_tourism = [i * j * k for i, j, k in zip(kd["Tourism_Numbers"], tourism_spend, CI_pred)]


# 2024是55/吨， 25年开始按5%递增
C_price = [33.77, 35.45, 37.23, 39.09, 41.04, 43.09, 45.25, 47.51, 49.89, 52.38, 55, 57.75, 60.64, 63.67, 66.85]

C_cost = np.array([i * j for i, j in zip(C_tourism, C_price)])

# tourism的数据统一除以10^6
tourism = np.array([i / 1e6 for i in tourism])

# C_cost的数据统一除以10^7
C_cost = np.array([i / 1e7 for i in C_cost])

# plt.figure(figsize=(10, 6))
# plt.scatter(tourism, C_cost, label='Carbon Cost')

# plt.xlabel('Number of Tourists')
# plt.ylabel('Carbon Cost')
# plt.title('Carbon Cost Forecast')
# plt.legend()
# plt.grid(True)
# plt.show()

# 拟合数据
popt, pcov = curve_fit(quad_func, tourism, C_cost)

# 输出拟合函数的参数
print(f'Fitted parameters: a={popt[0]}, b={popt[1]}, c={popt[2]}')

# 预测值
# C_cost_pred = exp_func(tourism, *popt)
C_cost_pred = quad_func(tourism, *popt)

# 绘制图像
plt.figure(figsize=(10, 6))
plt.scatter(tourism, C_cost, label='Carbon Cost')
# plt.plot(tourism, C_cost_pred, color='red', label='Exponential Fit')

# 绘制拟合的指数函数
x_fit = np.linspace(min(tourism), max(tourism), 100)
y_fit = quad_func(x_fit, *popt)
plt.plot(x_fit, y_fit, color='red', label='Fitted Quadratic Function')

plt.xlabel('Number of Tourists')
plt.ylabel('Carbon Cost')
plt.title('Carbon Cost Forecast')
plt.legend()
plt.grid(True)
plt.show()