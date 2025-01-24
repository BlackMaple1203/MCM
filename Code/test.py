import numpy as np
from scipy.optimize import minimize

# 定义目标函数
def objective_function(vars):
    x, y, z = vars
    return (x - 1)**2 + (y - 2)**2 + (z - 3)*(y - 3)

# 初始猜测值（猜测从哪开始优化）
initial_guess = [10, 10, 10]

# 使用scipy的minimize函数来最小化目标函数
result = minimize(objective_function, initial_guess)

# 输出结果
if result.success:
    fitted_params = result.x
    print(fitted_params)
else:
    raise ValueError(result.message)