import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# Генерація випадкових даних
m = 100
X = 6 * np.random.rand(m, 1) - 5  # X значення в діапазоні [-5, 1]
y = 0.7 * X ** 2 + X + 3 + np.random.randn(m, 1)  # Квадратична залежність з шумом
# Побудова графіка даних
plt.scatter(X, y, color='green', label='Данні')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Випадкові дані')
plt.legend()
plt.show()
# Лінійна регресія
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)
y_pred_linear = linear_regressor.predict(X)
# Побудова графіка лінійної регресії
plt.scatter(X, y, color='green', label='Данні')
plt.plot(X, y_pred_linear, color='blue', linewidth=3, label='Лінійна регресія')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Лінійна регресія')
plt.legend()
plt.show()
# Поліноміальна регресія
poly_features = PolynomialFeatures(degree=2)  # Ступінь полінома 2
X_poly = poly_features.fit_transform(X)
poly_regressor = LinearRegression()
poly_regressor.fit(X_poly, y)
y_pred_poly = poly_regressor.predict(X_poly)
# Побудова графіка поліноміальної регресії
plt.scatter(X, y, color='green', label='Данні')
plt.plot(np.sort(X, axis=0), poly_regressor.predict(poly_features.transform(np.sort(X, axis=0))), color='red', linewidth=3, label='Поліноміальна регресія')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Поліноміальна регресія (ступінь 2)')
plt.legend()
plt.show()
# Оцінка якості моделей
# Лінійна регресія
mse_linear = mean_squared_error(y, y_pred_linear)
r2_linear = r2_score(y, y_pred_linear)
mae_linear = mean_absolute_error(y, y_pred_linear)
print("Лінійна регресія:")
print("Mean Squared Error (MSE):", round(mse_linear, 2))
print("R2 score:", round(r2_linear, 2))
print("Mean Absolute Error (MAE):", round(mae_linear, 2))
# Отримання коефіцієнтів і перехоплення
print("Коефіцієнти лінійної регресії:", linear_regressor.coef_)
print("Перехоплення лінійної регресії:", linear_regressor.intercept_)
# Поліноміальна регресія
mse_poly = mean_squared_error(y, y_pred_poly)
r2_poly = r2_score(y, y_pred_poly)
mae_poly = mean_absolute_error(y, y_pred_poly)
print("\nПоліноміальна регресія:")
print("Mean Squared Error (MSE):", round(mse_poly, 2))
print("R2 score:", round(r2_poly, 2))
print("Mean Absolute Error (MAE):", round(mae_poly, 2))
# Отримання коефіцієнтів і перехоплення
print("Коефіцієнти поліноміальної регресії:", poly_regressor.coef_)
print("Перехоплення поліноміальної регресії:", poly_regressor.intercept_)