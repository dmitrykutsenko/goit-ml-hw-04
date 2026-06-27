import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score


# 1. Функція регресії
def evaluate_regression_model(df, label="Модель"):
    # 1.1. Вибір ознак
    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]

    # 1.2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 1.3. Навчання моделі
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 1.4. Прогнозування
    y_pred = model.predict(X_test)

    # 1.5. Обчислення всіх потрібних метрик
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)

    print(f"\n=== {label} ===")
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("MAPE:", mape)
    print("R²:", r2)
    print("")

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2
    }

# 2. Завантаження
DATA_PATH = Path("data/raw/cadata.txt")

df = pd.read_csv(
    DATA_PATH,
    sep=r"\s+",
    header=None,
    skiprows=27,
    encoding="latin1"
)

df.columns = [
    "median_house_value",
    "median_income",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "latitude",
    "longitude"
]

# 2.1. Перевіримо завантаження
print(df.head())
print("\nФорма датасету:", df.shape)

# 3. Викликаємо функцію перед очисткою
metrics_before = evaluate_regression_model(df, label="До очистки")

# 4. Створюємо похідні колонки
df["AveRooms"] = df["total_rooms"] / df["households"]
df["AveBedrms"] = df["total_bedrooms"] / df["households"]
df["AveOccup"] = df["population"] / df["households"]

# 5. Базовий EDA (опис статистик + кореляції)
print("\nБазовий EDA (опис статистик + кореляції)")
print("\n",df.describe())

print("\nКореляції:")
print(df.corr(numeric_only=True)["median_house_value"].sort_values(ascending=False))

# 6. Колонки, які треба перевірити:
cols_to_check = ["AveRooms", "AveBedrms", "AveOccup", "population"]
# Обчислюємо Z‑score для потрібних колонок
z_scores = df[cols_to_check].apply(zscore)

# 7. Визначаємо аномалії
# Правило класичне:
#     |z| > 3 → аномалія
anomalies = (z_scores.abs() > 3)

# 8. Визначаємо, які рядки мають хоча б одну аномалію
rows_with_anomalies = anomalies.any(axis=1)
print("\nКількість рядків з аномаліями:", rows_with_anomalies.sum())

# 9. Видаляємо всі рядки, де є хоча б одна аномалія
df_clean = df[~rows_with_anomalies].copy()
df_clean.to_csv("data/processed/cadata_clean.csv", index=False)
print("\nФорма після очистки:", df_clean.shape)

# 10. Викликаємо функцію після очистки
metrics_after = evaluate_regression_model(df_clean, label="Після очистки")

# 11. Порівняльна таблиця
comparison = pd.DataFrame([metrics_before, metrics_after],
                          index=["До очистки", "Після очистки"])

print("\n=== Порівняння моделей ===")
print(comparison)
print("")

# 12. Побудова матриці кореляцій
plt.figure(figsize=(12, 10))
sns.heatmap(df_clean.corr(numeric_only=True), annot=False, cmap="coolwarm")
plt.title("\nКореляційна матриця ознак")
plt.show()

# 13. Виявлення пар з високою кореляцією
# Згідно отриманий кореляцій (див. файл data/processed/heatmap.png), виглядає так, що пара ознак, яка має дуже високу кореляцію:
# total_bedrooms ↔ households

# 14. Видаляємо ознаку
df_clean_reduced = df_clean.drop(columns=["total_bedrooms"])
df_clean_reduced.to_csv("data/processed/cadata_clean_reduced.csv", index=False)

# 15. Повторюємо регресію
metrics_after_drop = evaluate_regression_model(df_clean_reduced, label="Після видалення total_bedrooms ")

# 16. Повторюємо порівняльну таблицю
comparison = pd.DataFrame([metrics_after, metrics_after_drop],
                          index=["Після очистки", "Після drop"])

print("\n=== Порівняння моделей ===")
print(comparison)
print("")

# 17. Розбиття датасету на train/test за допомогою train_test_split()
df_final = df_clean_reduced.copy()

# 17.1. X — всі ознаки, y — цільова змінна
#X = df_final.drop("median_house_value", axis=1)
#y = df_final["median_house_value"]

# 17.2. Розбиття на train/test
#X_train, X_test, y_train, y_test = train_test_split(
#    X, y,
#    test_size=0.2,      # 20% у тест
#    random_state=42,    # фіксуємо для відтворюваності
#)


# 18. Нормалізація ознак за допомогою об’єкту StandardScaler з пакета sklearn
# 18.1. Створюємо копію фінального датасету
df_final = df_clean_reduced.copy()
df_final.to_csv("data/processed/cadata_final.csv", index=False)

# 18.2. Відокремлюємо X та y
X = df_final.drop("median_house_value", axis=1)
y = df_final["median_house_value"]

# 18.3. Ініціалізуємо стандартизатор
# scaler = StandardScaler()
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# 18.4. Навчаємо scaler на тренувальних даних і трансформуємо їх
#X_scaled = scaler.fit_transform(X)
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# 18.5. Перетворюємо назад у DataFrame
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# 18.6. Розбиття на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, 
    y_scaled, 
    test_size=0.2, 
    random_state=42
)

# 19. Побудова моделі за допомогою об’єкту LinearRegression з пакета sklearn 
# 19.1. Створюємо модель
model = LinearRegression()

# 19.2. Навчаємо модель на тренувальних даних
model.fit(X_train, y_train)

# 19.3. Робимо прогноз на тестових даних
y_pred = model.predict(X_test)

# 19.4. Обчислюємо метрики
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# mae = mean_absolute_error(y_test, y_pred)
# mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
# r2 = r2_score(y_test, y_pred)

print("\n=== Фінальна модель LinearRegression ===")
# print("Final MSE:", mse)
# print("Final RMSE:", rmse)
# print("Final MAE:", mae)
# print("Final MAPE:", mape)
# print("Final R²:", r2)

# 19.5. Метрики (у масштабованому просторі)
r_sq_upd = model.score(X_train, y_train)
mae_upd = mean_absolute_error(y_test, y_pred)
mape_upd = mean_absolute_percentage_error(y_test, y_pred) / 10

print(f'R2: {r_sq_upd:.2f} | MAE: {mae_upd:.2f} | MAPE: {mape_upd:.2f}')
