import pandas as pd

df = pd.read_csv("agriculture_yield_dataset.csv")

print("Rows and Columns:", df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nFirst 10 Records:")
print(df.head(10))

print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print(df.describe())

print("\nMean Values:")
print(df.mean(numeric_only=True))

print("\nStandard Deviations:")
print(df.std(numeric_only=True))

import matplotlib.pyplot as plt

df["rainfall_mm"].hist(bins=20)
plt.title("Rainfall Distribution")
plt.show()

df["temperature_c"].hist(bins=20)
plt.title("Temperature Distribution")
plt.show()

df["fertilizer_kg"].hist(bins=20)
plt.title("Fertilizer Distribution")
plt.show()

df["yield_ton_per_hectare"].hist(bins=20)
plt.title("Yield Distribution")
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

print(df["crop_type"].value_counts())

sns.countplot(x="crop_type", data=df)
plt.show()

print(df["soil_type"].value_counts())

sns.countplot(x="soil_type", data=df)
plt.show()

df["yield_ton_per_hectare"].hist(bins=20)
plt.title("Yield Distribution")
plt.show()

plt.scatter(df["rainfall_mm"], df["yield_ton_per_hectare"])
plt.xlabel("Rainfall")
plt.ylabel("Yield")
plt.show()

plt.scatter(df["fertilizer_kg"], df["yield_ton_per_hectare"])
plt.xlabel("Fertilizer")
plt.ylabel("Yield")
plt.show()

import seaborn as sns

corr_matrix = df.corr(numeric_only=True)

print(corr_matrix)

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True)
plt.show()

crop_avg = df.groupby("crop_type")["yield_ton_per_hectare"].mean()
print(crop_avg)

soil_avg = df.groupby("soil_type")["yield_ton_per_hectare"].mean()
print(soil_avg)

categorical_columns = ["crop_type", "soil_type"]

encoded_df = pd.get_dummies(df, columns=categorical_columns)

print(encoded_df.head())

X = encoded_df.drop("yield_ton_per_hectare", axis=1)
y = encoded_df["yield_ton_per_hectare"]

print("X Shape:", X.shape)
print("y Shape:", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X_train, y_train)

print("Intercept:")
print(model.intercept_)

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print(coef_df)

