import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pickle

# Sample dataset
data = pd.DataFrame({
    "area": [1000, 1500, 2000, 2500, 3000],
    "bedrooms": [2, 3, 3, 4, 4],
    "bathrooms": [1, 2, 2, 3, 3],
    "price": [50, 75, 100, 130, 160]
})

X = data.drop("price", axis=1)
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = XGBRegressor()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved!")
