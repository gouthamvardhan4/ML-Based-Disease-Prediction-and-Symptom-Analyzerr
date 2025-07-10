import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

data_path = r"C:\Users\sudeep\Desktop\m2\adiposity.csv"
data = pd.read_csv(data_path)

label_encoder = LabelEncoder()

categorical_columns = ['Gender', 'family_history_with_overweight', 'FAVC', 'FCVC', 'CAEC', 'SMOKE', 'CH2O', 
                       'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS']

for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

X = data[['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 
          'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS']]

y = data['NObeyesdad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

joblib.dump(model, "adiposity_random_forest_model.pkl")

print("Model trained and saved successfully!")
