import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Step 1: Generate Synthetic HR Dataset
print("--- Generating Synthetic HR Dataset ---")
np.random.seed(42)
n_samples = 1000

data = {
    'employee_id': [f"EMP{i:04d}" for i in range(1, n_samples + 1)],
    'age': np.random.randint(22, 60, size=n_samples),
    'experience_years': np.random.randint(0, 35, size=n_samples),
    'department': np.random.choice(['Engineering', 'Sales', 'HR', 'Marketing'], size=n_samples),
    'salary': np.random.randint(30000, 150000, size=n_samples),
    'training_hours': np.random.randint(5, 50, size=n_samples),
    'on_time_delivery_rate': np.random.uniform(0.5, 1.0, size=n_samples),
    'manager_score': np.random.randint(1, 6, size=n_samples),
    'perf_band_next': np.random.choice(['Low', 'Medium', 'High'], size=n_samples, p=[0.2, 0.5, 0.3])
}

df = pd.DataFrame(data)
print(f"Dataset Shape: {df.shape}")
print(df.head())

# Step 2: Define Target and Features
target = 'perf_band_next'
X = df.drop(columns=['employee_id', target])
y = df[target]

cat_cols = X.select_dtypes(include='object').columns
num_cols = X.select_dtypes(include='number').columns

# Step 3: Create Preprocessing Pipeline
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])

# Step 4: Build Model Pipeline
pipe = Pipeline([
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))
])

# Step 5: Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Step 6: Hyperparameter Tuning using GridSearchCV
print("\n--- Training Model (GridSearchCV) ---")
param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [6, 10],
    'clf__min_samples_leaf': [1, 3]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid_search = GridSearchCV(pipe, param_grid, scoring='f1_macro', cv=cv, n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best CV Macro-F1 Score: {grid_search.best_score_:.3f}")
best_model = grid_search.best_estimator_

# Step 7: Evaluate on Test Set
print("\n--- Evaluation Report ---")
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 8: Save the Model
model_filename = "employee_perf_model.pkl"
joblib.dump(best_model, model_filename)
print(f"\nModel successfully saved as {model_filename}")
