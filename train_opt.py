import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

# ==============================================================================
# PART 1: Fertilizer Prediction Model
# ==============================================================================
print("--- Step 1: Training Optimal N Fertilizer Model ---")

# 1. Load Data
df_fer = pd.read_csv('/train_data_fer.csv')

features_fer = [
    'rain', 'tmax', 'tmin', 'srad', 'lon', 'lat',
    'AWC', 'SBDM_mean', 'SLOC_mean', 'SLCL_mean', 'SLSI_mean', 'SLHW_mean', 'SCEC_mean',
    'rain_more_50', 'maxt_sum2',
    'irrigation', # Categorical: Rainfed vs Irrigated
    'zone', 'scenario'
]
target_fer = 'fer'

# Select features and target, remove NaNs
df_model_fer = df_fer[features_fer + [target_fer]].dropna()
X = df_model_fer[features_fer]
y = df_model_fer[target_fer]

# 2. One-Hot Encoding
# 'irrigation' is included here as it is a categorical variable
cols_to_encode = ['zone', 'scenario', 'irrigation']
X_encoded = pd.get_dummies(X, columns=cols_to_encode, dummy_na=False)

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# 4. Model Training
model_fer = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50
)

model_fer.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# 5. Evaluation and Saving
y_pred = model_fer.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"Fertilizer Model R²: {r2:.3f}")

joblib.dump(model_fer, '/model_fer.pkl')


# ==============================================================================
# PART 2: Irrigation Prediction Model
# ==============================================================================
print("\n--- Step 2: Training Irrigation Model ---")

# 1. Load Data
df_irri = pd.read_csv('/train_data_irri.csv')

features_irri = [
    'rain', 'tmax', 'srad', 'lon', 'lat',
    'AWC', 'SBDM_mean', 'SLOC_mean', 'SLCL_mean', 'SLSI_mean', 'SLHW_mean',
    'maxt1', 'rain1', 'srad1',
    'maxt2', 'rain2', 'srad2',
    'zone', 'scenario'
]
target_irri = 'irri'

# Select features and target, remove NaNs
df_model_irri = df_irri[features_irri + [target_irri]].dropna()
X = df_model_irri[features_irri]
y = df_model_irri[target_irri]

# 2. One-Hot Encoding
X_encoded = pd.get_dummies(X, columns=['zone', 'scenario'], dummy_na=False)

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# 4. Log Transformation (Log1p)
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

# 5. Model Training
model_irri = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50
)

# Train on log-transformed data
model_irri.fit(
    X_train, y_train_log,
    eval_set=[(X_test, y_test_log)],
    verbose=False
)

# 6. Prediction and Inverse Transformation
y_pred_log = model_irri.predict(X_test)
y_pred = np.expm1(y_pred_log) # Inverse of log1p

# Evaluation on original scale
r2 = r2_score(y_test, y_pred)
print(f"Irrigation Model R²: {r2:.3f}")

joblib.dump(model_irri, '/model_irri.pkl')