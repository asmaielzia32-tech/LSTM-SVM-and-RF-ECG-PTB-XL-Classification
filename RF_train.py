
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from shallow_preprocessing import shallow_preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import joblib
import os

from utils import compute_metrics
from yaml_config_manager import load_config

print("Random Forest training script started")

# -- 1. Configuration -----------------------------------------------------
config = load_config("config/config.yaml")

# --- 0. DATA SPLITTING AND PREPROCESSING ---
X_train_feat, y_train, X_val_feat, y_val = shallow_preprocessing(mode="train")

# --- 1. LABELS MANAGEMENT  ---
mlb = MultiLabelBinarizer()

# Compute the multi-hot encoding
y_train_bin = mlb.fit_transform(y_train)
y_val_bin   = mlb.transform(y_val)

print(f"Classes: {mlb.classes_}")

print("\nRF training...")

RF_clasifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
RF_clasifier.fit(X_train_feat,y_train_bin)

# --- 2. FEATURE PREPROCESSING ---
#scaler = StandardScaler()

# Compute mean and std on the training set, then scale it
#X_train_scaled = scaler.fit_transform(X_train_feat)

# Apply the same transformation to validation and test sets
#X_val_scaled  = scaler.transform(X_val_feat)


# --- 3. MODEL DEFINITION AND TRAINING ---
# We use OneVsRestClassifier to handle the multi-label classification



print("Training completed!")

# --- 4. EVALUATION ON THE VALIDATION SET ---
print("\nEvaluating on the validation set...")
y_val_pred = RF_clasifier.predict(X_val_feat)

probabilty = RF_clasifier.predict_proba(X_val_feat)
y_val_scores = np.column_stack([p[:, 1] for p in probabilty])

# Compute metrics
metrics = compute_metrics(y_val_pred, y_val_bin, y_val_scores)

# -- 6. Save best model ---------------------------------------------------
os.makedirs(config.training.checkpoint_dir, exist_ok=True)
# Build a descriptive name"
name = "RF.pkl"

print(f"\nValidation Results - {name}")
print("-" * 35)
for key, value in metrics.items():
    print(f"  {key:<12}: {value:.4f}")

save_path = os.path.join(config.training.checkpoint_dir, name)
joblib.dump(RF_clasifier, save_path)
joblib.dump(mlb, save_path.replace(".pkl", "_mlb.pkl"))
print(f"\nmodel saved --> {save_path}")
