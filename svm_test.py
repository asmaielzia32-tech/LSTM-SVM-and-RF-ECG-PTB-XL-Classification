from shallow_preprocessing import shallow_preprocessing
import joblib
import os
from sklearn.metrics import classification_report
from sklearn.metrics import hamming_loss

from utils import compute_metrics, save_results_csv
from yaml_config_manager import load_config

if __name__ == "__main__":

    # -- 1. Configuration -----------------------------------------------------
    config = load_config("config/config.yaml")

    # -- 2. Load saved model artifacts ----------------------------------------
    name = f"{config.shallow.model_type}_{config.shallow.kernel}.pkl"
    model_path  = os.path.join(config.training.checkpoint_dir, name)
    scaler_path = model_path.replace(".pkl", "_scaler.pkl")
    mlb_path    = model_path.replace(".pkl", "_mlb.pkl")

    for p in (model_path, scaler_path, mlb_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"Artifact not found: '{p}'. Run svm.py first."
            )

    svm_clf = joblib.load(model_path)
    scaler  = joblib.load(scaler_path)
    mlb     = joblib.load(mlb_path)

    # -- 3. Load and preprocess test signals ----------------------------------
    X_test_feat, y_test = shallow_preprocessing(mode="test")
    y_test_bin  = mlb.transform(y_test)
    X_test_scaled = scaler.transform(X_test_feat)

    # -- 4. Evaluate ----------------------------------------------------------
    y_pred = svm_clf.predict(X_test_scaled)
    h_loss = hamming_loss(y_test_bin, y_pred)
    print(f"Hamming Loss: {h_loss:.4f}")
    
    print(classification_report(
    y_test_bin,
    y_pred,
    target_names=mlb.classes_,
    digits=4,
    zero_division=0
    ))
 
    metrics = compute_metrics(y_pred, y_test_bin, svm_clf.decision_function(X_test_scaled))
 
    print(f"\nTest Results - {name}")
    print("-" * 35)
    for key, value in metrics.items():
        print(f"  {key:<12}: {value:.4f}")

    # -- 6. Save to CSV -------------------------------------------------------
    row = {
        "model":         name,
        "model_type":    f"{config.shallow.model_type}_{config.shallow.kernel}",
        "bidirectional": "None",
        "hidden_size":   "None",
        "num_layers":    "None",
        "batch_size":    "None",
        "dropout":       "None",
        "learning_rate": "None",
        **{k: round(v, 4) for k, v in metrics.items()},
        "loss":          "None"
    }
    save_results_csv(config.training.results_csv, row)
    print(f"\nResults appended --> {config.training.results_csv}")
