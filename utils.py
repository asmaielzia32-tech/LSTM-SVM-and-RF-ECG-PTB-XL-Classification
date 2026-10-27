import csv
import os

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, average_precision_score
from sklearn.metrics import hamming_loss


def filter_ptbxl(ptbxl_database: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out low-quality or artifact-contaminated records from the PTB-XL database.

    The following records are excluded:
    - Static noise affecting all leads ('alles' in 'static_noise')
    - Burst noise affecting all leads ('alles' in 'burst_noise')
    - Recordings from patients with a pacemaker
    - Recordings with swapped electrodes ('Elektroden vertauscht' in 'electrodes_problems')

    Args:
        ptbxl_database: pandas DataFrame loaded from ptbxl_database.csv

    Returns:
        Filtered DataFrame.
    """
    static_noise = ptbxl_database['static_noise'].str.contains('alles', na=False)
    burst_noise = ptbxl_database['burst_noise'].str.contains('alles', na=False)
    pacemaker = ptbxl_database['pacemaker'].notna()
    electrodes_problems = ptbxl_database['electrodes_problems'].str.contains('Elektroden vertauscht', na=False)

    filtered_ptbxl = ptbxl_database[~(static_noise | burst_noise | pacemaker | electrodes_problems)]
    return filtered_ptbxl


def compute_stats(dataloader) -> tuple:
    """Compute per-channel mean and std over a set of windows.
    Always call this on the *training* signals only, then reuse the
    returned stats for validation and test sets.
    Args:
        dataloader: DataLoader wrapping the training PTBDataset
    Returns:
        mean: FloatTensor (12,)
        std:  FloatTensor (12,)  -- channels with near-zero std are set to 1
    """
    all_signals = []
    for batch in dataloader:
        all_signals.append(batch["signal"])
    
    signals = torch.cat(all_signals, dim=0)  # (N, 1000, 12)
    mean = signals.mean(dim=(0, 1))
    std  = signals.std(dim=(0, 1))
    std[std < 1e-8] = 1.0
    return mean, std


def evaluate(model: nn.Module, dataloader, criterion, device) -> dict:
    """Run a full evaluation pass and return loss + classification metrics."""
    model.eval()
    running_loss = 0.0
    predictions, references, score = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            signals = batch["signal"].to(device)
            labels  = batch["label"].to(device)

            outputs = model(signals)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            raw_pred = torch.sigmoid(outputs)
            pred = (torch.sigmoid(outputs) > 0.5).float()
            predictions.extend(pred.cpu().tolist())
            references.extend(labels.cpu().tolist())
            score.extend(raw_pred.cpu().tolist())
    
  
    
    metrics = compute_metrics(predictions, references, score)
    metrics["loss"] = running_loss / len(dataloader)
    
    
    print("\nPer-class classification report:")
    class_names = ["NORM", "MI", "STTC", "CD", "HYP"]
    print(classification_report(
        references,
        predictions,
        target_names=class_names,
        digits=4,
        zero_division=0
    ))
    return metrics


def compute_metrics(predictions: list, references: list, score: list) -> dict:
    """Compute classification metrics from flat prediction and reference lists."""
    return {
        "accuracy":  accuracy_score(references, predictions),
        "precision": precision_score(references, predictions, average="macro", zero_division=0),
        "recall":    recall_score(references, predictions, average="macro", zero_division=0),
        "f1":        f1_score(references, predictions, average="macro", zero_division=0),
        "auprc":     average_precision_score(references, score, average="macro"),
        "hamming_loss": hamming_loss(references, predictions),
    }


def get_device(requested: str) -> torch.device:
    """Return the best available device matching the requested one.

    Priority: cuda > mps > cpu.  Falls back to CPU with a warning if the
    requested accelerator is not available.

    Args:
        requested: "cuda", "mps", or "cpu"
    """
    requested = requested.lower()
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("CUDA not available - falling back to CPU.")
    elif requested == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("MPS not available - falling back to CPU.")
    return torch.device("cpu")

def save_results_csv(csv_path: str, row: dict) -> None:
    """Append one result row to a CSV file.

    If the file does not exist yet, it is created and a header line is written
    first.  This way, running test.py for each model in sequence accumulates a
    clean comparison table in a single file.

    Args:
        csv_path: path to the CSV file (created if absent)
        row:      dict mapping column names to values for this run
    """
    file_exists = os.path.isfile(csv_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
