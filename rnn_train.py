import os
import pandas as pd
import torch
import torch.nn as nn

from tqdm import tqdm
from yaml_config_manager import load_config
from torch.utils.data import DataLoader
from data_classes.ptb_dataset import PTBDataset
from model_classes.rnn import Classifier 
from utils import filter_ptbxl, compute_stats, evaluate, compute_metrics, get_device

# -- Training loop -------------------------------------------------------------------

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    predictions, references, score = [], [], []

    for batch in tqdm(dataloader, desc="Training", leave=False):
        signals = batch["signal"].to(device)
        labels  = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        raw_pred = torch.sigmoid(outputs)
        pred = (torch.sigmoid(outputs) > 0.5).float()
        predictions.extend(pred.cpu().tolist())
        references.extend(labels.cpu().tolist())
        score.extend(raw_pred.cpu().tolist())

    metrics = compute_metrics(predictions, references, score)
    metrics["loss"] = running_loss / len(dataloader)
    return metrics

def _update_best(model, metric_name, val_metrics, best_metric, best_state, lower_is_better):
    """Check whether the current epoch improved the monitored metric.

    Returns:
        best_metric: updated best value
        best_state:  updated state dict (cloned to CPU) if improved, else unchanged
        improved:    bool - True when a new best was found
    """
    current = val_metrics[metric_name]
    improved = current < best_metric if lower_is_better else current > best_metric
    if improved:
        print(f"  --> Found new best - val {metric_name}: {current:.4f} - saving checkpoint")
        best_metric = current
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    return best_metric, best_state, improved

# -- Entry point ---------------------------------------------------------------------

if __name__ == "__main__":

    # -- 1. Configuration -----------------------------------------------------
    config = load_config("config/config.yaml")
    
    csv_path = os.path.join(config.data.ptbxl_dataset, 'ptbxl_database.csv')
    df = pd.read_csv(csv_path, index_col=0)

    #filter the noisy samples from the dataset
    df = filter_ptbxl(df)
    
    #split the dataset into train, val and test using the strat_fold column
    df_train = df[df['strat_fold'] <= 8]
    df_val   = df[df['strat_fold'] == 9]
    df_test  = df[df['strat_fold'] == 10]

    print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")


    # -- 2. Data --------------------------------------------------------------
    # Load raw signals first (no normalization yet)

    train = PTBDataset(config.data.ptbxl_dataset, config.data.sampling_rate, df_train)
    val = PTBDataset(config.data.ptbxl_dataset, config.data.sampling_rate, df_val)

    #sample = train[0]
    #print(sample["signal"].shape)
    #print(sample["signal"])
    #print(sample["label"])

    mean,std = None, None
    if config.data.normalize: 

        stat_dl = DataLoader(train, batch_size=config.training.batch_size, shuffle=True, num_workers=config.training.num_workers, pin_memory=config.training.pin_memory)

        mean, std = compute_stats(stat_dl)
        print(f"Normalization: per-channel z-score  (mean range [{mean.min():.3f}, {mean.max():.3f}])")
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)
        torch.save({'mean': mean, 'std': std}, 
                   os.path.join(config.training.checkpoint_dir, "norm_stats.pt"))
        train = PTBDataset(config.data.ptbxl_dataset, config.data.sampling_rate, df_train, mean, std)
        val = PTBDataset(config.data.ptbxl_dataset, config.data.sampling_rate, df_val, mean, std)
    
    train_dl = DataLoader(train, batch_size=config.training.batch_size, shuffle=True, num_workers=config.training.num_workers, pin_memory=config.training.pin_memory)
    val_dl = DataLoader(val, batch_size=config.training.batch_size, shuffle=False, num_workers=config.training.num_workers, pin_memory=config.training.pin_memory) 

# -- 3. Model -------------------------------------------------------------
    model = Classifier(
        model_type=config.model.model_type,
        bidirectional=config.model.bidirectional,
        input_size=config.model.input_size,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        num_classes=config.model.num_classes,
    )

    device = get_device(config.training.device)

    model.to(device)
    print(f"Model  : {model}")
    print(f"Device : {device}")

    # -- 4. Optimizer & scheduler ---------------------------------------------
    with torch.no_grad():
        all_labels = torch.cat([batch["label"] for batch in train_dl], dim=0)
    pos_counts = all_labels.sum(dim=0)
    neg_counts = len(all_labels) - pos_counts
    pos_weight = (neg_counts / pos_counts.clamp(min=1)).to(device)
    del all_labels
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)

    total_steps = len(train_dl) * config.training.epochs
    warmup_steps = int(total_steps * config.training.warmup_ratio)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step/ max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
     
    # -- 5. Training loop -----------------------------------------------------
    lower_is_better = config.training.monitor_lower_is_better
    best_val_metric = float('inf') if lower_is_better else float('-inf')
    best_state = None
    patience = config.training.early_stopping_patience
    epochs_no_improve = 0

    for epoch in range(config.training.epochs):
        print(f"\nEpoch {epoch + 1}/{config.training.epochs}")
        train_metrics = train_one_epoch(model, train_dl, criterion, optimizer, scheduler, device)
        val_metrics = evaluate(model, val_dl, criterion, device)

        print(
            f"  [train]: loss={train_metrics['loss']:.4f}  "
            f"acc={train_metrics['accuracy']:.4f}  f1={train_metrics['f1']:.4f}"
        )
        print(
            f"  [val]:    loss={val_metrics['loss']:.4f}  "
            f"acc={val_metrics['accuracy']:.4f}  f1={val_metrics['f1']:.4f}"
        )

        best_val_metric, best_state, improved = _update_best(
            model,
            config.training.monitor_metric,
            val_metrics,
            best_val_metric,
            best_state,
            lower_is_better
        )

        if improved:
            epochs_no_improve = 0
        else: 
            epochs_no_improve += 1
            if patience > 0:
                print(f"  No improvement for {epochs_no_improve}/{patience} epoch(s)")
                if epochs_no_improve >= patience:
                    print(f"Early stopping after {epoch + 1} epochs without improvement.")
                    break
                
# -- 6. Save best model ---------------------------------------------------
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    # Build a descriptive name: e.g. "best_bilstm.pt" or "best_gru.pt"
    prefix = "bi" + config.model.model_type if config.model.bidirectional else config.model.model_type
    full_prefix = f"{prefix}_h{config.model.hidden_size}_b{config.training.batch_size}_lr{config.training.learning_rate}.pt"
    save_path = os.path.join(config.training.checkpoint_dir, full_prefix)
    torch.save(best_state, save_path)
    print(f"\nBest model saved --> {save_path}")