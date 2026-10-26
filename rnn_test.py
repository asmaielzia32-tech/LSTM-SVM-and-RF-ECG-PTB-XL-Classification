import os

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from yaml_config_manager import load_config
from sklearn.metrics import classification_report

from data_classes.ptb_dataset import PTBDataset
from model_classes.rnn import Classifier 
from utils import filter_ptbxl, evaluate, get_device, save_results_csv


if __name__ == "__main__":

    # -- 1. Configuration -----------------------------------------------------
    config = load_config("config/config.yaml")
    device = get_device(config.training.device)

    csv_path = os.path.join(config.data.ptbxl_dataset, 'ptbxl_database.csv')
    df = pd.read_csv(csv_path, index_col=0)

    #filter the noisy samples from the dataset
    df = filter_ptbxl(df)
    
    #take the test split
    df_test  = df[df['strat_fold'] == 10]

    print(f"Test: {len(df_test)}")


    # -- 2. Data --------------------------------------------------------------
    # Load normalization stats that were saved during training (if enabled).
    # Using the train-set stats here ensures the test set is processed
    # identically - without ever using test data to compute statistics.
    mean, std = None, None
    if config.data.normalize:
        stats_path = os.path.join(config.training.checkpoint_dir, "norm_stats.pt")
        if not os.path.isfile(stats_path):
            raise FileNotFoundError(
                f"Normalization stats not found at '{stats_path}'. "
                "Run train.py first."
            )
        stats = torch.load(stats_path, map_location="cpu", weights_only=True)
        mean, std = stats["mean"], stats["std"]

    test = PTBDataset(config.data.ptbxl_dataset, config.data.sampling_rate, df_test, mean, std)
    test_dl = DataLoader(test, batch_size=config.training.batch_size, shuffle=False, num_workers=config.training.num_workers, pin_memory=config.training.pin_memory)

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

    prefix = "bi" + config.model.model_type if config.model.bidirectional else config.model.model_type
    full_prefix = f"{prefix}_h{config.model.hidden_size}_b{config.training.batch_size}_lr{config.training.learning_rate}.pt"
    checkpoint_path = os.path.join(config.training.checkpoint_dir, full_prefix)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: '{checkpoint_path}'. "
            "Run train.py first."
        )

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.to(device)
    direction = "BiDirectional" if config.model.bidirectional else "UniDirectional"
    print(f"Loaded : {config.model.model_type.upper()} ({direction}) from {checkpoint_path}")
    print(f"Device : {device}")

    # -- 4. Evaluate ----------------------------------------------------------
    criterion = nn.BCEWithLogitsLoss()
    test_metrics = evaluate(model, test_dl, criterion, device)

    print(f"\nTest Results - {full_prefix}")
    print("-" * 35)
    for key, value in test_metrics.items():
        print(f"  {key:<12}: {value:.4f}")

    # -- 5. Save to CSV -------------------------------------------------------
    row = {
        "model":         full_prefix,
        "model_type":    config.model.model_type,
        "bidirectional": config.model.bidirectional,
        "hidden_size":   config.model.hidden_size,
        "num_layers":    config.model.num_layers,
        "batch_size":    config.training.batch_size,
        "dropout":       config.model.dropout,
        "learning_rate": config.training.learning_rate,
        **{k: round(v, 4) for k, v in test_metrics.items()},
    }
    save_results_csv(config.training.results_csv, row)
    print(f"\nResults appended --> {config.training.results_csv}")