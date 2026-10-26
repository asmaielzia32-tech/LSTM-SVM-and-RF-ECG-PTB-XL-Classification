# ECG Arrhythmia Classification with RNNs and SVM — PTB-XL Dataset

A project for multi-label classification of cardiac abnormalities from 12-lead ECG signals, developed for the Machine Learning course at Università Kore of Enna.

The project compares recurrent neural network architectures (RNN, LSTM, GRU, and their bidirectional variants) against a classical SVM baseline on the [PTB-XL dataset](https://physionet.org/content/ptb-xl/1.0.3/), a large publicly available 12-lead ECG dataset.

| Information | Details |
|---|---|
| Authors | *Vito Cammarata* and *Gabriele Trovato* |
| Course | Machine Learning @ UniKore |
| Dataset | PTB-XL — 12-lead ECG (PhysioNet) |
| Task | Multi-label cardiac superclass classification (5 classes) |

---

## Task Description

Each ECG recording in PTB-XL is annotated with one or more diagnostic **superclasses**:

| Label | Description |
|---|---|
| NORM | Normal ECG |
| MI | Myocardial Infarction |
| STTC | ST/T-wave Changes |
| CD | Conduction Disturbance |
| HYP | Hypertrophy |

This is a **multi-label** problem: a single recording can belong to more than one class simultaneously. Models are evaluated with macro F1, AUPRC, precision and recall.

The input signal is either 1000 timesteps × 12 channels (100 Hz) or 5000 × 12 (500 Hz), configurable via `config.yaml`.

---

## Approach

Two families of models are compared:

**Deep models** — a single `Classifier` class (in `model_classes/rnn.py`) wraps PyTorch's `nn.RNN`, `nn.LSTM`, or `nn.GRU` with an optional bidirectional flag, a dropout layer, and a linear classification head. Training uses `BCEWithLogitsLoss` with per-class positive weights to handle label imbalance, an Adam optimizer with a linear warmup + linear decay schedule, and early stopping on validation F1.

**Shallow baseline** — a One-vs-Rest SVM (`svm.py`) trained on handcrafted temporal features (per-channel mean, min, max, std over time), extracted in `shallow_preprocessing.py`.

Low-quality recordings (static noise, burst noise, pacemaker, swapped electrodes) are filtered out before any split.

---

## Project Structure

```
.
├── config/
│   └── config.yaml              # all hyperparameters
├── data_classes/
│   └── ptb_dataset.py           # PTBDataset (PyTorch Dataset)
├── model_classes/
│   └── rnn.py                   # Classifier: RNN / LSTM / GRU
├── checkpoints/                 # saved model weights (created at runtime)
├── results/
│   └── comparison.csv           # test results accumulated across runs
├── rnn_train.py                 # deep model training loop
├── rnn_test.py                  # deep model evaluation on the test set
├── svm_train.py                 # SVM training and validation
├── svm_test.py                  # SVM testing
├── shallow_preprocessing.py     # feature extraction for the SVM
├── utils.py                     # filtering, metrics, evaluate(), stats
└── environment.yml
```

---

## Environment Setup
 
Requires Python ≥ 3.9 and [Conda](https://docs.conda.io/en/latest/) (Miniconda or Anaconda).
 
```bash
conda env create -f environment.yml
conda activate ML
```

The environment includes all the necessary dependencies. If you want to install PyTorch with Cuda Support specifically for your GPU, consult [pytorch.org](https://pytorch.org/get-started/locally/) and replace the corresponding voice in the `environment.yml`.


---

## Dataset Download

The PTB-XL dataset is hosted on PhysioNet and is freely available. This requires about **1.8 GB** of disk space.

You can download it from:

> https://physionet.org/content/ptb-xl/1.0.3/

and place the extracted folder at:

```
./dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/
```


---

## Configuration

All hyperparameters are in `config.yaml`. The most relevant entries:

```yaml
model:
  model_type: "lstm"      # rnn | lstm | gru
  bidirectional: false
  hidden_size: 256
  num_layers: 1
  dropout: 0.3
  num_classes: 5

training:
  epochs: 100
  batch_size: 64
  learning_rate: 1e-3
  monitor_metric: "f1"
  early_stopping_patience: 15

shallow:
  model_type: "svm"
  kernel: "rbf"
```

---

## Training

### Deep models (RNN / LSTM / GRU)

Edit `config.yaml` and run:

```bash
python rnn_train.py --config config/config.yaml
```

The best checkpoint is saved to `checkpoints/` with a descriptive name encoding the architecture and hyperparameters, e.g. `lstm_h256_b64_lr0.001.pt`. Normalization statistics computed on the training set are saved to `checkpoints/norm_stats.pt`.

To compare multiple architectures, run the script `run_all.sh`:

```bash
chmod +x run_all.sh
./run_all.sh
```

### SVM baseline

```bash
python svm.py --config config/config.yaml
```

The trained model, scaler, and label binarizer are saved to `checkpoints/`.

---

## Evaluation

```bash
python rnn_test.py --config config/config.yaml
```

This loads the checkpoint matching the current `config.yaml` settings, evaluates on fold 10 (the held-out test set), prints per-metric results, and appends a row to `results/comparison.csv`.

Run once per trained checkpoint to accumulate the full comparison table.

---

## Results

Test set results (fold 10). All deep models: 1 layer, dropout 0.3, learning rate 1e-3, z-score normalization, 100 Hz.

### SVM Baseline

| Model | F1 (macro) | AUPRC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|
| SVM — RBF | 0.5887 | 0.6221 | 0.3549 | 0.5115 | 0.7028 |
| SVM — Linear | 0.5417 | 0.5318 | 0.2945 | 0.4552 | 0.6806 |
| SVM — Poly | 0.5383 | 0.5455 | 0.4301 | 0.5692 | 0.5538 |

### RNN

| Config | F1 (macro) | AUPRC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|
| hidden=128, batch=64 | 0.3723 | 0.2916 | 0.2122 | 0.3041 | 0.4943 |
| hidden=128, batch=128 | 0.4065 | 0.3342 | 0.1557 | 0.3460 | 0.5362 |
| hidden=256, batch=64 | 0.3841 | 0.3056 | 0.3482 | 0.3328 | 0.4704 |
| hidden=256, batch=128 | 0.3931 | 0.3091 | 0.1082 | 0.3263 | 0.5479 |

### LSTM (Unidirectional)

| Config | F1 (macro) | AUPRC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|
| hidden=128, batch=64 | 0.6965 | 0.7560 | 0.5153 | 0.6302 | 0.7837 |
| hidden=128, batch=128 | 0.6881 | 0.7499 | 0.5048 | 0.6222 | 0.7770 |
| hidden=256, batch=64 | 0.7130 | 0.7731 | 0.5263 | 0.6472 | 0.7982 |
| hidden=256, batch=128 | 0.7148 | 0.7781 | 0.5364 | 0.6471 | 0.8097 |

### GRU (Unidirectional)

| Config | F1 (macro) | AUPRC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|
| hidden=128, batch=64 | 0.7068 | 0.7682 | 0.5187 | 0.6508 | 0.7796 |
| hidden=128, batch=128 | 0.7133 | 0.7810 | 0.5465 | 0.6595 | 0.7834 |
| hidden=256, batch=64 | 0.7150 | 0.7866 | 0.5378 | 0.6591 | 0.7955 |
| hidden=256, batch=128 | 0.7152 | 0.7813 | 0.5307 | 0.6538 | 0.7979 |

### BiLSTM / BiGRU

| Model | Config | F1 (macro) | AUPRC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|
| BiLSTM | hidden=256, batch=128 | **0.7070** | **0.7687** | 0.5340 | 0.6522 | 0.7829 |
| BiGRU | hidden=256, batch=128 | 0.4916 | 0.4995 | 0.3271 | 0.4051 | 0.6531 |

### Key Takeaways

- The vanilla **RNN** performs poorly across all configurations, consistently below 0.34 AUPRC, confirming the well-known difficulty of learning long-range dependencies with simple recurrence.
- **LSTM and GRU** achieve comparable performance (~0.75–0.79 AUPRC), with GRU being slightly more stable across configurations.
- The **BiLSTM** matches the best unidirectional results without a meaningful gain, suggesting that for this task the future context does not add significant information.
- The **BiGRU** result (0.50 AUPRC) is an outlier and likely reflects a training instability rather than a property of the architecture.
- All deep models comfortably outperform the SVM baselines, with the best SVM (RBF, 0.62 AUPRC) falling about 15 points behind the best recurrent models.