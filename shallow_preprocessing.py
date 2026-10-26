import pandas as pd
import numpy as np
import wfdb
import ast
from yaml_config_manager import load_config
from utils import filter_ptbxl

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

def aggregate_diagnostic(y_dic, agg_df):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

def extract_temporal_features(data):
    """From a tensor (N, 1000, 12) create a matrix (N, 48)"""
    X_mean = np.mean(data, axis=1)
    X_min  = np.min(data, axis=1)
    X_max  = np.max(data, axis=1)
    X_std  = np.std(data, axis=1)
    return np.concatenate([X_mean, X_min, X_max, X_std], axis=1)

def shallow_preprocessing(mode):
    config = load_config("config/config.yaml")
    path = config.data.ptbxl_dataset
    sampling_rate=config.data.sampling_rate


    dataset = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    Y = filter_ptbxl(dataset)

    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    train_mask = Y.strat_fold <= 8
    val_mask   = Y.strat_fold == 9
    test_mask  = Y.strat_fold == 10

    # Load raw signal data
    X = load_raw_data(Y, sampling_rate, path)

    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]


    # Apply diagnostic superclass
    Y['diagnostic_superclass'] = Y.scp_codes.apply(lambda x: aggregate_diagnostic(x, agg_df))

    X_train_raw, y_train = X[np.where(train_mask)], Y[train_mask].diagnostic_superclass
    X_val_raw,   y_val   = X[np.where(val_mask)],   Y[val_mask].diagnostic_superclass
    X_test_raw,  y_test  = X[np.where(test_mask)],  Y[test_mask].diagnostic_superclass

    X_train_feat = extract_temporal_features(X_train_raw)
    X_val_feat   = extract_temporal_features(X_val_raw)
    X_test_feat  = extract_temporal_features(X_test_raw)

    if mode == "train":
        return X_train_feat, y_train, X_val_feat, y_val
    elif mode == "test":
        return X_test_feat, y_test