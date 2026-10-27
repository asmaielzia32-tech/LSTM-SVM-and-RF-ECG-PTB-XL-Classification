
Overview

This project is based on the original implementation by Vito Cammarata, which compares shallow and deep learning approaches for ECG classification using the PTB-XL dataset.

Original repository: https://github.com/VitoCammarata/Shallow-and-Deep-Learning-Comparison_PTB-XL-Dataset

The project was extended with an additional Random Forest (RF) model to compare its performance with the existing SVM and LSTM approaches. The SVM and LSTM implementations were adapted from the original repository, and credit is given to the original author for the code used.

My Contributions

1. Adapted the structure of svm_train.py to implement the Random Forest training script (RF_train.py).

2. Adapted the structure of svm_test.py to implement the Random Forest testing script (RF_test.py).

3. Added per-diagnostic precision, recall, and F1-score reporting to RF_test.py, svm_test.py, and rnn_test.py to provide a more detailed comparison of model performance across the five diagnostic categories.

Evaluated and documented the results for Random Forest, SVM, and LSTM:

| Model | Feature/Epochs | Accuracy | Precision | Recall | F1-Score | Hamming-Lose|
| ----- | -------------- | -------- | --------- | ------ | -------- | ----------- |
| RF    | 48 Features    | 0.3812   | 0.7470    | 0.3607 | 0.4569   | 0.1887      |
| SVM   | 48 Features    | 0.3549   | 0.5115    | 0.7028 | 0.5887   | 0.2323      |
| LSTM  | 10 epochs      | 0.4363   | 0.5569    | 0.7694 | 0.6412   | 0.1964      |


RF 5 diagnostics: 

| Diagnostics | Precision | Recall | F1-score | Support | 
| ----------- | --------- | ------ | -------- | ------- |
| CD          | 0.8223    | 0.4261 | 0.5614   | 467     |
| HYP         | 0.7524    | 0.3110 | 0.4401   | 254     |
| MI          | 0.7317    | 0.2841 | 0.4093   | 528     |
| NORM        | 0.7141    | 0.6823 | 0.6978   | 941     |
| STTC        | 0.7143    | 0.1002 | 0.1757   | 499     |


|             | Precision | Recall | F1-score | Support | 
| ----------- | --------- | ------ | -------- | ------- |
| Micro avg   | 0.7364    | 0.4165 | 0.5321   | 2689    |
| Macro avg   | 0.7270    | 0.3607 | 0.4569   | 2689    |
| Weighted avg| 0.7400    | 0.4165 | 0.4962   | 2689    |
| Samples avg | 0.4779    | 0.4419 | 0.4505   | 2689    |


SVM 5 diagnostics: 

| Diagnostics | Precision | Recall | F1-score | Support | 
| ----------- | --------- | ------ | -------- | ------- |
| CD          | 0.5712    | 0.6531 | 0.6094   | 467     |
| HYP         | 0.3952    | 0.7126 | 0.5084   | 254     |
| MI          | 0.4986    | 0.6553 | 0.5663   | 528     |
| NORM        | 0.6672    | 0.8459 | 0.7460   | 941     |
| STTC        | 0.4266    | 0.6473 | 0.5135   | 499     |


|             | Precision | Recall | F1-score | Support | 
| ----------- | --------- | ------ | -------- | ------- |
| Micro avg   | 0.5363    | 0.7255 | 0.6167   | 2689    |
| Macro avg   | 0.5115    | 0.7028 | 0.5887   | 2689    |
| Weighted avg| 0.5469    | 0.7255 | 0.6214   | 2689    |
| Samples avg | 0.5823    | 0.7283 | 0.6200   | 2689    |

LSTM 5 diagnostics: 

| Diagnostics | Precision | Recall | F1-score | Support | 
| ----------- | --------- | ------ | -------- | ------- |
| CD          | 0.4961    | 0.6724 | 0.5709   | 467     |
| HYP         | 0.3874    | 0.7520 | 0.5114   | 254     |
| MI          | 0.5371    | 0.7538 | 0.6273   | 528     |
| NORM        | 0.7827    | 0.8650 | 0.8218   | 941     |
| STTC        | 0.5812    | 0.8035 | 0.6745   | 499     |


|             | Precision | Recall | F1-score | Support | 
| ----------- | --------- | ------ | -------- | ------- |
| Micro avg   | 0.5888    | 0.7877 | 0.6739   | 2689    |
| Macro avg   | 0.5569    | 0.7694 | 0.6412   | 2689    |
| Weighted avg| 0.6100    | 0.7877 | 0.6834   | 2689    |
| Samples avg | 0.6461    | 0.7841 | 0.6819   | 2689    |




