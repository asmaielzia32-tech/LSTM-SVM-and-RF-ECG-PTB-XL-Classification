#!/usr/bin/env bash
# run_all.sh
# Train and evaluate 5 architectures on the PTB-XL dataset using 100Hz samplings (4 RNN's variants and 1 SVM).
# Each variant saves its own checkpoint so results never overwrite each other.
#
# Usage:
#   bash run_all.sh                        # use config defaults (Cuda)
#   bash run_all.sh --training.device mps  # run on Apple Silicon
#   bash run_all.sh --training.device cpu  # run on cpu
#
# Any extra arguments passed to this script are forwarded to every command.
# ----------------------------------------------------------------------------

EXTRA_ARGS="$@"
CFG="--config config/config.yaml"

echo "============================================================"
echo "  1 / 5  LSTM (unidirectional)"
echo "============================================================"
python rnn_train.py $CFG --model.model_type lstm $EXTRA_ARGS
python rnn_test.py  $CFG --model.model_type lstm $EXTRA_ARGS

echo ""
echo "============================================================"
echo "  2 / 5  LSTM (bidirectional)"
echo "============================================================"
python rnn_train.py $CFG --model.model_type lstm  --model.bidirectional true $EXTRA_ARGS
python rnn_test.py  $CFG --model.model_type lstm  --model.bidirectional true $EXTRA_ARGS

echo ""
echo "============================================================"
echo "  3 / 5  GRU (unidirectional)"
echo "============================================================"
python rnn_train.py $CFG --model.model_type gru $EXTRA_ARGS
python rnn_test.py  $CFG --model.model_type gru $EXTRA_ARGS

echo ""
echo "============================================================"
echo "  4 / 5  GRU (bidirectional)"
echo "============================================================"
python rnn_train.py $CFG --model.model_type gru  --model.bidirectional true $EXTRA_ARGS
python rnn_test.py  $CFG --model.model_type gru  --model.bidirectional true $EXTRA_ARGS

echo ""
echo "============================================================"
echo "  5 / 5  SVM (Rbf Kernel)"
echo "============================================================"
python svm_train.py $CFG --shallow.kernel rbf $EXTRA_ARGS
python svm_test.py  $CFG --shallow.kernel rbf $EXTRA_ARGS

echo ""
echo "All runs completed. Checkpoints are in ./checkpoints/"