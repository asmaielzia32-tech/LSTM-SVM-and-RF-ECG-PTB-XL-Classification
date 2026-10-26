import torch
import torch.nn as nn

# Maps the model_type string to the corresponding PyTorch RNN class
_RNN_REGISTRY = {
    "rnn":  nn.RNN,
    "lstm": nn.LSTM,
    "gru":  nn.GRU,
}

class Classifier(nn.Module):
    def __init__(
        self,
        model_type: str,
        bidirectional: bool,
        input_size: int,
        hidden_size: int,   
        num_layers: int,
        dropout: float,
        num_classes: int,
    ):

        super().__init__()

        model_type = model_type.lower()
        if model_type not in _RNN_REGISTRY:
            raise ValueError(
                f"model_type='{model_type}' is not supported. "
                f"Choose one of {sorted(_RNN_REGISTRY)}."
            )

        self.model_type   = model_type
        self.bidirectional = bidirectional

        rnn_cls = _RNN_REGISTRY[model_type]
        # PyTorch's dropout arg only applies *between* layers (needs num_layers > 1)
        rnn_dropout = dropout if num_layers > 1 else 0.0

        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=rnn_dropout,
            batch_first=True,        # shapes: (batch, seq_len, features) (BTS)
            bidirectional=bidirectional,
        )

        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fc_input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size) - raw sensor window

        Returns:
            logits: (batch, num_classes)
        """
        # out: (batch, seq_len, hidden_size * num_directions)
        # _  : hidden state(s) - not needed for classification
        out, _ = self.rnn(x)

        # Take the last timestep as the sequence representation
        out = out[:, -1, :]           # (batch, hidden_size * num_directions)
        out = self.dropout(out)
        return self.fc(out)           # (batch, num_classes)

    # Provide a clear textual description of the model
    def __repr__(self) -> str:
        direction = "bidirectional" if self.bidirectional else "unidirectional"
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"{self.__class__.__name__}("
            f"model_type={self.model_type}, {direction}, "
            f"parameters={n_params:,})"
        )