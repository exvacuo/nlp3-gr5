from __future__ import annotations
from typing import TypeAlias, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    DistilBertForSequenceClassification, 
    Trainer,
    TrainingArguments,
)
from src.preprocessing import transformer_preprocessor

# Type alias for pandas Series or numpy array
PandasSeriesAny: TypeAlias = Any
class TextLSTM(nn.Module):
    """An LSTM for text clasification"""
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int)-> None:
        """
        Initialize the TextLSTM model with an embedding layer, LSTM layer, and fully connected layer.
        
        :param vocab_size: Size of the vocabulary (number of unique tokens)
        :param embed_dim: Dimensionality of the embedding layer
        :param num_classes: Number of output classes for classification
        :return: None
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the forward pass of the Text LSTM model, with embedding, LSTM, and fully connected layers to the input tensor.
        
        :param x: Input tensor of shape (batch_size, sequence_length) containing token indices
        :return: Output tensor of shape (batch_size, num_classes) containing class scores
        """
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        x = self.fc(hidden[-1])
        return x

def train_lstm(
    X_train: np.ndarray,
    y_train: PandasSeriesAny | np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: PandasSeriesAny | np.ndarray | None = None,
    vocab_size: int = 20000,
    embed_dim: int = 128,
    num_classes: int = 4,
    epochs: int = 2,
    batch_size: int = 128,
    patience: int = 3,
):
    # Determine the device to use for training (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TextLSTM(vocab_size=vocab_size, embed_dim=embed_dim, num_classes=num_classes)

    # Move the model to the appropriate device -GPU or CPU
    model.to(device)

    # Convert training data to PyTorch tensors and create a DataLoader for batching
    X_train_tensor = torch.tensor(X_train, dtype=torch.long)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    # Create a DataLoader for the training data, which will handle batching and shuffling
    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=batch_size,
        shuffle=True,
    )

    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    # Define the optimizer for updating model parameters during training
    optimizer = torch.optim.Adam(model.parameters())

    # Early stopping setup
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    # Track training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'stopped_epoch': None
    }

    # Training loop for the specified number of epochs
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation and early stopping
        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                X_val_tensor = torch.from_numpy(np.asarray(X_val)).long().to(device)
                y_val_tensor = torch.from_numpy(np.asarray(y_val)).long().to(device)
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
            
            history['val_loss'].append(val_loss)
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict().copy()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    history['stopped_epoch'] = epoch + 1
                    model.load_state_dict(best_model_state)
                    break
        else:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f}")

    return model, history


def finetune_transformer(tokenized_train, tokenized_dev):    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=4,
        problem_type="single_label_classification",
    )

    training_args = TrainingArguments(
        output_dir="results/transformer",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        tokenizer=tokenizer
    )

    trainer.train()

    train_loss: list[float] = []
    val_loss: list[float] = []
    for log in getattr(trainer.state, "log_history", []) or []:
        if "loss" in log and "eval_loss" not in log:
            train_loss.append(float(log["loss"]))
        if "eval_loss" in log:
            val_loss.append(float(log["eval_loss"]))

    history = {"train_loss": train_loss, "val_loss": val_loss, "stopped_epoch": None}

    return model, history
