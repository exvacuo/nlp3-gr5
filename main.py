from src.data import load_data, split_dataset, load_both, load_headline_only
from src.preprocessing import preprocess_data, feature_engineering, transformer_preprocessor
from src.models import train_lstm, finetune_transformer
from src.evaluation import evaluate_model, collect_misclassified_samples, plot_confusion_matrix, plot_learning_curves
from transformers import AutoTokenizer
import json
import os

class Pipeline:
    """ A class to encapsulate the entire machine learning pipeline for the AG News classification task, including data loading, preprocessing, model training, evaluation, and analysis of misclassified samples."""
    def __init__(self, max_tokens:int=10000, headline_only = None, train_size: float = 1.0) -> None:
        """
        Initialize the Pipeline class with placeholders for datasets, models, and evaluation results.

        :return: None
        """
        self.train = None
        self.dev = None
        self.test = None
        self.LSTM = None
        self.Transformer = None
        self.max_tokens = max_tokens
        self.headline_only = headline_only
        self.input_type = None
        self.train_size = train_size
        
    def run(self) -> None:
        """
        Execute the entire machine learning pipeline, including data loading, preprocessing, model training, evaluation, and analysis of misclassified samples.

        :return: None
        """
        self.train, self.test = load_data()

        # Load data
        if self.headline_only == True:
            self.train, self.dev = load_headline_only(self.train)
        elif self.headline_only == False:
            self.train, self.dev = load_both(self.train)
        else:
            # Split dataset
            self.train, self.dev = split_dataset(self.train, train_size=self.train_size)
            print(f"Loaded data for pipeline {self.max_tokens}")
        
        # Preprocess data for both LSTM and Transformer models, ensuring that the same preprocessing steps are applied to all datasets for consistency.
        self.train_LSTM = preprocess_data(self.train)
        self.dev_LSTM = preprocess_data(self.dev)
        self.test_LSTM = preprocess_data(self.test)

        transformer_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        tokenized_train = transformer_preprocessor(transformer_tokenizer, self.train)
        tokenized_dev = transformer_preprocessor(transformer_tokenizer, self.dev)
        tokenized_test = transformer_preprocessor(transformer_tokenizer, self.test)

        # Maximum length of the output sequence after vectorization (padding/truncating)
        output_sequence_length = 128

        # Dimensionality of the embedding layer
        embed_dim = 64

        # Number of training epochs for both models
        epochs = 30

        # Feature engineering using TF Text Vectorization
        self.X_train_LSTM, vocab = feature_engineering(self.train_LSTM, column_name="description", max_tokens=self.max_tokens, output_sequence_length=output_sequence_length)
         # Convert from 1-indexed to 0-indexed
        self.y_train = self.train['label'].values - 1

        self.X_dev_LSTM, _ = feature_engineering(self.dev_LSTM, column_name="description", max_tokens=self.max_tokens, output_sequence_length=output_sequence_length, vocab=vocab)
        self.y_dev = self.dev['label'].values - 1

        self.X_test_LSTM, _ = feature_engineering(self.test_LSTM, column_name="description", max_tokens=self.max_tokens, output_sequence_length=output_sequence_length, vocab=vocab)
        self.y_test = self.test['label'].values - 1

        # Train LSTM model with larger batch size for faster training, using dev set for validation
        self.LSTM, self.LSTM_history = train_lstm(self.X_train_LSTM, self.y_train, X_val=self.X_dev_LSTM, y_val=self.y_dev, vocab_size=self.max_tokens, embed_dim=embed_dim, epochs=epochs, batch_size=256)
        
        # Train Transformer model with the same hyperparameters for a fair comparison, using dev set for validation
        self.Transformer, self.Transformer_history = finetune_transformer(tokenized_train, tokenized_dev, transformer_tokenizer)

        # Evaluate models on the test set
        self.LSTM_predictions, self.LSTM_metrics = evaluate_model(self.LSTM, self.X_test_LSTM, self.y_test)
        self.Transformer_predictions, self.Transformer_metrics = evaluate_model(self.Transformer, tokenized_test, self.y_test)

        # Collect misclassified for both models for creation of error categories
        self.LSTM_misclassified = collect_misclassified_samples(self.LSTM, self.X_test_LSTM, self.y_test, n_samples=10)
        self.Transformer_misclassified = collect_misclassified_samples(self.Transformer, tokenized_test, self.y_test, n_samples=10)

        self.predictions = {
            "LSTM": self.LSTM_predictions,
            "Transformer": self.Transformer_predictions
        }
        
        for model_name, y_pred in self.predictions.items():
            plot_confusion_matrix(
                self.y_test, 
                y_pred, 
                f"Confusion Matrix – {model_name}, Max Length={self.max_tokens}"
            )
        # Plot learning curves for both models, using the same max_tokens and input type in the title for clarity in comparison.
        if self.headline_only is True:
            self.input_type = "Headline Only"
        elif self.headline_only is False:
            self.input_type = "Headline + Description"
        else:
            self.input_type = "Description Only"

        # Plot learning curves for both models
        plot_learning_curves(
            {f"LSTM - {self.input_type}": self.LSTM_history},
            title=f"Learning Curve LSTM - Input Type={self.input_type}, Train Size={int(self.train_size*100)}%", 
            max_tokens=self.max_tokens
        )
        plot_learning_curves(
            {f"DistilBERT - {self.input_type}": self.Transformer_history},
            title=f"Learning Curve DistilBERT - Input Type={self.input_type}, Train Size={int(self.train_size*100)}%", 
            max_tokens=self.max_tokens
        )

def train_size_sensitivity():
    """
    Evaluate model performance when trained on different fractions of the training data.

    :return: None
    """
    for train_size in [0.25, 0.5, 0.75, 1.0]:
        pipeline = Pipeline(train_size=train_size)
        pipeline.run()
        with open(f'results/train_size_sensitivity/train_size_{int(train_size*100)}.json', 'w') as f:
            json.dump({
                "LSTM_metrics": pipeline.LSTM_metrics,
                "Transformer_metrics": pipeline.Transformer_metrics
            }, f, indent=4)
        

def input_stress_test(head_only: bool = True) -> None:
    """
    Evaluate model performance under input field stress tests by training and evaluating models using only the headline as input, and comparing it to the performance when both the headline and description are used.
    
    :param head_only: A boolean flag indicating whether to use only the headline as input (True) or both the headline and description (False) for training and evaluation.
    :return: None
    """
    pipeline = Pipeline(headline_only=head_only)
    pipeline.run()
    with open(f'results/input_stress_test/input_type_{pipeline.input_type}.json', 'w') as f:
        json.dump({
            "LSTM_metrics": pipeline.LSTM_metrics,
            "Transformer_metrics": pipeline.Transformer_metrics
        }, f, indent=4)

def label_noise_sensitivity(train_size: float=0.25) -> None:
    """
    Evaluate model sensitivity to label noise by training on different fractions of the training data and comparing performance metrics.
    
    :param train_size: The fraction of the training data to use for training (e.g., 0.25 for 25%)
    :return: None
    """
    pipeline = Pipeline(train_size=train_size)
    pipeline.run()

    with open(f'results/label_noise_sensitivity/train_size_{int(train_size*100)}.json', 'w') as f:
        json.dump({
            "LSTM_metrics": pipeline.LSTM_metrics,
            "Transformer_metrics": pipeline.Transformer_metrics
        }, f, indent=4)

def robustness_evaluation() -> None:
    """
    Evaluate model robustness under input field stress tests and training size sensitivity.
    
    :return: None
    """

    # Input field stress test: Evaluate model performance when only the headline is used as input, and compare it to the performance when both the headline and description are used. This will help determine how much the model relies on the description versus the headline for classification.
    #input_stress_test(head_only=True)
    #input_stress_test(head_only=False)

    # Train size sensitivity: Evaluate model performance when trained on different fractions of the training data (e.g., 25%, 50%, 75%, and 100%) to understand how the amount of training data affects the model's performance and its ability to generalize.
    for train_size in [0.25, 0.5, 0.75, 1.0]:
        label_noise_sensitivity(train_size=train_size)

if __name__ == "__main__":
    # # Instantiate and run the machine learning pipeline for AG News classification with max_tokens 1000
    # pipeline = Pipeline()
    # pipeline.run()

    # # Print evaluation metrics for both models and save them to JSON files, along with the misclassified samples for further analysis.
    # print("LSTM Metrics:", pipeline.LSTM_metrics)

    # # Save metrics and misclassified samples to files for further analysis and reporting.

    # os.makedirs("results", exist_ok=True)

    # with open('results/lstm_metrics.json', 'w') as f:
    #     json.dump(pipeline.LSTM_metrics, f, indent=4)
    
    # with open('results/lstm_history.json', 'w') as f:
    #     json.dump(pipeline.LSTM_history, f, indent=4)

    # with open('results/transformer_metrics.json', 'w') as f:
    #     json.dump(pipeline.Transformer_metrics, f, indent=4)
    
    # with open('results/transformer_history.json', 'w') as f:
    #     json.dump(pipeline.Transformer_history, f, indent=4)

    # # Save misclassified samples for both models
    # pipeline.LSTM_misclassified.to_csv('results/LSTM_misclassified.csv', index=False)
    # pipeline.Transformer_misclassified.to_csv('results/Transformer_misclassified.csv', index=False)

    # Run robustness evaluation to test model performance under various conditions and save results for analysis.
    robustness_evaluation()

    
