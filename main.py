from src.data import load_data, split_dataset
from src.preprocessing import preprocess_data, feature_engineering
from src.models import train_model
from src.evaluation import evaluate_model, collect_misclassified_samples, plot_confusion_matrix, plot_learning_curves
import json

class Pipeline:
    """ A class to encapsulate the entire machine learning pipeline for the AG News classification task, including data loading, preprocessing, model training, evaluation, and analysis of misclassified samples."""
    def __init__(self, max_tokens:int=10000) -> None:
        """
        Initialize the Pipeline class with placeholders for datasets, models, and evaluation results.

        :return: None
        """
        self.train = None
        self.dev = None
        self.test = None
        self.LSTM = None
        self.max_tokens = max_tokens

    def run(self) -> None:
        """
        Execute the entire machine learning pipeline, including data loading, preprocessing, model training, evaluation, and analysis of misclassified samples.

        :return: None
        """
        # Load data
        self.train, self.test = load_data()
        
        # Split dataset
        self.train, self.dev = split_dataset(self.train)
        print(f"Loaded data for pipeline {self.max_tokens}")
        
        # Preprocess data
        self.train = preprocess_data(self.train)
        self.dev = preprocess_data(self.dev)
        self.test = preprocess_data(self.test)

        # Maximum length of the output sequence after vectorization (padding/truncating)
        output_sequence_length = 128
        # Dimensionality of the embedding layer
        embed_dim = 64

        # Number of training epochs for both models
        epochs = 30

        
        # Feature engineering using TF Text Vectorization
        self.X_train, vocab = feature_engineering(self.train, column_name="description", max_tokens=self.max_tokens, output_sequence_length=output_sequence_length)
         # Convert from 1-indexed to 0-indexed
        self.y_train = self.train['label'].values - 1

        self.X_dev, _ = feature_engineering(self.dev, column_name="description", max_tokens=self.max_tokens, output_sequence_length=output_sequence_length, vocab=vocab)
        self.y_dev = self.dev['label'].values - 1

        self.X_test, _ = feature_engineering(self.test, column_name="description", max_tokens=self.max_tokens, output_sequence_length=output_sequence_length, vocab=vocab)
        self.y_test = self.test['label'].values - 1

        # Train LSTM model with larger batch size for faster training, using dev set for validation
        self.LSTM, self.LSTM_history = train_model('lstm', self.X_train, self.y_train, X_val=self.X_dev, y_val=self.y_dev, vocab_size=self.max_tokens, embed_dim=embed_dim, epochs=epochs, batch_size=256)

        # Evaluate models on the test set
        self.LSTM_predictions, self.LSTM_metrics = evaluate_model(self.LSTM, self.X_test, self.y_test)

        # Collect misclassified for both models for creation of error categories
        self.LSTM_misclassified = collect_misclassified_samples(self.LSTM, self.X_test, self.y_test, n_samples=10)

        self.predictions = {
            "LSTM": self.LSTM_predictions
        }
        
        for model_name, y_pred in self.predictions.items():
            plot_confusion_matrix(
                self.y_test, 
                y_pred, 
                f"Confusion Matrix – {model_name}, Max Length={self.max_tokens}"
            )
        
        # Plot learning curves for both models
        plot_learning_curves(
            {"LSTM": self.LSTM_history},
            title=f"Learning Curves – max_tokens={self.max_tokens}", 
            max_tokens=self.max_tokens
        )

if __name__ == "__main__":
    # Instantiate and run the machine learning pipeline for AG News classification with max_tokens 1000
    pipeline = Pipeline()
    pipeline.run()

    # Print evaluation metrics for both models and save them to JSON files, along with the misclassified samples for further analysis.
    print("LSTM Metrics:", pipeline.LSTM_metrics)

    # Save metrics and misclassified samples to files for further analysis and reporting.

    with open('results/lstm_metrics.json', 'w') as f:
        json.dump(pipeline.LSTM_metrics, f, indent=4)
    
    with open('results/lstm_history.json', 'w') as f:
        json.dump(pipeline.LSTM_history, f, indent=4)

    # Save misclassified samples for the best performing model and both models for error analysis.
    pipeline.LSTM_misclassified.to_csv('results/LSTM_misclassified.csv', index=False)

    
