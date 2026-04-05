## Project Setup

**Clone the project:**
```bash
git clone https://github.com/nikulkaaa/nlp1-gr5.git
cd nlp1-gr5
```

**Setup with uv:**
```bash
uv sync
```

**Activate virtual environment:**

macOS/Linux/WSL:
```bash
source .venv/bin/activate
```

Windows:
```powershell
.\.venv\Scripts\activate
```

## One Command Run
To run the pipeline of this project, run:
```powershell
python main.py
```
This will load AG News classification data, preprocess them with different input types (headline only or headline + description), train LSTM and Transformer (DistilBERT) models for text classification, and evaluate them using Accuracy, Macro F1, and Confusion Matrices. The program will collect the first 10 misclassified predictions from both models for error analysis into the results folder. We save .json files with all metrics, learning curves for each model, and conduct robustness evaluations including input stress tests (comparing headline-only vs. headline+description inputs) and label noise sensitivity tests (training on 25%, 50%, 75%, and 100% of the training data) to assess model performance through different conditions.
