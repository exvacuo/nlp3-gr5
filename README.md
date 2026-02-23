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
This will load the data, preprocess them, used TF-IDF vectorizer to engineer features, train baseline models (Logistic Regression & Linear SVM) and evaluate them (using Accuracy, Macro F1 and Confusion Matrices). The program will select the best performing model and save the first 10 misclassified predictions of this model for error analysis into the results folder. Additionally, we save .json files with all metrics, first 20 misclassified examples for both LR and SVM models for error category creation and plotted confusion matrices for each model in the results folder. 
