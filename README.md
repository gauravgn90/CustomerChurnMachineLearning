# Customer Churn Prediction

Telco customer churn end-to-end ML solution - EDA, Decision Tree
model, aur REST API.

## Setup

```bash
git clone https://github.com/gauravgn90/CustomerChurnMachineLearning.git
cd CustomerChurnMachineLearning
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run Notebook 

```bash
jupyter lab notebook/churn_analysis.ipynb
```
Execute all cells. Model will be saved in `model/churn_model.pkl`.

## Run API

```bash
uvicorn app:app --reload
```
Docs: http://127.0.0.1:8000/docs

## Sample request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

Response:
```json
{"prediction": "Yes", "churn_probability": 0.5583}
```

## Model

Decision Tree - `max_depth=5`, `min_samples_leaf=50`, `random_state=42`.
70:30 train-test split.

| Metric | Score |
|---|---|
| Accuracy | 0.794 |
| Precision | 0.615 |
| Recall | 0.602 |
| F1 | 0.609 |
| ROC-AUC | 0.828 |

Top drivers: Contract type (52%), tenure (18%), Fiber optic internet (15%).

## Structure

```
customer_churn_project/
├── data/                   # dataset + data dictionary
├── notebook/
│   └── churn_analysis.ipynb
├── model/
│   ├── churn_model.pkl
│   └── decision_tree.png
├── features.py             # shared feature engineering
├── app.py                  # FastAPI service
├── requirements.txt
├── sample_request.json
└── README.md
```