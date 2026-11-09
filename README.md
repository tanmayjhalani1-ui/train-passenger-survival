<div align="center">
  <img src="assets/portfolio-banner.svg" alt="Passenger Survival Prediction — Tanmay Jhalani" width="100%" />
</div>

<div align="center">
  <strong>Designed and engineered by Tanmay Jhalani</strong> · <a href="https://github.com/tanmayjhalani1-ui">GitHub Profile</a>
</div>

---

# Passenger Survival Prediction

An end-to-end machine learning project that extends the original passenger survival notebook into a reproducible training workflow, experiment tracking, interactive dashboard, and lightweight prediction API.

> **Dataset limitation:** The included CSV contains only 70 rows. It is suitable for demonstrating an ML engineering workflow, not for establishing reliable real-world survival predictions. Metrics can vary substantially with such a small sample. This project does not claim production-grade predictive validity.

## Project capabilities

- Data validation and leakage-conscious preprocessing with scikit-learn pipelines.
- Comparison of Logistic Regression, Random Forest, and Gradient Boosting.
- Stratified holdout evaluation and five-fold cross-validation using ROC-AUC.
- Accuracy, precision, recall, F1, ROC-AUC, classification report, and confusion matrix.
- MLflow experiment tracking for parameters, metrics, metadata, and the fitted model.
- Streamlit dashboard for interactive predictions and model-run summaries.
- Simple FastAPI endpoints for health checks and predictions.
- Docker and Docker Compose for running the dashboard and API.
- GitHub Actions workflow to install dependencies, compile modules, and run the training workflow.

## Repository structure

```text
.
├── api/
│   └── main.py
├── artifacts/                 # Generated after training; not committed
├── mlruns/                     # Local MLflow tracking store; not committed
├── src/
│   ├── inference.py
│   └── train.py
├── .github/workflows/ci.yml
├── app.py                      # Streamlit dashboard
├── crash.ipynb                 # Original notebook, retained
├── train_passenger_survival.csv
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Quick start

Python 3.11 is recommended.

```bash
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install dependencies and train:

```bash
pip install -r requirements.txt
python -m src.train
```

Training writes the selected pipeline and metadata to `artifacts/` and logs the run to the local MLflow store in `mlruns/`.

Open the MLflow UI in another terminal:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Then visit `http://127.0.0.1:5000`.

## Run the applications

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

Start the FastAPI service:

```bash
uvicorn api.main:app --reload
```

API documentation is available at `http://127.0.0.1:8000/docs`.

### API example

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"age": 28, "fare": 45.5, "gender": "Female", "seat_class": "Second"}'
```

The response contains the predicted class, readable label, and estimated survival probability. The API validates input ranges and category values.

## Run with Docker

Build and start both services:

```bash
docker compose up --build
```

- Streamlit: `http://localhost:8501`
- FastAPI: `http://localhost:8000/docs`

Train the model before starting the services. The generated `artifacts/` directory is mounted into the containers.

## Modeling approach

The target is `Survived`. The model uses `Age`, `Fare`, `Gender`, and `SeatClass`. `PassengerID` is excluded because it is an identifier rather than a meaningful passenger attribute.

Numeric missing values are median-imputed and scaled. Categorical missing values are mode-imputed and one-hot encoded. All preprocessing is fitted inside the scikit-learn pipeline using training folds only, preventing preprocessing leakage into validation data.

The model with the highest mean five-fold cross-validation ROC-AUC on the training split is selected and evaluated once on a held-out test split. The fixed random seed supports reproducibility; it does not remove uncertainty caused by the tiny dataset.

## MLflow tracking

Each training execution records:
- Selected estimator and run parameters.
- Cross-validation ROC-AUC for each candidate.
- Holdout evaluation metrics.
- Training metadata and the fitted scikit-learn pipeline.

The local tracking directory is excluded from Git because it contains generated run data and artifacts.

## Limitations and responsible interpretation

- Only 70 records are available; estimates are statistically unstable.
- The dataset's provenance and representativeness are not established here.
- The target labels and feature relationships should not be interpreted as causal.
- This project is an engineering demonstration and must not be used for real safety, transport, or emergency decisions.
- A larger, documented, representative dataset and external validation would be required before making real-world claims.

## Original work

The original `crash.ipynb` notebook and `train_passenger_survival.csv` dataset are retained. The new application workflow is added alongside them so the initial exploration remains available.
