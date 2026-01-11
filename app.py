import os
import json
from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import logging


# =========================
# Config
# =========================
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_NAME = os.getenv("DB_NAME", "ternaklele")
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "")

MODEL_DIR = os.getenv("MODEL_DIR", "./models")
MODEL_PATH = os.path.join(MODEL_DIR, "linear_regression_profit.joblib")
META_PATH = os.path.join(MODEL_DIR, "linear_regression_profit.meta.json")

os.makedirs(MODEL_DIR, exist_ok=True)

MYSQL_URL = (
    f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    "?charset=utf8mb4"
)

engine = create_engine(MYSQL_URL, pool_pre_ping=True, pool_recycle=3600)

# Features used for training (keep stable!)
FEATURE_COLS = [
    "duration_days",
    "seed_count",
    "feed_cost",
    "other_cost",
    "last_avg_weight",
    "last_death_count",
    "pond_size",
]

TARGET_COL = "profit"


# =========================
# FastAPI schema
# =========================
class TrainResponse(BaseModel):
    model_version: str
    trained_at: str
    rows_used: int
    metrics: Dict[str, float]
    features: list[str]


class PredictRequest(BaseModel):
    cycle_id: int = Field(..., gt=0)


class PredictResponse(BaseModel):
    cycle_id: int
    model_version: str
    predicted_profit: float
    features_used: Dict[str, Any]


app = FastAPI(title="Lele Profit ML API", version="1.0.0")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lele_ml_api")


# =========================
# DB helpers
# =========================
def load_training_dataset(min_rows: int = 10) -> pd.DataFrame:
    """
    Pull dataset from v_profit_dataset.
    Require profit not null to train.
    """
    sql = text("""
        SELECT
            v.cycle_id,
            v.duration_days,
            v.seed_count,
            v.feed_cost,
            v.other_cost,
            v.last_avg_weight,
            v.last_death_count,
            p.ukuran AS pond_size,
            v.profit
        FROM v_profit_dataset v
        JOIN ponds p ON p.id = v.pond_id
        WHERE v.profit IS NOT NULL
          AND v.duration_days IS NOT NULL
    """)
    df = pd.read_sql(sql, engine)

    # basic sanity
    df = df.dropna(subset=["cycle_id"])
    if len(df) < min_rows:
        raise ValueError(
            f"Not enough training rows: {len(df)}. "
            f"Need at least {min_rows} cycles with profit filled."
        )
    return df


def load_features_for_cycle(cycle_id: int) -> pd.DataFrame:
    """
    Load one row of feature vector for given cycle_id.
    """
    sql = text("""
        SELECT
            v.cycle_id,
            v.duration_days,
            v.seed_count,
            v.feed_cost,
            v.other_cost,
            v.last_avg_weight,
            v.last_death_count,
            p.ukuran AS pond_size
        FROM v_profit_dataset v
        JOIN ponds p ON p.id = v.pond_id
        WHERE v.cycle_id = :cycle_id
        LIMIT 1
    """)
    df = pd.read_sql(sql, engine, params={"cycle_id": cycle_id})
    if df.empty:
        raise ValueError(f"cycle_id={cycle_id} not found in v_profit_dataset.")
    return df


# =========================
# ML helpers
# =========================
def build_pipeline() -> Pipeline:
    """
    Simple Linear Regression with numeric imputation.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, FEATURE_COLS),
        ],
        remainder="drop"
    )

    model = LinearRegression()

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])
    return pipe


def evaluate(y_true, y_pred) -> Dict[str, float]:
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    # Older sklearn versions don't support squared=False; compute manually for compatibility.
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(mse ** 0.5)
    return {"r2": float(r2), "mae": float(mae), "rmse": float(rmse)}


def save_model(pipe: Pipeline, meta: dict) -> None:
    joblib.dump(pipe, MODEL_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_model() -> tuple[Pipeline, dict]:
    if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
        raise FileNotFoundError("Model not found. Train first via POST /train.")
    pipe = joblib.load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return pipe, meta


# =========================
# API endpoints
# =========================
@app.get("/health")
def health():
    return {"ok": True, "time": datetime.utcnow().isoformat()}


@app.post("/train", response_model=TrainResponse)
def train():
    df = load_training_dataset(min_rows=10)

    # Training set
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    # Split for metrics
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    metrics = evaluate(y_test, preds)

    model_version = f"lr_profit_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    meta = {
        "model_version": model_version,
        "trained_at": datetime.utcnow().isoformat(),
        "rows_used": int(len(df)),
        "features": FEATURE_COLS,
        "target": TARGET_COL,
        "metrics": metrics,
    }

    save_model(pipe, meta)

    return TrainResponse(
        model_version=model_version,
        trained_at=meta["trained_at"],
        rows_used=meta["rows_used"],
        metrics=metrics,
        features=FEATURE_COLS,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        pipe, meta = load_model()
    except FileNotFoundError as e:
        logger.warning("Predict failed: model not found. detail=%s", e)
        raise HTTPException(status_code=400, detail=str(e))

    try:
        feat_df = load_features_for_cycle(req.cycle_id)
    except ValueError as e:
        logger.warning("Predict failed: features missing for cycle_id=%s detail=%s", req.cycle_id, e)
        raise HTTPException(status_code=404, detail=str(e))

    # Predict
    X = feat_df[FEATURE_COLS].copy()
    yhat = float(pipe.predict(X)[0])

    features_used = {col: (None if pd.isna(feat_df.iloc[0][col]) else float(feat_df.iloc[0][col]))
                     for col in FEATURE_COLS}

    logger.info(
        "Predict success cycle_id=%s model_version=%s predicted_profit=%.4f features_used=%s",
        req.cycle_id,
        meta.get("model_version", "unknown"),
        yhat,
        features_used,
    )

    return PredictResponse(
        cycle_id=req.cycle_id,
        model_version=meta.get("model_version", "unknown"),
        predicted_profit=yhat,
        features_used=features_used,
    )
