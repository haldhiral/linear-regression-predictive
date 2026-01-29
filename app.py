import os
import json
from datetime import datetime
from typing import Dict, Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import sklearn


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
    "feed_qty_raw",
    "other_cost",
    "last_avg_weight",
    "last_death_count",
    "last_remaining_count",
    "pond_size",
    "mortality_rate",
    "survival_rate",
    "feed_cost_per_seed",
    "other_cost_per_seed",
    "feed_qty_per_seed",
    "seed_density",
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
    Require completed cycles (end_date set, revenue > 0) to train.
    """
    sql = text("""
        SELECT
            v.cycle_id,
            v.end_date,
            v.duration_days,
            v.seed_count,
            v.feed_cost,
            v.feed_qty_raw,
            v.other_cost,
            v.last_avg_weight,
            v.last_death_count,
            v.last_remaining_count,
            p.ukuran AS pond_size,
            v.revenue,
            v.profit
        FROM v_profit_dataset v
        JOIN ponds p ON p.id = v.pond_id
        WHERE v.profit IS NOT NULL
          AND v.end_date IS NOT NULL
          AND v.revenue > 0
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
            v.feed_qty_raw,
            v.other_cost,
            v.last_avg_weight,
            v.last_death_count,
            v.last_remaining_count,
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
def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, pd.NA)
    return numerator / denominator


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix with derived ratios.
    """
    feats = df.copy()
    feats["mortality_rate"] = safe_divide(feats["last_death_count"], feats["seed_count"])
    feats["survival_rate"] = safe_divide(feats["last_remaining_count"], feats["seed_count"])
    feats["feed_cost_per_seed"] = safe_divide(feats["feed_cost"], feats["seed_count"])
    feats["other_cost_per_seed"] = safe_divide(feats["other_cost"], feats["seed_count"])
    feats["feed_qty_per_seed"] = safe_divide(feats["feed_qty_raw"], feats["seed_count"])
    feats["seed_density"] = safe_divide(feats["seed_count"], feats["pond_size"])
    return feats[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")


def log_model_environment(meta: dict | None = None) -> None:
    logger.info("Model environment sklearn=%s pandas=%s", sklearn.__version__, pd.__version__)
    if not meta:
        return
    meta_sklearn = meta.get("sklearn_version")
    if meta_sklearn and meta_sklearn != sklearn.__version__:
        logger.warning(
            "Model sklearn_version mismatch model=%s runtime=%s",
            meta_sklearn,
            sklearn.__version__,
        )
    meta_features = meta.get("features")
    if meta_features and meta_features != FEATURE_COLS:
        logger.warning(
            "Model features differ from code features model=%s code=%s",
            meta_features,
            FEATURE_COLS,
        )


def collect_predict_warnings(raw_row: pd.Series, derived_row: pd.Series) -> list[str]:
    warnings: list[str] = []
    seed_count = raw_row.get("seed_count")
    pond_size = raw_row.get("pond_size")
    last_remaining = raw_row.get("last_remaining_count")
    last_death = raw_row.get("last_death_count")

    if pd.isna(seed_count) or seed_count <= 0:
        warnings.append("seed_count<=0")
    if pd.isna(pond_size) or pond_size <= 0:
        warnings.append("pond_size<=0")
    if not pd.isna(last_remaining) and not pd.isna(seed_count) and last_remaining > seed_count:
        warnings.append("last_remaining_count>seed_count")
    if not pd.isna(last_death) and not pd.isna(seed_count) and last_death > seed_count:
        warnings.append("last_death_count>seed_count")

    for rate_name in ["mortality_rate", "survival_rate"]:
        rate = derived_row.get(rate_name)
        if not pd.isna(rate) and (rate < 0 or rate > 1):
            warnings.append(f"{rate_name} out_of_range")

    return warnings


def build_pipeline() -> Pipeline:
    """
    Regularized Linear Regression with numeric imputation.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, FEATURE_COLS),
        ],
        remainder="drop"
    )

    model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])

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
    log_model_environment(meta)
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
    df = df.sort_values("end_date")

    # Training set
    X = prepare_features(df)
    y = df[TARGET_COL].copy()

    logger.info(
        "Training dataset rows=%s end_date_range=%s..%s",
        len(df),
        df["end_date"].min(),
        df["end_date"].max(),
    )
    logger.info("Training feature_missing=%s", X.isna().sum().to_dict())

    # Time-aware split for metrics
    test_size = max(2, int(len(df) * 0.2))
    train_size = len(df) - test_size
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    metrics = evaluate(y_test, preds)
    selected_alpha = getattr(pipe.named_steps["model"], "alpha_", None)
    if selected_alpha is not None:
        logger.info("Training selected_alpha=%s", selected_alpha)

    model_version = f"lr_profit_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    meta = {
        "model_version": model_version,
        "trained_at": datetime.utcnow().isoformat(),
        "rows_used": int(len(df)),
        "features": FEATURE_COLS,
        "target": TARGET_COL,
        "metrics": metrics,
        "sklearn_version": sklearn.__version__,
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
    X = prepare_features(feat_df)
    yhat = float(pipe.predict(X)[0])

    raw_features = feat_df.iloc[0].to_dict()
    derived_row = X.iloc[0]
    features_used = {
        col: (None if pd.isna(derived_row[col]) else float(derived_row[col]))
        for col in FEATURE_COLS
    }
    logger.info("Predict input raw cycle_id=%s raw_features=%s", req.cycle_id, raw_features)

    predict_warnings = collect_predict_warnings(feat_df.iloc[0], derived_row)
    if predict_warnings:
        logger.warning(
            "Predict input anomalies cycle_id=%s issues=%s",
            req.cycle_id,
            predict_warnings,
        )
    missing_features = derived_row.isna()
    if missing_features.any():
        logger.warning(
            "Predict input missing derived features cycle_id=%s missing=%s",
            req.cycle_id,
            list(missing_features[missing_features].index),
        )

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
