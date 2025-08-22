from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import logging
import time
import json
import sys
import os
import psutil

# OpenTelemetry imports for tracing
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

# --------------------------------------------------------------------------
# 1. Custom JSON Formatter for Google Cloud Logging
# --------------------------------------------------------------------------
class JsonGCPFormatter(logging.Formatter):
    """
    A custom formatter to produce structured logs in a format
    that Google Cloud Logging can parse effectively.
    """
    def format(self, record):
        log_object = {
            "timestamp": self.formatTime(record, self.datefmt),
            "severity": record.levelname,
            "message": record.getMessage(),
            "name": record.name
        }
        if isinstance(record.msg, dict):
            log_object.update(record.msg)
            if 'message' in record.msg:
                log_object['message'] = record.msg['message']
        if record.exc_info:
            log_object['exc_info'] = self.formatException(record.exc_info)
        return json.dumps(log_object)

# --------------------------------------------------------------------------
# 2. Setup Logging and Tracing
# --------------------------------------------------------------------------

# Setup Tracer for Google Cloud Trace
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(CloudTraceSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# Setup structured logging
logger = logging.getLogger("fraud-detection-api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonGCPFormatter())
logger.addHandler(handler)
logger.propagate = False

logger.info({"event": "application_initializing", "status": "starting"})

# --------------------------------------------------------------------------
# 3. Load Your Trained Model
# --------------------------------------------------------------------------

try:
    # This line specifically loads your trained model from the file.
    model = joblib.load("new_model.joblib")
    logger.info({
        "event": "model_load_success",
        "model_path": "model.joblib",
        "model_type": str(type(model))
    })
except Exception as e:
    logger.exception({"event": "model_load_failure", "error": str(e)})
    # If the model fails to load, the application cannot serve predictions.
    sys.exit(1)

# Define the human-readable labels for the prediction output
CLASS_LABELS = {0: "Not Fraud", 1: "Fraud"}

# --------------------------------------------------------------------------
# 4. Initialize FastAPI App and Define Data Schemas
# --------------------------------------------------------------------------

app = FastAPI(
    title="💳 Fraud Detection API",
    description="Predicts fraudulent transactions using the trained model.",
    version="1.0.1"
)

# This Pydantic 'BaseModel' defines the structure for API *requests*.
# It is NOT a machine learning model. It's used for data validation.
class TransactionFeatures(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

    class Config:
        # Example data for API documentation
        schema_extra = {
            "example": {
                "V1": -1.35, "V2": -0.07, "V3": 2.53, "V4": 1.37, "V5": -0.33,
                "V6": 0.46, "V7": 0.23, "V8": 0.09, "V9": 0.36, "V10": 0.09,
                "V11": -0.55, "V12": -0.61, "V13": -0.99, "V14": -0.31, "V15": 1.46,
                "V16": -0.47, "V17": 0.20, "V18": 0.02, "V19": 0.40, "V20": 0.25,
                "V21": -0.01, "V22": 0.27, "V23": -0.11, "V24": 0.06, "V25": 0.12,
                "V26": -0.18, "V27": 0.13, "V28": -0.02, "Amount": 149.62
            }
        }

# This defines the structure for the API *response*.
class PredictionResponse(BaseModel):
    prediction: int
    predicted_class: str
    probability: float = None

# --------------------------------------------------------------------------
# 5. App State, Probes, and Middleware
# --------------------------------------------------------------------------

app_state = {
    "is_ready": False,
    "pod_name": os.getenv("HOSTNAME", "unknown-pod"),
}

@app.on_event("startup")
async def startup_event():
    time.sleep(2) # Simulate warm-up
    app_state["is_ready"] = True
    logger.info({"event": "app_startup_complete", "pod_name": app_state["pod_name"]})

@app.get("/live_check", tags=["Probes"])
async def liveness_probe():
    return {"status": "alive"}

@app.get("/ready_check", tags=["Probes"])
async def readiness_probe():
    if app_state["is_ready"]:
        return {"status": "ready"}
    return Response(status_code=503, content="Service not ready")

@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    start_time = time.time()
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x") if span.is_recording() else "N/A"
    response = await call_next(request)
    duration = round((time.time() - start_time) * 1000, 2)
    logger.info({
        "event": "http_request",
        "trace_id": trace_id,
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "duration_ms": duration,
    })
    response.headers["X-Process-Time-ms"] = str(duration)
    return response

# --------------------------------------------------------------------------
# 6. API Endpoints
# --------------------------------------------------------------------------

@app.get("/", tags=["Root"])
def read_root():
    return {"message": f"Welcome to the Fraud Detection API v{app.version}"}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request_data: TransactionFeatures):
    """
    Accepts transaction features and returns a fraud prediction
    using the pre-loaded model.
    """
    with tracer.start_as_current_span("prediction_endpoint") as span:
        trace_id = format(span.get_span_context().trace_id, "032x")
        try:
            # Convert the incoming request data into a pandas DataFrame
            input_df = pd.DataFrame([request_data.dict()])

            # This is the core step:
            # Using your loaded 'model' to make the prediction.
            prediction_result = model.predict(input_df)
            prediction_int = int(prediction_result[0])

            # Get probability if the model supports it
            confidence = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_df)[0]
                confidence = round(float(probs[prediction_int]), 4)

            # Map the integer prediction to its text label
            predicted_label = CLASS_LABELS.get(prediction_int, "Unknown")

            logger.info({
                "event": "prediction_success",
                "trace_id": trace_id,
                "prediction": predicted_label,
                "confidence": confidence
            })

            return {
                "prediction": prediction_int,
                "predicted_class": predicted_label,
                "probability": confidence
            }

        except Exception as e:
            logger.exception({
                "event": "prediction_error",
                "trace_id": trace_id,
                "error": str(e)
            })
            raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

