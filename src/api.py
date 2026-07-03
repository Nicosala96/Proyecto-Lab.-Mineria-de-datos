import os
import time
import joblib
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Rutas a los artefactos ───────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(BASE_DIR, "..", "models"))
MODEL_VERSION = "1.1.0"

# ── Carga de artefactos al arrancar ──────────────────────────────────────────
logger.info("Cargando artefactos desde %s", MODELS_DIR)
modelo       = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.joblib"))
scaler       = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
enc_contract = joblib.load(os.path.join(MODELS_DIR, "encoder_contract.joblib"))
enc_payment  = joblib.load(os.path.join(MODELS_DIR, "encoder_payment.joblib"))
enc_internet = joblib.load(os.path.join(MODELS_DIR, "encoder_internet.joblib"))
enc_region   = joblib.load(os.path.join(MODELS_DIR, "encoder_region.joblib"))
logger.info("Artefactos cargados correctamente")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AndesLink – API de Predicción de Churn",
    description="Predice si un cliente abandonará el servicio (churn = 1) o no (churn = 0).",
    version=MODEL_VERSION,
)

# ── Prometheus — métricas automáticas de HTTP ────────────────────────────────
Instrumentator().instrument(app).expose(app)

# ── Prometheus — métricas de negocio ─────────────────────────────────────────
PREDICCIONES_TOTAL = Counter(
    "churn_predicciones_total",
    "Total de predicciones realizadas",
    ["resultado"]  # labels: churn o no_churn
)

PROBABILIDAD_HISTOGRAM = Histogram(
    "churn_probabilidad",
    "Distribución de probabilidades de churn predichas",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

LATENCIA_INFERENCIA = Histogram(
    "churn_inferencia_segundos",
    "Tiempo de inferencia del modelo en segundos",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

# ── Esquema de entrada ────────────────────────────────────────────────────────
class ClienteInput(BaseModel):
    tenure_months:        int   = Field(..., ge=0,  description="Meses de antigüedad del cliente")
    monthly_charge:       float = Field(..., ge=0,  description="Cargo mensual en USD")
    support_tickets:      int   = Field(..., ge=0,  description="Tickets de soporte abiertos")
    late_payments:        int   = Field(..., ge=0,  description="Cantidad de pagos tardíos")
    avg_monthly_usage_gb: float = Field(..., ge=0,  description="Uso promedio mensual en GB")
    customer_age:         int   = Field(..., ge=18, description="Edad del cliente")
    contract_type:    Literal["mensual", "anual", "bianual"]                    = Field(..., description="Tipo de contrato")
    payment_method:   Literal["transferencia", "debito", "efectivo", "credito"] = Field(..., description="Método de pago")
    internet_service: Literal["cable", "fibra", "movil", "ninguno"]             = Field(..., description="Tipo de servicio de internet")
    region:           Literal["centro", "norte", "oeste", "sur"]                = Field(..., description="Región geográfica")

    model_config = {
        "json_schema_extra": {
            "example": {
                "tenure_months": 24,
                "monthly_charge": 85.0,
                "support_tickets": 2,
                "late_payments": 0,
                "avg_monthly_usage_gb": 40.0,
                "customer_age": 35,
                "contract_type": "anual",
                "payment_method": "debito",
                "internet_service": "fibra",
                "region": "centro"
            }
        }
    }

# ── Esquema de salida ─────────────────────────────────────────────────────────
class PrediccionOutput(BaseModel):
    churn:        int   = Field(..., description="0 = el cliente se queda, 1 = el cliente abandona")
    probabilidad: float = Field(..., description="Probabilidad de churn entre 0 y 1")
    etiqueta:     str   = Field(..., description="Descripción legible del resultado")
    model_version: str  = Field(..., description="Versión del modelo utilizado")

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Estado"])
def root():
    return {"estado": "ok", "mensaje": "API de churn operativa", "version": MODEL_VERSION}


@app.get("/health", tags=["Estado"])
def health():
    """Valida que el modelo y todos los artefactos están cargados en memoria."""
    try:
        artefactos = {
            "modelo": modelo is not None,
            "scaler": scaler is not None,
            "enc_contract": enc_contract is not None,
            "enc_payment": enc_payment is not None,
            "enc_internet": enc_internet is not None,
            "enc_region": enc_region is not None,
        }
        todos_ok = all(artefactos.values())
        return {
            "estado": "ok" if todos_ok else "degradado",
            "model_version": MODEL_VERSION,
            "artefactos": artefactos
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Servicio no disponible: {str(e)}")


@app.post("/predecir", response_model=PrediccionOutput, tags=["Predicción"])
def predecir(cliente: ClienteInput):
    logger.info("Request recibido: %s", cliente.model_dump())
    try:
        # 1. Encoding de variables categóricas
        contract = enc_contract.transform([[cliente.contract_type]])[0][0]
        payment  = enc_payment.transform([[cliente.payment_method]])[0][0]
        internet = enc_internet.transform([[cliente.internet_service]])[0][0]
        region   = enc_region.transform([[cliente.region]])[0][0]

        # 2. Armar DataFrame con las 10 variables en el orden exacto del entrenamiento
        columnas_num = ['tenure_months', 'monthly_charge', 'support_tickets',
                        'late_payments', 'avg_monthly_usage_gb', 'customer_age']

        fila = pd.DataFrame([{
            'tenure_months':        cliente.tenure_months,
            'monthly_charge':       cliente.monthly_charge,
            'support_tickets':      cliente.support_tickets,
            'late_payments':        cliente.late_payments,
            'avg_monthly_usage_gb': cliente.avg_monthly_usage_gb,
            'contract_type':        contract,
            'customer_age':         cliente.customer_age,
            'payment_method':       payment,
            'internet_service':     internet,
            'region':               region,
        }])

        # 3. Escalado y predicción con medición de latencia
        fila[columnas_num] = scaler.transform(fila[columnas_num])

        inicio = time.time()
        pred   = int(modelo.predict(fila)[0])
        prob   = float(modelo.predict_proba(fila)[0][1])
        latencia = time.time() - inicio

        # 4. Registrar métricas de negocio
        resultado_label = "churn" if pred == 1 else "no_churn"
        PREDICCIONES_TOTAL.labels(resultado=resultado_label).inc()
        PROBABILIDAD_HISTOGRAM.observe(prob)
        LATENCIA_INFERENCIA.observe(latencia)

        etiqueta = "El cliente probablemente abandonará el servicio" if pred == 1 \
                   else "El cliente probablemente permanecerá activo"

        logger.info("Prediccion: churn=%d, probabilidad=%.4f, latencia=%.4fs", pred, prob, latencia)

        return PrediccionOutput(
            churn=pred,
            probabilidad=round(prob, 4),
            etiqueta=etiqueta,
            model_version=MODEL_VERSION
        )

    except Exception as e:
        logger.error("Error en prediccion: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")