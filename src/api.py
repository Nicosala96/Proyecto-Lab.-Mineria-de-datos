import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

# ── Rutas a los artefactos ──────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "..", "models")

# ── Carga de artefactos al arrancar ─────────────────────────────────────────
modelo          = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.joblib"))
scaler          = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
enc_contract    = joblib.load(os.path.join(MODELS_DIR, "encoder_contract.joblib"))
enc_payment     = joblib.load(os.path.join(MODELS_DIR, "encoder_payment.joblib"))
enc_internet    = joblib.load(os.path.join(MODELS_DIR, "encoder_internet.joblib"))
enc_region      = joblib.load(os.path.join(MODELS_DIR, "encoder_region.joblib"))

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AndesLink – API de Predicción de Churn",
    description="Predice si un cliente abandonará el servicio (churn = 1) o no (churn = 0).",
    version="1.0.0",
)

# ── Esquema de entrada ───────────────────────────────────────────────────────
class ClienteInput(BaseModel):
    tenure_months:        int   = Field(..., ge=0,  description="Meses de antigüedad del cliente")
    monthly_charge:       float = Field(..., ge=0,  description="Cargo mensual en USD")
    support_tickets:      int   = Field(..., ge=0,  description="Tickets de soporte abiertos")
    late_payments:        int   = Field(..., ge=0,  description="Cantidad de pagos tardíos")
    avg_monthly_usage_gb: float = Field(..., ge=0,  description="Uso promedio mensual en GB")
    customer_age:         int   = Field(..., ge=18, description="Edad del cliente")
    contract_type:        Literal["mensual", "anual", "bianual"]                        = Field(..., description="Tipo de contrato")
    payment_method:       Literal["transferencia", "debito", "efectivo", "credito"]     = Field(..., description="Método de pago")
    internet_service:     Literal["cable", "fibra", "movil", "ninguno"]                 = Field(..., description="Tipo de servicio de internet")
    region:               Literal["centro", "norte", "oeste", "sur"]                    = Field(..., description="Región geográfica")

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

# ── Esquema de salida ────────────────────────────────────────────────────────
class PrediccionOutput(BaseModel):
    churn:       int   = Field(..., description="0 = el cliente se queda, 1 = el cliente abandona")
    probabilidad: float = Field(..., description="Probabilidad de churn entre 0 y 1")
    etiqueta:    str   = Field(..., description="Descripción legible del resultado")

# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/", tags=["Estado"])
def health_check():
    return {"estado": "ok", "mensaje": "API de churn operativa"}


@app.post("/predecir", response_model=PrediccionOutput, tags=["Predicción"])
def predecir(cliente: ClienteInput):
    try:
        # 1. Encoding de variables categóricas
        contract  = enc_contract.transform([[cliente.contract_type]])[0][0]
        payment   = enc_payment.transform([[cliente.payment_method]])[0][0]
        internet  = enc_internet.transform([[cliente.internet_service]])[0][0]
        region    = enc_region.transform([[cliente.region]])[0][0]

        # 2. Armar DataFrame con el orden exacto que espera el modelo
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
        }])

        # 3. Escalado de variables numéricas
        fila[columnas_num] = scaler.transform(fila[columnas_num])

        # 4. Predicción
        pred      = int(modelo.predict(fila)[0])
        prob      = float(modelo.predict_proba(fila)[0][1])
        etiqueta  = "El cliente probablemente abandonará el servicio" if pred == 1 \
                    else "El cliente probablemente permanecerá activo"

        return PrediccionOutput(churn=pred, probabilidad=round(prob, 4), etiqueta=etiqueta)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")
