import pytest
from fastapi.testclient import TestClient
import sys
import os

# Apuntar los modelos a la carpeta correcta antes de importar la app
os.environ["MODELS_DIR"] = os.path.join(os.path.dirname(__file__), "..", "models")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from api import app

client = TestClient(app)

# ── Datos de ejemplo válidos ─────────────────────────────────────────────────
CLIENTE_VALIDO = {
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

# ── Tests ────────────────────────────────────────────────────────────────────

def test_root():
    """El endpoint raíz debe responder 200 y confirmar que la API está operativa."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["estado"] == "ok"


def test_health_artefactos_cargados():
    """El endpoint /health debe confirmar que todos los artefactos están cargados."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["estado"] == "ok"
    assert data["artefactos"]["modelo"] is True
    assert data["artefactos"]["scaler"] is True
    assert data["artefactos"]["enc_contract"] is True
    assert data["artefactos"]["enc_payment"] is True
    assert data["artefactos"]["enc_internet"] is True
    assert data["artefactos"]["enc_region"] is True


def test_health_version_modelo():
    """El endpoint /health debe informar la versión del modelo."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "model_version" in response.json()


def test_prediccion_valida():
    """Con datos válidos debe devolver churn 0 o 1 y probabilidad entre 0 y 1."""
    response = client.post("/predecir", json=CLIENTE_VALIDO)
    assert response.status_code == 200
    data = response.json()
    assert data["churn"] in [0, 1]
    assert 0.0 <= data["probabilidad"] <= 1.0
    assert isinstance(data["etiqueta"], str)


def test_prediccion_incluye_version_modelo():
    """La respuesta de /predecir debe incluir la versión del modelo."""
    response = client.post("/predecir", json=CLIENTE_VALIDO)
    assert response.status_code == 200
    assert "model_version" in response.json()


def test_prediccion_contrato_invalido():
    """Un valor de contrato no permitido debe devolver error 422."""
    cliente_malo = CLIENTE_VALIDO.copy()
    cliente_malo["contract_type"] = "trimestral"
    response = client.post("/predecir", json=cliente_malo)
    assert response.status_code == 422


def test_prediccion_campo_faltante():
    """Si falta un campo obligatorio debe devolver error 422."""
    cliente_incompleto = CLIENTE_VALIDO.copy()
    del cliente_incompleto["monthly_charge"]
    response = client.post("/predecir", json=cliente_incompleto)
    assert response.status_code == 422


def test_prediccion_edad_invalida():
    """Una edad menor a 18 debe devolver error 422."""
    cliente_malo = CLIENTE_VALIDO.copy()
    cliente_malo["customer_age"] = 10
    response = client.post("/predecir", json=cliente_malo)
    assert response.status_code == 422


def test_prediccion_valores_negativos():
    """Valores negativos en campos numéricos deben devolver error 422."""
    cliente_malo = CLIENTE_VALIDO.copy()
    cliente_malo["monthly_charge"] = -50.0
    response = client.post("/predecir", json=cliente_malo)
    assert response.status_code == 422


def test_prediccion_payment_method_invalido():
    """Un método de pago no permitido debe devolver error 422."""
    cliente_malo = CLIENTE_VALIDO.copy()
    cliente_malo["payment_method"] = "criptomoneda"
    response = client.post("/predecir", json=cliente_malo)
    assert response.status_code == 422


def test_prediccion_internet_service_invalido():
    """Un servicio de internet no permitido debe devolver error 422."""
    cliente_malo = CLIENTE_VALIDO.copy()
    cliente_malo["internet_service"] = "satelite"
    response = client.post("/predecir", json=cliente_malo)
    assert response.status_code == 422


def test_prediccion_region_invalida():
    """Una región no permitida debe devolver error 422."""
    cliente_malo = CLIENTE_VALIDO.copy()
    cliente_malo["region"] = "patagonia"
    response = client.post("/predecir", json=cliente_malo)
    assert response.status_code == 422