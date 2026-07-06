"""
Monitoreo de datos y modelo con Evidently
AndesLink Servicios Digitales S.A.
"""

import os
import numpy as np
import pandas as pd
import joblib
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.metrics import (
    DatasetDriftMetric,
    ColumnDriftMetric,
)

# ── Rutas ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "..", "data", "processed")
MODELS_DIR  = os.path.join(BASE_DIR, "..", "models")
REPORTS_DIR = os.path.join(BASE_DIR, "..", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Cargar artefactos ────────────────────────────────────────────────────────
print("Cargando datos y modelos...")
df = pd.read_csv(os.path.join(DATA_DIR, "churn_sintetico_EDA.csv"))

modelo       = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.joblib"))
scaler       = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
enc_contract = joblib.load(os.path.join(MODELS_DIR, "encoder_contract.joblib"))
enc_payment  = joblib.load(os.path.join(MODELS_DIR, "encoder_payment.joblib"))
enc_internet = joblib.load(os.path.join(MODELS_DIR, "encoder_internet.joblib"))
enc_region   = joblib.load(os.path.join(MODELS_DIR, "encoder_region.joblib"))

# ── Preparar dataset ─────────────────────────────────────────────────────────
print("Preparando dataset...")
df[['contract_type']]    = enc_contract.transform(df[['contract_type']])
df[['payment_method']]   = enc_payment.transform(df[['payment_method']])
df[['internet_service']] = enc_internet.transform(df[['internet_service']])
df[['region']]           = enc_region.transform(df[['region']])

columnas_X   = ['tenure_months', 'monthly_charge', 'support_tickets',
                'late_payments', 'avg_monthly_usage_gb', 'contract_type',
                'customer_age', 'payment_method', 'internet_service', 'region']
columnas_num = ['tenure_months', 'monthly_charge', 'support_tickets',
                'late_payments', 'avg_monthly_usage_gb', 'customer_age']

X = df[columnas_X].copy()
y = df['churn'].copy()
X[columnas_num] = scaler.transform(X[columnas_num])

# ── Dividir en referencia y actual ───────────────────────────────────────────
np.random.seed(42)
n        = len(X)
ref_size = int(n * 0.7)

X_ref = X.iloc[:ref_size].copy()
y_ref = y.iloc[:ref_size].copy()
X_cur = X.iloc[ref_size:].copy()
y_cur = y.iloc[ref_size:].copy()

# Simular drift en produccion
X_cur['monthly_charge']  = X_cur['monthly_charge']  + np.random.normal(0.3, 0.1, len(X_cur))
X_cur['support_tickets'] = X_cur['support_tickets']  + np.random.normal(0.2, 0.05, len(X_cur))

# ── Agregar predicciones ──────────────────────────────────────────────────────
ref_data               = X_ref.copy()
ref_data['target']     = y_ref.values
ref_data['prediction'] = modelo.predict(X_ref)

cur_data               = X_cur.copy()
cur_data['target']     = y_cur.values
cur_data['prediction'] = modelo.predict(X_cur)

# ── Reporte 1: Data Drift ────────────────────────────────────────────────────
print("Generando reporte de Data Drift...")
reporte_drift = Report(metrics=[
    DatasetDriftMetric(),
    ColumnDriftMetric(column_name="monthly_charge"),
    ColumnDriftMetric(column_name="support_tickets"),
    ColumnDriftMetric(column_name="tenure_months"),
    ColumnDriftMetric(column_name="avg_monthly_usage_gb"),
])
reporte_drift.run(reference_data=X_ref, current_data=X_cur)
reporte_drift.save_html(os.path.join(REPORTS_DIR, "reporte_drift.html"))
print("  -> reports/reporte_drift.html generado")

# ── Reporte 2: Clasificacion ──────────────────────────────────────────────────
print("Generando reporte de Clasificacion...")
reporte_clasificacion = Report(metrics=[ClassificationPreset()])
reporte_clasificacion.run(reference_data=ref_data, current_data=cur_data)
reporte_clasificacion.save_html(os.path.join(REPORTS_DIR, "reporte_clasificacion.html"))
print("  -> reports/reporte_clasificacion.html generado")

# ── Resumen en consola ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("RESUMEN DEL MONITOREO — AndesLink Servicios Digitales")
print("="*60)

drift_result   = reporte_drift.as_dict()
dataset_drift  = drift_result['metrics'][0]['result']

print(f"\nDataset drift detectado: {dataset_drift['dataset_drift']}")
print(f"Columnas con drift:       {dataset_drift['number_of_drifted_columns']} / {dataset_drift['number_of_columns']}")
print(f"Share de columnas drift:  {dataset_drift['share_of_drifted_columns']:.1%}")

print("\nColumnas analizadas:")
for m in drift_result['metrics'][1:]:
    col      = m['result'].get('column_name', 'N/A')
    score    = m['result'].get('drift_score', 0)
    detected = m['result'].get('drift_detected', False)
    print(f"  {col:30s} drift_score={score:.4f}  drift={'SI' if detected else 'NO'}")

print("\nInterpretacion:")
if dataset_drift['dataset_drift']:
    print("  ALERTA: Se detecto drift en el dataset de produccion.")
    print("  Acciones correctivas sugeridas:")
    print("  - Revisar distribucion de monthly_charge y support_tickets")
    print("  - Evaluar reentrenamiento del modelo con datos recientes")
    print("  - Ajustar umbrales si el F1 cae mas de 5%")
else:
    print("  OK: No se detecto drift significativo.")
    print("  El modelo puede continuar operando sin cambios.")

print("\nReportes guardados en reports/")
print("  - reporte_drift.html")
print("  - reporte_clasificacion.html")
