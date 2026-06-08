import streamlit as st
import requests
import os
API_URL = os.getenv("API_URL", "http://localhost:8000/predecir")

st.set_page_config(
    page_title="AndesLink – Predictor de Churn",
    page_icon="📡",
    layout="centered"
)

st.title("📡 AndesLink Servicios Digitales")
st.subheader("Predictor de Abandono de Clientes")
st.markdown("Completá los datos del cliente para estimar la probabilidad de churn.")
st.divider()

# ── Formulario ───────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    tenure_months = st.number_input("Antigüedad (meses)", min_value=0, value=12)
    monthly_charge = st.number_input("Cargo mensual (USD)", min_value=0.0, value=60.0)
    support_tickets = st.number_input("Tickets de soporte", min_value=0, value=1)
    late_payments = st.number_input("Pagos tardíos", min_value=0, value=0)
    avg_monthly_usage_gb = st.number_input("Uso promedio mensual (GB)", min_value=0.0, value=30.0)

with col2:
    customer_age = st.number_input("Edad del cliente", min_value=18, value=35)
    contract_type = st.selectbox("Tipo de contrato", ["mensual", "anual", "bianual"])
    payment_method = st.selectbox("Método de pago", ["transferencia", "debito", "efectivo", "credito"])
    internet_service = st.selectbox("Servicio de internet", ["cable", "fibra", "movil", "ninguno"])
    region = st.selectbox("Región", ["centro", "norte", "oeste", "sur"])

st.divider()

# ── Predicción ───────────────────────────────────────────────────────────────
if st.button("🔍 Predecir Churn", use_container_width=True):
    payload = {
        "tenure_months": tenure_months,
        "monthly_charge": monthly_charge,
        "support_tickets": support_tickets,
        "late_payments": late_payments,
        "avg_monthly_usage_gb": avg_monthly_usage_gb,
        "customer_age": customer_age,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "internet_service": internet_service,
        "region": region,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=5)

        if response.status_code == 200:
            resultado = response.json()
            prob = resultado["probabilidad"]
            churn = resultado["churn"]
            etiqueta = resultado["etiqueta"]

            st.subheader("Resultado")

            if churn == 1:
                st.error(f"⚠️ **{etiqueta}**")
            else:
                st.success(f"✅ **{etiqueta}**")

            st.metric(label="Probabilidad de churn", value=f"{prob * 100:.1f}%")
            st.progress(prob)

        else:
            st.error(f"Error en la API: {response.status_code} – {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("❌ No se pudo conectar con la API. Verificá que esté corriendo en http://localhost:8000")
    except requests.exceptions.Timeout:
        st.error("❌ La API tardó demasiado en responder.")
