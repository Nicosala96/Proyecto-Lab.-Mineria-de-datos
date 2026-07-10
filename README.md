# 🔍 MLOps Local – Predicción de Churn de Clientes

**Proyecto académico · ISTEA – Laboratorio de Minería de Datos**  
**Empresa simulada:** AndesLink Servicios Digitales S.A.  
**Docente:** Prof. Diego Mosquera

---

## 📋 Descripción del Proyecto

Solución end-to-end de MLOps local para predecir el abandono de clientes (churn) de AndesLink Servicios Digitales S.A. El sistema permite entrenar, versionar, desplegar y monitorear un modelo de clasificación binaria que estima la probabilidad de que un cliente cancele su suscripción.

**Variable objetivo:**
- `1` → El cliente abandona el servicio (churn)
- `0` → El cliente permanece activo

---

## 🗂️ Estructura del Proyecto

```
proyecto/
│
├── data/
│   ├── raw/                        # Dataset original sin modificar
│   │   └── churn_sintetico.csv
│   └── processed/                  # Dataset post-EDA listo para entrenamiento
│       └── churn_sintetico_EDA.csv
│
├── models/                         # Artefactos serializados listos para inferencia
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── scaler.joblib
│   ├── encoder_contract.joblib
│   ├── encoder_payment.joblib
│   ├── encoder_internet.joblib
│   └── encoder_region.joblib
│
├── notebooks/                      # Exploración y entrenamiento en Jupyter
│   ├── EDA.ipynb
│   └── Entrenamiento.ipynb
│
├── src/                            # Código fuente de la solución
│   ├── api.py                      # API de inferencia (FastAPI)
│   ├── app.py                      # Interfaz gráfica (Streamlit)
│   └── monitoreo_evidently.py      # Script de monitoreo de datos y modelo
│
├── tests/                          # Pruebas automáticas con pytest
│   ├── __init__.py
│   └── test_api.py
│
├── reports/                        # Informes técnicos y reportes de monitoreo
│   ├── evidencia_tests.txt
│   ├── evidencia_docker.txt
│   ├── reporte_drift.html          # Generado por Evidently
│   └── reporte_clasificacion.html  # Generado por Evidently
│
├── monitoring/                     # Configuración de Prometheus y Grafana
│   ├── prometheus.yml
│   └── grafana/
│       └── provisioning/
│           ├── datasources/
│           │   └── datasource.yml
│           └── dashboards/
│               ├── dashboard_provider.yml
│               └── andeslink_dashboard.json
│
├── mlruns/                         # Experimentos MLflow (auto-generado)
│
├── Dockerfile.api                  # Imagen Docker de la API
├── Dockerfile.streamlit            # Imagen Docker de la GUI
├── docker-compose.yml              # Orquestación local (API + GUI + Prometheus + Grafana)
├── requirements.txt                # Dependencias para Docker
├── environment.yml                 # Entorno reproducible de conda
├── .gitignore
├── .dvcignore
└── README.md
```

---

## ⚙️ Requisitos Previos

- [Anaconda](https://www.anaconda.com/download) o Miniconda
- [Git](https://git-scm.com/)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Cuenta gratuita en [DagsHub](https://dagshub.com) para poder hacer `dvc pull`

### Obtener token de DagsHub

1. Creá una cuenta gratuita en [https://dagshub.com](https://dagshub.com)
2. Andá a [https://dagshub.com/user/settings/tokens](https://dagshub.com/user/settings/tokens)
3. Generá un token nuevo y copialo — lo vas a necesitar en el paso de DVC

---

## ⚠️ Advertencia importante antes de levantar Docker

**El build de Docker requiere que los archivos `.joblib` estén presentes localmente.**  
Si no ejecutás `dvc pull` antes de `docker-compose up --build`, el build va a fallar porque los modelos no van a existir en la carpeta `models/`.

El orden correcto es siempre:
```
1. dvc pull                    →   descarga los modelos
2. docker-compose up --build   →   construye las imágenes
```

---

## 🚀 Opción A — Despliegue con Docker (recomendado)

### 1. Clonar el repositorio

```bash
git clone https://github.com/Nicosala96/Proyecto-Lab.-Mineria-de-datos
cd Proyecto-Lab.-Mineria-de-datos
```

### 2. Crear y activar el entorno

```bash
conda create -n andeslink-churn python=3.12
conda activate andeslink-churn
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
pip install "dvc[http]"
pip install mlflow jupyter jupyterlab
pip install "numpy==1.26.4" --only-binary=numpy
pip install "scipy<1.13"
pip install evidently==0.4.33
```

### 4. Configurar credenciales de DagsHub

```bash
dvc remote modify --local origin auth basic
dvc remote modify --local origin user TU_USUARIO_DAGSHUB
dvc remote modify --local origin password TU_TOKEN_DAGSHUB
```

### 5. Descargar datos y modelos ⚠️ obligatorio antes del build

```bash
dvc pull
```

### 6. Asegurarse de tener Docker Desktop abierto y corriendo

### 7. Levantar los servicios

```bash
docker-compose up --build
```

La primera vez tarda unos minutos mientras descarga las imágenes base.

### 8. Acceder a los servicios

| Servicio | URL |
|----------|-----|
| GUI (Streamlit) | http://localhost:8501 |
| API (documentación interactiva) | http://localhost:8000/docs |
| API health check | http://localhost:8000/health |
| Métricas Prometheus | http://localhost:8000/metrics |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (usuario: admin / contraseña: admin) |

### 9. Detener los servicios

```bash
docker-compose down
```

---

## 🐍 Opción B — Ejecución local con conda

### 1. Clonar el repositorio

```bash
git clone https://github.com/Nicosala96/Proyecto-Lab.-Mineria-de-datos
cd Proyecto-Lab.-Mineria-de-datos
```

### 2. Crear y activar el entorno

```bash
conda create -n andeslink-churn python=3.12
conda activate andeslink-churn
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
pip install "dvc[http]"
pip install mlflow jupyter jupyterlab
pip install "numpy==1.26.4" --only-binary=numpy
pip install "scipy<1.13"
pip install evidently==0.4.33
```

### 4. Configurar credenciales de DagsHub

```bash
dvc remote modify --local origin auth basic
dvc remote modify --local origin user TU_USUARIO_DAGSHUB
dvc remote modify --local origin password TU_TOKEN_DAGSHUB
```

### 5. Descargar datos y modelos

```bash
dvc pull
```

### 6. Ejecutar los notebooks en orden

```bash
jupyter lab
```

- Primero `notebooks/EDA.ipynb` → genera `data/processed/churn_sintetico_EDA.csv`
- Luego `notebooks/Entrenamiento.ipynb` → genera los modelos en `models/`

### 7. Levantar la API (Terminal 1)

```bash
cd src
uvicorn api:app --reload
```

### 8. Levantar la GUI (Terminal 2)

```bash
cd src
streamlit run app.py
```

---

## 📊 Monitoreo con Evidently

Para generar los reportes de drift y degradación del modelo:

```bash
conda activate andeslink-churn
cd src
python monitoreo_evidently.py
```

Los reportes se guardan en `reports/`:
- `reporte_drift.html` — detecta cambios en la distribución de las variables
- `reporte_clasificacion.html` — evalúa degradación del modelo

---

## 🧪 Pruebas automáticas

```bash
conda activate andeslink-churn
pytest tests/test_api.py -v
```

Resultado esperado: **12 passed**.

---

## 🔧 Troubleshooting

| Problema | Causa | Solución |
|----------|-------|----------|
| `docker-compose up --build` falla con `FileNotFoundError` | Los `.joblib` no están en `models/` | Ejecutar `dvc pull` primero |
| `dvc pull` falla con error de autenticación | Token de DagsHub no configurado | Ejecutar los `dvc remote modify --local` del paso 4 |
| `dvc` no reconocido como comando | DVC no instalado en el entorno activo | Verificar que `andeslink-churn` esté activo y ejecutar `pip install "dvc[http]"` |
| Puerto 8000, 8501, 9090 o 3000 ocupado | Otro proceso usa el puerto | Ejecutar `docker-compose down` o cerrar el proceso |
| GUI muestra error de conexión | La API no está corriendo | Verificar que `andeslink-api` esté `healthy` con `docker ps` |
| Grafana no muestra datos | Prometheus aún no recolectó métricas | Hacer algunas predicciones desde la GUI y esperar 15 segundos |
| `evidently` falla con error de numpy | Incompatibilidad de versiones | Ejecutar `pip install "numpy==1.26.4" --only-binary=numpy` |

---

## 📊 Seguimiento de experimentos con MLflow

```bash
mlflow ui
```

Abrí en el navegador: http://localhost:5000

---

## 🤖 Modelos entrenados

| Modelo | Archivo | Descripción |
|--------|---------|-------------|
| Regresión Logística | `logistic_regression.joblib` | Penalty L1, solver liblinear, v1.1.0 |
| Árbol de Decisión | `decision_tree.joblib` | max_depth=3, criterion gini |
| Scaler | `scaler.joblib` | StandardScaler — variables numéricas |
| Encoders | `encoder_*.joblib` | OrdinalEncoder por variable categórica |

---

## 📁 Dataset

| Campo | Detalle |
|-------|---------|
| Fuente | Dataset sintético generado para el proyecto |
| Archivo original | `data/raw/churn_sintetico.csv` |
| Archivo procesado | `data/processed/churn_sintetico_EDA.csv` |
| Registros | 5.000 clientes |
| Variable objetivo | `churn` (0 = activo, 1 = abandona) |
| Variables numéricas | `tenure_months`, `monthly_charge`, `support_tickets`, `late_payments`, `avg_monthly_usage_gb`, `customer_age` |
| Variables categóricas | `contract_type`, `payment_method`, `internet_service`, `region` |
| Balance de clases | Aproximadamente 70% no churn / 30% churn |

---

## 🔄 Versionado

- **Git** — control de versiones del código
- **DVC** — versionado de datos y modelos con remote en DagsHub
- **MLflow** — tracking de experimentos de entrenamiento

---

## 📌 Estado del Proyecto

| Entrega | Estado | Descripción |
|---------|--------|-------------|
| ✅ Primer Parcial | Completado | EDA, entrenamiento, serialización, MLflow, DVC, Git |
| ✅ Segundo Parcial | Completado | API FastAPI, GUI Streamlit, Docker, pytest |
| ✅ Examen Final | Completado | Prometheus, Grafana, Evidently, monitoreo |

---

## 👥 Participantes

| Nombre |
|--------|
| Ronald Boyd |
| Rodrigo Figueredo |
| Nicolas Sala |

**ISTEA · Laboratorio de Minería de Datos**
