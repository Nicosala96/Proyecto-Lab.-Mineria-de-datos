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
│   └── app.py                      # Interfaz gráfica (Streamlit)
│
├── tests/                          # Pruebas automáticas con pytest
│   ├── __init__.py
│   └── test_api.py
│
├── reports/                        # Informes técnicos
│
├── mlruns/                         # Experimentos MLflow (auto-generado)
│
├── Dockerfile.api                  # Imagen Docker de la API
├── Dockerfile.streamlit            # Imagen Docker de la GUI
├── docker-compose.yml              # Orquestación local
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

## 🚀 Opción A — Despliegue con Docker (recomendado)

### 1. Clonar el repositorio

```bash
git clone https://github.com/Nicosala96/Proyecto-Lab.-Mineria-de-datos
cd Proyecto-Lab.-Mineria-de-datos
```

### 2. Crear y activar el entorno

```bash
conda create -n andeslink-churn python=3.13
conda activate andeslink-churn
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
pip install "dvc[http]"
pip install mlflow jupyter jupyterlab
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
conda create -n andeslink-churn python=3.13
conda activate andeslink-churn
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
pip install "dvc[http]"
pip install mlflow jupyter jupyterlab
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

## 🧪 Pruebas automáticas

```bash
conda activate andeslink-churn
pytest tests/test_api.py -v
```

Resultado esperado: **6 passed**.

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
| Regresión Logística | `logistic_regression.joblib` | Penalty L1, solver liblinear |
| Árbol de Decisión | `decision_tree.joblib` | max_depth=3, criterion gini |
| Scaler | `scaler.joblib` | StandardScaler — variables numéricas |
| Encoders | `encoder_*.joblib` | OrdinalEncoder por variable categórica |

---

## 📁 Dataset

| Campo | Detalle |
|-------|---------|
| Fuente | Dataset sintético generado para el proyecto |
| Registros | 5.000 clientes |
| Variable objetivo | `churn` (0/1) |

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
| 🔲 Examen Final | Pendiente | Prometheus, Grafana, Evidently, video |

---

## 👥 Participantes

| Nombre |
|--------|
| Ronald Boyd |
| Rodrigo Figueredo |
| Nicolas Sala |

**ISTEA · Laboratorio de Minería de Datos**
