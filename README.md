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
│   ├── logistic_regression.joblib  # Modelo de Regresión Logística
│   ├── decision_tree.joblib        # Modelo de Árbol de Decisión
│   ├── scaler.joblib               # StandardScaler (normalización numérica)
│   └── ordinal_encoder.joblib      # OrdinalEncoder (variables categóricas)
│
├── notebooks/                      # Exploración y entrenamiento en Jupyter
│   ├── EDA.ipynb                   # Análisis exploratorio de datos
│   └── Entrenamiento.ipynb         # Entrenamiento, evaluación y serialización
│
├── src/                            # (Próximas entregas) Módulos reutilizables
│   ├── preprocess.py
│   └── train.py
│
├── tests/                          # (Próximas entregas) Pruebas con pytest
│
├── scripts/                        # Scripts auxiliares de ejecución
│
├── reports/                        # Informes técnicos y gráficos generados
│
├── mlruns/                         # Experimentos registrados por MLflow (auto-generado)
│
├── environment.yml                 # Entorno reproducible de conda
├── .gitignore                      # Archivos excluidos del control de versiones
├── .dvc/                           # Configuración de DVC (versionado de datos)
└── README.md                       # Este archivo
```

---

## ⚙️ Requisitos Previos

- [Anaconda](https://www.anaconda.com/download) o Miniconda instalado
- [Git](https://git-scm.com/) instalado
- Sistema operativo: Windows 10/11 (o Linux/macOS)

---

## 🚀 Instalación y Configuración del Entorno

### 1. Clonar el repositorio

```bash
git clone <URL-del-repositorio>
cd proyecto
```

### 2. Crear el entorno de conda

```bash
conda env create -f environment.yml
```

### 3. Activar el entorno

```bash
conda activate andeslink-churn
```

### 4. Verificar la instalación

```bash
python -c "import sklearn, mlflow, joblib, pandas; print('Entorno OK')"
```

---

## 📓 Ejecución de los Notebooks

Los notebooks deben ejecutarse **en orden** desde la carpeta `notebooks/`.

### Paso 1 – Análisis Exploratorio (EDA)

```bash
jupyter lab notebooks/EDA.ipynb
```

**Qué hace:**
- Carga el dataset crudo desde `data/raw/churn_sintetico.csv`
- Analiza calidad de datos (nulos, duplicados, tipos)
- Genera visualizaciones de distribución y comportamiento por variables clave
- Exporta el dataset procesado a `data/processed/churn_sintetico_EDA.csv`

### Paso 2 – Entrenamiento y Serialización

```bash
jupyter lab notebooks/Entrenamiento.ipynb
```

**Qué hace:**
- Carga el dataset procesado desde `data/processed/`
- Aplica encoding ordinal a variables categóricas (`contract_type`, `payment_method`, `internet_service`, `region`)
- Escala variables numéricas con `StandardScaler`
- Entrena dos modelos: **Árbol de Decisión** y **Regresión Logística**
- Evalúa con métricas: accuracy, F1-score, matriz de confusión, classification report
- Registra experimentos en MLflow (`mlruns/`)
- Serializa todos los artefactos en `models/`

---

## 📊 Seguimiento de Experimentos con MLflow

Una vez ejecutado el notebook de entrenamiento, podés acceder a la interfaz de MLflow para comparar experimentos:

```bash
mlflow ui
```

Luego abrí en el navegador: [http://localhost:5000](http://localhost:5000)

Verás el experimento `Analisis_Churn_Clientes` con las runs:
- `Regresion_Logistica` — accuracy, f1_score, artefacto del modelo
- `Arbol_Decision` — accuracy, f1_score, artefacto del modelo

---

## 🤖 Modelos Entrenados

| Modelo | Archivo | Descripción |
|--------|---------|-------------|
| Regresión Logística | `models/logistic_regression.joblib` | Penalty L1, solver liblinear, class_weight ajustado |
| Árbol de Decisión | `models/decision_tree.joblib` | max_depth=3, criterion gini, class_weight ajustado |
| Scaler | `models/scaler.joblib` | StandardScaler ajustado sobre variables numéricas |
| Encoder | `models/ordinal_encoder.joblib` | OrdinalEncoder para variable `region` |

### Cargar un modelo para inferencia

```python
import joblib
import pandas as pd

# Cargar artefactos
modelo = joblib.load('models/logistic_regression.joblib')
scaler = joblib.load('models/scaler.joblib')

# Preparar un ejemplo de entrada (valores ya procesados)
X_nuevo = pd.DataFrame([{
    'tenure_months': 24,
    'monthly_charge': 85.0,
    'support_tickets': 2,
    'late_payments': 0,
    'avg_monthly_usage_gb': 40.0,
    'contract_type': 1,   # anual
    'customer_age': 35
}])

columnas_num = ['tenure_months', 'monthly_charge', 'support_tickets',
                'late_payments', 'avg_monthly_usage_gb', 'customer_age']

X_nuevo[columnas_num] = scaler.transform(X_nuevo[columnas_num])
prediccion = modelo.predict(X_nuevo)
print("Churn:", prediccion[0])  # 0 = se queda, 1 = se va
```

---

## 📁 Dataset

| Campo | Detalle |
|-------|---------|
| Fuente | Dataset sintético generado para el proyecto |
| Archivo original | `data/raw/churn_sintetico.csv` |
| Registros | 5.000 clientes |
| Variable objetivo | `churn` (0/1) |

**Variables principales:**

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `customer_age` | Numérica | Edad del cliente |
| `tenure_months` | Numérica | Meses de antigüedad |
| `monthly_charge` | Numérica | Cargo mensual |
| `total_charges` | Numérica | Total facturado |
| `avg_monthly_usage_gb` | Numérica | Uso mensual promedio en GB |
| `support_tickets` | Numérica | Tickets de soporte abiertos |
| `late_payments` | Numérica | Pagos tardíos |
| `contract_type` | Categórica | mensual / anual / bianual |
| `payment_method` | Categórica | transferencia / debito / efectivo / credito |
| `internet_service` | Categórica | cable / fibra / movil / ninguno |
| `region` | Categórica | centro / norte / oeste / sur |
| `churn` | Binaria (target) | 0 = activo, 1 = abandona |

---

## 🔄 Versionado del Proyecto

### Git – Control de versiones del código

```bash
# Ver estado del repositorio
git status

# Ver historial de commits
git log --oneline
```

### DVC – Versionado de datos y pipeline

```bash
# Ver el estado de los archivos trackeados por DVC
dvc status

# Reproducir el pipeline completo
dvc repro
```

---

## 🐍 Entorno reproducible (`environment.yml`)

El archivo `environment.yml` en la raíz del proyecto define todas las dependencias necesarias. Si necesitás actualizar el entorno después de cambios:

```bash
conda env update -f environment.yml --prune
```

---

## 📌 Estado del Proyecto por Entrega

| Entrega | Estado | Descripción |
|---------|--------|-------------|
| ✅ Primer Parcial | Completado | EDA, entrenamiento, serialización, MLflow, Git |
| 🔲 Segundo Parcial | Pendiente | API FastAPI, GUI Streamlit, Docker |
| 🔲 Examen Final | Pendiente | Prometheus, Grafana, Evidently, video |

---

## 👥 Autores

Proyecto académico desarrollado para la materia **Laboratorio de Minería de Datos** – ISTEA.

- Ronald Boyd
- Rodrigo Figueredo
- Nicolás Sala
=======
# Proyecto-Lab.-Mineria-de-datos
Trabajo integrador de la asignatura Mineria de Datos II