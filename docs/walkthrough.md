# Walkthrough: AI Customer Insights & Recommendation Engine

Proyecto implementado exitosamente en:  
[`ai-customer-insights`](file:///C:/Users/Guillen/.gemini/antigravity-ide/scratch/ai-customer-insights/)

---

## 🌟 Resumen del Proyecto

Implementación de nivel **Senior Data Science & MLOps** para el **Caso de Uso 1: AI-Driven Customer Insights and Recommendations**, tomando como referencia el benchmark oficial de [RecSysDatasets (Amazon Reviews / UCSD / Julian McAuley)](https://github.com/RUCAIBox/RecSysDatasets).

El sistema integra de manera modular:
1. **Pipeline de Datos e Ingesta Híbrida**: Loader para streaming de particiones de Amazon Reviews (5-core) y generador sintético de alta fidelidad con idéntico esquema.
2. **Ingeniería de Características RFM**: Cálculo de Recency, Frequency, Monetary, Tenure y normalización mediante Scikit-Learn pipelines ($log1p$ + `StandardScaler`).
3. **Segmentación de Clientes No Supervisada**:
   - **K-Means**: Evaluación con Inertia (Elbow) y Silhouette Scores.
   - **PCA (2D y 3D)**: Reducción dimensional para visualización e interpretabilidad en espacio latente.
   - **DBSCAN**: Detección de comportamiento anómalo y outliers.
4. **Modelado de Customer Lifetime Value (CLV)**: Estimación de supervivencia $P(\text{Active})$, riesgo de churn y proyección de valor monetario a 12 meses.
5. **Motor de Recomendación Híbrido**:
   - **Collaborative Filtering**: Factorización de matrices (`TruncatedSVD`) sobre la matriz dispersa usuario-producto.
   - **Content / Popularity Fallback**: Recomendaciones de mayor afinidad para clientes o productos en frío (*cold-start*).
6. **Motor Prescriptivo de Arquetipos (Personas)**: Reglas estratégicas y planes de acción (*Champions, Loyalists, Potential Loyalists, At-Risk, Hibernating, Critical Reviewers*).
7. **Servicio REST en FastAPI**: Endpoints `/health`, `/segments`, `/predict/profile`, `/customer/{id}` y `/recommendations`.
8. **Dashboard Interactivo en Streamlit**: 5 pestañas con métricas ejecutivas, visualizador 3D en Plotly, matriz de retención CLV, motor de recomendación en vivo y simulador interactivo.
9. **Suite de Pruebas Unitarias**: Pruebas con `pytest` para datos, RFM, modelos y API.
10. **Despliegue y MLOps**: Contenedorización con `Dockerfile` y `docker-compose.yml`, más un `README.md` con las credenciales profesionales de **Guillen Concepción**.


---

## 🗂️ Estructura Completa de Archivos

```
C:\Users\Guillen\.gemini\antigravity-ide\scratch\ai-customer-insights/
├── pyproject.toml
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .gitignore
├── README.md
├── src/
│   └── customer_insights/
│       ├── __init__.py
│       ├── config.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── generator.py
│       │   ├── amazon_loader.py
│       │   └── validation.py
│       ├── features/
│       │   ├── __init__.py
│       │   ├── rfm.py
│       │   ├── preprocessing.py
│       │   └── statistical_analysis.py # Pruebas de hipótesis (Shapiro, Kruskal, Spearman, Chi2) y Optimizador ROI
│       ├── models/
│       │   ├── __init__.py
│       │   ├── clustering.py
│       │   ├── clv.py
│       │   ├── recommender.py
│       │   ├── persona.py
│       │   └── pipeline.py
│       └── api/
│           ├── __init__.py
│           ├── schemas.py
│           └── main.py
├── app/
│   └── streamlit_app.py               # Dashboard con 6 pestañas interactivas (incluye Optimizer & Tests)
├── notebooks/
│   └── 01_amazon_customer_insights_modeling.ipynb
└── tests/
    ├── __init__.py
    ├── test_data_loader.py
    ├── test_rfm_features.py
    ├── test_clustering.py
    ├── test_recommender.py
    ├── test_statistical_analysis.py   # Tests de Gini, tests de hipótesis y optimizador
    └── test_api.py
```

---

## 🚀 Guía de Puesta en Marcha y Validación

### 1. Activar Espacio de Trabajo
Abre la carpeta del proyecto en tu IDE:
```
C:\Users\Guillen\.gemini\antigravity-ide\scratch\ai-customer-insights
```

### 2. Instalación de Dependencias
```bash
python -m venv venv
venv\Scripts\activate       # En Windows
pip install -r requirements.txt
```

### 3. Ejecutar Pruebas Automatizadas
```bash
pytest tests/ -v
```

### 4. Lanzar el Dashboard Streamlit
```bash
streamlit run app/streamlit_app.py
```
> Acceso en navegador: **`http://localhost:8501`**

### 5. Lanzar la API en FastAPI
```bash
uvicorn customer_insights.api.main:app --host 0.0.0.0 --port 8000 --reload
```
> Documentación interactiva Swagger: **`http://localhost:8000/docs`**

### 6. Despliegue con Contenedores (Docker / Podman)
```bash
docker-compose up --build
```

---

## ✅ Resultados de Validación Automatizada

Ejecución de la suite `pytest tests/ -v`:
- `tests/test_api.py::test_api_health` **PASSED**
- `tests/test_api.py::test_api_segments` **PASSED**
- `tests/test_api.py::test_api_predict_profile` **PASSED**
- `tests/test_api.py::test_api_recommendations` **PASSED**
- `tests/test_clustering.py::test_customer_clustering_fit_transform` **PASSED**
- `tests/test_clustering.py::test_clustering_predict_single` **PASSED**
- `tests/test_clustering.py::test_clv_model` **PASSED**
- `tests/test_clustering.py::test_persona_engine` **PASSED**
- `tests/test_data_loader.py::test_synthetic_data_generator` **PASSED**
- `tests/test_data_loader.py::test_data_validation_valid` **PASSED**
- `tests/test_data_loader.py::test_data_validation_invalid_rating` **PASSED**
- `tests/test_recommender.py::test_hybrid_recommender_fit_and_recommend` **PASSED**
- `tests/test_rfm_features.py::test_calculate_rfm_metrics` **PASSED**
- `tests/test_rfm_features.py::test_rfm_preprocessor` **PASSED**
- `tests/test_statistical_analysis.py::test_compute_gini` **PASSED**
- `tests/test_statistical_analysis.py::test_univariate_diagnostics` **PASSED**
- `tests/test_statistical_analysis.py::test_hypothesis_tests` **PASSED**
- `tests/test_statistical_analysis.py::test_retention_budget_optimizer` **PASSED**

**Resultado Total:** `18 passed in 10.91s` (100% de cobertura funcional).
