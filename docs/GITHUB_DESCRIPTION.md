# 🐙 Descripciones y Metadatos del Proyecto para GitHub

**Autor:** Guillen Concepción — *Senior Data Scientist & MLOps Engineer*  
**Proyecto:** DS-AI Customer Insights, Survival Lifetime Value (CLV) & Recommendation Engine  
**Repositorio Oficial:** [github.com/GuillenConcepcion/AI-Driven-Customer-Insights-Recommendation-Engine](https://github.com/GuillenConcepcion/AI-Driven-Customer-Insights-Recommendation-Engine)

---

## 📌 1. Campo "About" del Repositorio de GitHub (Repo Bio)

> 💡 **Instrucciones:** En la página principal de tu repositorio en GitHub, haz clic en el icono de engranaje ⚙️ junto a **About** (esquina superior derecha) y pega una de las siguientes opciones:

### 🌐 Opción A: En Inglés (Recomendada para visibilidad internacional y reclutadores globales)
```text
Production-grade Customer Insights & Survival CLV Engine: 3D PCA centroids, Poisson GLM, Dual Hybrid Recommender (SVD + TF-IDF) & 0-1 Knapsack retention optimizer. Served via FastAPI (<25ms P99) & Streamlit Cockpit.
```
*(226 caracteres — conciso, técnico y con métricas clave)*

---

### 🇪🇸 Opción B: En Español
```text
Motor empresarial de Customer Insights, CLV de Supervivencia y Recomendación Híbrida (SVD + TF-IDF) con optimización Knapsack de presupuesto. Microservicios desacoplados en FastAPI (<25ms) y cockpit en Streamlit.
```
*(213 caracteres — formal, directo y de nivel Senior)*

---

## 🏷️ 2. Topics / Etiquetas de GitHub (Sección "Topics")

Copia y pega las siguientes etiquetas en el campo **Topics** de GitHub para optimizar el SEO y la indexación en búsquedas de GitHub:

```text
machine-learning, mlops, customer-lifetime-value, clv, recommender-system, customer-segmentation, fastapi, streamlit, decision-intelligence, docker, pca, python, data-science, predictive-analytics
```

---

## 🎯 3. Descripción Extendida (Para GitHub Releases, Pinned Repo o Show HN)

```markdown
### 🛍️ AI-Driven Customer Insights, Survival Lifetime Value (CLV) & Recommendation Engine
> **Enterprise-Grade Behavioral Econometrics, Survival CLV Modeling & Dual Hybrid Recommendations**  
> *Developed by [Guillen Concepción](https://github.com/GuillenConcepcion) — Senior Data Scientist & MLOps Engineer*

#### ⚡ Resumen Ejecutivo
Plataforma Cloud-Native de Machine Learning y Decision Intelligence benchmarkeada contra la colección oficial de **Amazon Product Reviews (McAuley / UCSD RecSysDatasets)**. Reemplaza las heurísticas comerciales estáticas por un sistema integrado de inferencia en tiempo real y optimización prescriptiva de presupuestos.

#### 🏛️ Puntos Clave de Ingeniería & Negocio:
1. **Tratamiento de Varianza y Asimetría:** Normalización no lineal $\ln(1+x)$ para mitigar la asimetría de Pareto ($g_1 = +2.41, g_2 = +7.62$) previo a K-Means ($k=4$) y PCA 3D con cálculo explícito de centroides y vector euclídeo.
2. **Diagnóstico Econométrico:** Curva de Lorenz y Coeficiente de Gini ($G \approx 0.62$), demostrando que el **20% superior genera el 74.2% del volumen transaccional**.
3. **Modelado Predictivo de CLV:** Estimación de supervivencia *Buy-Till-You-Die* (BTYD) y regresión Poisson GLM para erradicar predicciones negativas de OLS, con descuento NPV.
4. **Motor de Recomendación Híbrido Dual:** Factorización latente (TruncatedSVD) + Similitud semántica TF-IDF NLP con **$0.0\%$ fallos ante Cold-Start**.
5. **Analítica Prescriptiva:** Asignación óptima de presupuesto de retención formulada como un **Problema de la Mochila 0-1 (Knapsack)** con **$5.83\text{x}$ ROI proyectado**.
6. **Arquitectura Cloud-Native:** Microservicio FastAPI asíncrono con patrón Lifespan (latencia P99 $< 25\text{ ms}$), Cockpit Streamlit WebGL 3D a 300 DPI y **21/21 tests pasando en Pytest (100%)**.

#### ⚡ Verificación Rápida en 10 Segundos:
```bash
git clone https://github.com/GuillenConcepcion/AI-Driven-Customer-Insights-Recommendation-Engine.git
cd AI-Driven-Customer-Insights-Recommendation-Engine
python scripts/demo_quickstart.py
```
```
