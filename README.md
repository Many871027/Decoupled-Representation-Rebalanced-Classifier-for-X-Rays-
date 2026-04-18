# DeRC-X: Decoupled Representation and Cost-Sensitive Learning for Extreme Long-Tailed Medical Imaging

Repositorio oficial del proyecto de Investigación en MLOps Clínico para el diagnóstico asimétrico de radiografías de tórax, basado en los lineamientos de diseño analizados en el *paper* referencial *"DeRC-X: Decoupled Representation and Cost-Sensitive Learning for Extreme Long-Tailed Medical Imaging"*.

## 📌 Descripción General de la Arquitectura
El diagnóstico asistido para visión computacional médica (Chest X-Ray) enfrenta un desafío crítico de escala global: el desbalance extremo de clases o distribuciones de cola larga (*long-tailed*). Patologías minoritarias pero letales (e.g. COVID-19) son aplastadas estadísticamente durante el entrenamiento por culpa de patologías mayoritarias (Pacientes Sanos o Patrones genéricos de Neumonía), provocando una peligrosa "ilusión de precisión" algorítmica.

Para mitigar este sesgo, **DeRC-X** resuelve el colapso aislando el aprendizaje de la red neuronal mediante tres ejes técnicos formales:
1. **Decoupled Representation (Representación Desacoplada):** El entrenamiento rudo en clasificación clínica se segrega en dos fases temporales.
   - *Fase 1:* Se permite a la CNN completa extraer semántica global masiva.
   - *Fase 2:* Se congela (*freeze*) el extractor de características de la convolución y se ajusta únicamente la matriz logística (clasificador) con rebalanceo matemático profundo.
2. **Cost-Sensitive Learning (Focal Loss):** Abandonamos categóricamente la entropía cruzada estándar en favor de inyectar hiperparámetros matriciales estrictos ($\gamma=2.0$, $\alpha=0.5$). Esto penaliza implacablemente al optmizador de la red cuando se equivoca en la clase minoritaria, equilibrando los tensores de gradiente generados en la asimetría extrema actual de casi $11:1$.
3. **Medical-Safe Data Augmentation:** Montado directamente mediante las Capas Preprocesadoras nativas, se establecen restricciones geométricas inquebrantables (traslación rigurosa al ~5%, rotación clínica al ~1.5º) para maximizar orgánicamente el clúster minoritario sin inyectar deformaciones algorítmicas, evitando contagiarse de "*out-of-domain artifacts*".

---

## 🚀 Despliegue Bidireccional (Hybrid MLOps)
Hemos instrumentado este repositorio para operar bajo demanda, ya sea localmente en laboratorios en el borde (*Edge*) o explotando topologías distribuidas a nivel clúster asíncrono Tensor Processing Units (TPU).

### 💻 1. Ejecución Local (On-Premise / Edge Machine)
Estructura diseñada para iteraciones investigativas y validaciones rápidas basadas en PCs con GPUs de núcleo general moderno (Plataformas Windows NVidia/Ampere/RTX). 

**Entrada Lógica Principal:** `notebooks/chest_xray_experiment.py`
- **Crecimiento de Memoria (VRAM Memory-Growth):** Auto-crecimiento elástico paramétrico para evitar desbordes colapsados (Zero OOM Errors).
- **Aceleración Dinámica local (`mixed_float16`):** Degrada operaciones inofensivas a *float16* nativo ganando hasta el doble de ciclos en núcleos Tensor locales FP16, manteniendo *float32* para capas de activación.
- **Checkpoint Local Asíncrono:** Emplea mecanismos de retención pura volcando el modelo ganador de la validación cruzada y guardando el modelo a nivel disco local de forma continua en `artifacts/Local_Model_[TAG].keras`.

**Ejecución Rápida:**
```bash
python notebooks/chest_xray_experiment.py
```

### ☁️ 2. Ejecución Distribuida Asíncrona en Nube (Google Colab TPU v6e)
Motor productivo masivo diseñado para operar *training* colosal paralelo. 

**Entrada Lógica Principal:** `colab_runner.ipynb`
- **Dominio Jerárquico (`tpu_strategy.scope()`):** El modelo, backpropagation y funciones de costo entero nacen exclusivamente blindados bajo el Scope explícito de Google para multi-dispositivos.
- **Ancho de Banda Brain Floating (`mixed_bfloat16`):** Inyección de la política obligatoria de Float Brain, adaptándose exactamente como la TPU transfiere gradientes a matriz.
- **Topología Adaptativa Autónoma (`num_replicas_in_sync`):** Multiplica inteligentemente la constante matemática de Batch (`BATCH_SIZE = 16 * Cores`) permitiendo que tu ecosistema opere con igual perfección tanto si el sistema clúster asignó un sub-núcleo solitario como si lanzó una granja de 8 matrices hiperbólicas conectadas.
- **Salvaguardado en Drive Persistente:** Múltiples serializaciones del hiperentrenamiento viajan íntegramente de vuelta al G-Drive personal de Google.

**Procedimiento de Lanzamiento:**
1. Sube y monta tu entorno TPU base en Colab abriendo tu cuenta asociada de Drive.
2. Colab usará el notebook para descargar de forma automática, nativa e imperceptible la última rama de éste mismo repositorio Git.
3. El clúster se conectará distribuyendo los recursos in-memory y devolviendo el artefacto analizado.

---

## 🏆 Evaluación SOTA Integral (Reporte Científico)
En alineación total al paper del framework DeRC, el proyecto destierra categóricamente mediciones engañosas tipo `Accuracy`. Las operaciones de control están monitoreadas única y expresamente por el componente asíncrono customizado `MedicalReportCallback`, focalizando al modelo para guiarse bajo la lupa penalizable del **Macro-F1 Score**. 

Las pruebas arquitectónicas confirmaron el descubrimiento del *Punto Cúspide (Vértice)* donde bajo métricas equilibradas, nuestra penalización estricta por Focal Loss estabiliza el **Macro-F1 general a un 0.8355** deteniendo a la vez tempranamente el entrenamiento protegiendo a la red del olvido catastrófico de la patología escasa.

---
*Desarrollado para la Cima de Operaciones MLOps & Arquitectura AI Diagnóstica.*
