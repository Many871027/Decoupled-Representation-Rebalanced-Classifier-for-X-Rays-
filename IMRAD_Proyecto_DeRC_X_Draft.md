# Estrategias Avanzadas para MLOps en Imágenes Médicas Desbalanceadas

**Autores:** Equipo de Investigación y Desarrollo (I+D) y Arquitectura de IA
**Fecha:** Abril 2026

---

## 1. Resumen (Abstract)
El diagnóstico asistido por computadora en imágenes radiológicas torácicas (Chest X-Ray) enfrenta un desafío persistente: el desbalance extremo de clases de "cola larga" (long-tailed), donde patologías críticas como el COVID-19 representan una fracción mínima del conjunto de datos. Este artículo detalla la implementación de la arquitectura *Decoupled Representation & Rebalanced Classifier* (DeRC) desplegada sobre clústeres TPU v6e. Presentamos un marco MLOps diseñado específicamente para evadir la "ilusión de precisión", integrando métricas rigurosas (Macro-F1), optimización por *Focal Loss*, y técnicas de aumento de datos anatómicamente seguras ("Medical-Safe"). Los resultados demuestran que el desacoplamiento del extractor de características y el clasificador, combinado con distribuciones dirigidas por el host (Host-Driven Distribution), conforma un ecosistema robusto e industrial para aplicaciones de Deep Learning Médico.

## 2. Introducción
Los conjuntos de datos médicos en el mundo real exhiben naturalmente distribuciones asimétricas de cola larga. En la detección de enfermedades pulmonares, asegurar que los sistemas de Inteligencia Artificial aprendan a identificar características raras pero letales constituye un problema que no se resuelve con arquitecturas estándar. 

Por regla general, evaluar arquitecturas sobre datos asimétricos valiéndose simplemente de la métrica de exactitud (Accuracy) enmascara el fracaso del clasificador de IA sobre la clase minoritaria, lo cual resulta inaceptable en un entorno de salud digital. Con el propósito de mitigar este sesgo biológico-estadístico sin sacrificar el rendimiento por computadora, proponemos un rediseño de las arquitecturas MLOps estándar de clasificación, introduciendo métricas ajustadas, enfoques que manipulan el gradiente de error, y procesos asincrónicos distribuidos en el hardware.

## 3. Trabajo Relacionado
La vasta bibliografía referenciada en intervenciones sobre datos asimétricos clasifica las soluciones típicamente en métodos centrados en datos y métodos centrados en algoritmos. Según el documento base *"Arquitecturas y estrategias avanzadas para el manejo de conjuntos de datos de imágenes desbalanceados en sistemas de visión por computadora"*, si bien hoy existen enfoques de generación sintética de imágenes basados en difusión (*DiffuLT*), estas soluciones traen aparejados elevados costes computacionales y posibles sesgos de generación. A su vez, los paradigmas como *Mixup* o *CutMix*—que funcionan perfectamente sobre dominios genéricos—introducen artefactos no naturales en dominios médicos donde la estructura semántica anatómica es primordial. 

En cuanto a nivel arquitectónico, descubrimientos recientes de representación de cola larga establecen que las redes neuronales pueden aprender representaciones de alta calidad incluso bajo dominancia de múltiples clases mayoritarias, siempre que la etapa final de clasificación sea entrenada de manera desacoplada y penalizada simétricamente.

## 4. Materiales y Métodos

### 4.1. Sustitución Paramétrica: Rompiendo la Ilusión de Precisión
Dado el peligro inherente en reportar métricas de exactitud (`Accuracy`) frente a incidencias minoritarias como el COVID-19, se ha erradicado esta métrica como referente de evaluación directiva. El ecosistema DeRC opera en torno al **F1-Score (Macro)**. Se implementó un *callback* personalizado (`MedicalReportCallback`), el cual captura la especificidad y asertividad a nivel clase para ejecutar políticas dinámicas de `EarlyStopping`, forzando la recuperación de los pesos tensoriales que exhiben el mayor equilibrio diagnóstico integral.

### 4.2. Matemáticas Sensibles al Costo: Aplicación de Focal Loss y Resolución Espacial
La entropía cruzada predeterminada pierde resolución cognitiva al intentar discriminar clases saturadas con las anomalías escasas. Se resolvió integrando de forma estricta la **Focal Loss**, que modula el espectro de atención sobre ejemplos difíciles con un factor de enfoque empírico ($\gamma = 2.0$), combinado con una priorización balanceadora ($\alpha = 0.5$). 

**Impacto de la Resolución (512x512 vs 64x64):**
La eficacia del descenso de gradiente bajo Focal Loss está intrínsecamente ligada a la dimensionalidad de entrada. Se definió la ingesta hiperparamétrica en `IMG_HEIGHT = 512, IMG_WIDTH = 512`. La diferencia respecto a usar resoluciones reducidas (como 64x64) es categórica: a escalas bajas, las características sutiles del COVID-19 (opacidades en vidrio esmerilado o atenuaciones periféricas) se distorsionan y colapsan en ruido de píxeles. La dimensión de 512x512 permite a la arquitectura neuronal extraer detalles patológicos finos, suministrando representaciones de alta fidelidad semántica a la métrica de Focal Loss. Esto detona una precisión formidable en el descenso de la pérdida, permitiendo al gradiente castigar certeramente los verdaderos errores de la clase minoritaria sin ser cegado por el ruido del reescalado.

### 4.3. Aumentación Data-Centric "Medical-Safe"
Acatando las fuertes restricciones geométricas de la radiología clínica, donde modificaciones extremas distorsionan la proyección cardiomediastínica, se implementó directamente dentro de las capas preprocesadoras del modelo un bloque de *Data Augmentation* configurado a medida para mimetizar la varianza natural del equipamiento radiológico. Esta es una estrategia crucial para aumentar e hiper-sintetizar artificialmente la data disponible en el entrenamiento, atacando la escasez del COVID-19.

Las tolerancias (Hiperparámetros *Medical-Safe*) implementadas son:
- **Rotación (`AUG_ROTATION = 0.027`):** Refleja inclinaciones sutiles postulares del paciente ($\approx1.5^\circ$).
- **Zoom (`AUG_ZOOM = 0.1`):** Acepta expansiones respiratorias y variabilidad en la distancia focal emisor-receptor (±10%).
- **Traslación ortogonal (`AUG_SHIFT = 0.05`):** Oscilaciones del pulmón de máximo ±5% rellenadas con tensores de valor nulo que emulan la radiopacidad cero del entorno.
- **Volteo simétrico (`AUG_FLIP_H = True`):** Volteo horizontal fundamentado en la simetría pulmonar.

Bajo este rigor, la matriz de datos se permuta geométricamente en cada época, multiplicando la base del conocimiento sin incurrir en alteraciones biológicamente inviables o *"out-of-domain artifacts"*.

### 4.4. Arquitectura de Desacoplamiento (DeRC)
La infraestructura entera sigue la metodología `Decoupled Representation & Rebalanced Classifier`:
- **Fase 1 (Aislamiento de la Representación):** El *Backbone* extrae riqueza semántica integral en una fase masiva empleando clústeres TPU v6e. La canalización a través de la red neuronal aprovecha la estrategia de distribución dirigida por *host* (*Host-Driven Distribution*).
- **Fase 2 (Congelación y Ajuste):** Detención del esquema de gradiente general (*Freezing*), sometiendo únicamente a la "cabeza" de clasificación (`classifier_output`) a una asimilación final o ensamble logístico normalizando pesos bajo rebalanceo de clases.

## 5. Experimentación y Resultados
Se orquestaron experimentos con el objetivo de probar estas correcciones de hiperanálisis en Memoria Asíncrona bajo la estructura *TPU Distributed Strategy* sobre un clúster *TPU v6e*. La implementación en su iteración de alto rendimiento validó que, al rastrear el modelo por Macro-F1 bajo *Focal Loss*, se estabilizó asimétricamente a la red neural deteniendo entrenamientos que parecían prósperos sobre `Accuracy`, pero colapsaban en la matriz de confusión de pacientes con COVID-19.

## 6. Conclusiones y Recomendaciones Estratégicas Futuras
Las métricas macro, combinadas con una arquitectura desacoplada y una penalización algorítmica sensible al coste, dotan a esta plataforma tecnológica no solo del vigor necesario para contrarrestar los sesgos biológicos, sino también del cumplimiento formal de lineamientos MLOps en Deep Learning Médico. 

**Proyección Edge e Inferencia Activa (Active Learning)**
Como directriz futura para transicionar este backend funcional directamente a despliegues satelitales (integrando nuestra API basada en FastAPI), la infraestructura migrará paulatinamente a un **Pipeline de Aprendizaje Activo (Active Learning)**. Estructurar una lógica de muestreo por incertidumbre (*Uncertainty Sampling*) facultará a las clínicas periféricas de campo para enviar flujos difusos radiológicos de manera automatizada hacia la red en la nube principal, neutralizando gradualmente el *Data Drift* y perfeccionando orgánicamente las heurísticas resolutivas del modelo central.

## 7. Referencias
1. *Arquitecturas y estrategias avanzadas para el manejo de conjuntos de datos de imágenes desbalanceados en sistemas de visión por computadora.*
2. ANSI Z39.16-1972: *Preparation of Scientific Papers for Written or Oral Presentation* (Estructura IMRAD aplicativa).
3. Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, y Piotr Dollár. *Focal Loss for Dense Object Detection*. ICCV, 2017.
4. Kang, Bingyi, et al. *Decoupling Representation and Classifier for Long-Tailed Recognition*. ICLR, 2020.
