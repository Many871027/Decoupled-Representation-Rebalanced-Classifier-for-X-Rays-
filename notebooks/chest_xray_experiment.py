# %% [markdown]
# # Chest X-Ray MLOps: Fase de Experimentación
# Este notebook implementa la lógica de Búsqueda de Hiperparámetros (Grid Search) e integra el registro
# estricto a MLflow de acuerdo a la Opción de Desarrollo "Senior CNN Custom Desacoplada".

# %%
import os
import itertools
import mlflow
import tensorflow as tf
from src.config import EXPERIMENT_NAME, MLFLOW_TRACKING_URI, BATCH_SIZE
from src.data_pipeline import get_dataloaders
from src.model_pipeline import build_custom_cnn_backbone, build_full_model, FocalLoss, set_phase_2

# Initialize MLFlow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# %%
# 1. Cargar Dataloaders (Tarda unos segundos generando particion estratificada)
train_ds, val_ds, test_ds, class_weights = get_dataloaders({'batch_size': BATCH_SIZE})
print("Datasets compilados asincronamente.")

# %%
# 2. Definir Grid Search para hiperparametros (Phase 1)
# Por restricciones de VRAM, probaremos pocas tuplas de Dropouts y Learning Rates
grid_dropouts = [0.4, 0.5]
grid_lr = [1e-3, 5e-4]

# %%
# 3. Ciclo de Experimentación MLFlow
for idx, (dropout, lr) in enumerate(itertools.product(grid_dropouts, grid_lr)):
    run_name = f"GridSearch_Run_{idx}_DR{dropout}_LR{lr}"
    
    with mlflow.start_run(run_name=run_name):
        # Log Params
        mlflow.log_params({
            "dropout_rate": dropout,
            "learning_rate": lr,
            "batch_size": BATCH_SIZE,
            "focal_gamma": 2.0,
            "focal_alpha": 0.25,
            "phase": "Phase 1 - Fully Trainable Dropout Tuning"
        })
        
        # Build Model (Phase 1: Todos los pesos actualizables)
        backbone = build_custom_cnn_backbone()
        model = build_full_model(backbone, dropout_rate=dropout)
        
        # Optimizer & Compile
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        # Using FocalLoss defined in model_pipeline Custom function
        model.compile(optimizer=optimizer, loss=FocalLoss(), metrics=['accuracy'])
        
        print(f"\n--- Iniciando Run: {run_name} ---")
        
        # Entrenamiento Fase 1 (Limitado a 3 epocas para demostracion de Grid Search rapida)
        # Para desarrollo real se recomiendan 20 epocas con EarlyStopping
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=3,
            verbose=1
        )
        
        # Log Metrics
        val_acc = history.history['val_accuracy'][-1]
        val_loss = history.history['val_loss'][-1]
        
        mlflow.log_metric("val_accuracy", val_acc)
        mlflow.log_metric("val_loss", val_loss)
        
        # Opcional: Model Signature y artefactos 
        # mlflow.tensorflow.log_model(model, artifact_path="model")
        print(f"[{run_name}] Val Acc: {val_acc:.4f} - Logged to MLflow")

# %% [markdown]
# ### Phase 2: Desacople y Fine-Tuning
# Puedes tomar el mejor modelo obtenido arriba e invocar `set_phase_2(best_model)`
# y volver a entrenar por 5 épocas extra. Los tensores de convolución estarán apagados
# ahorrando el 80% de VRAM y ajustando únicamente los Logits para el balanceo de la red.
