import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🚀 Chest X-Ray MLOps Pipeline In-Memory (Google Colab Optimized)\n",
    "Este notebook ejecuta el pipeline de arquitectura desacoplada de Chest X-Ray. Dado que los sub-procesos impiden leer `history` de entrenamiento localmente, este Notebook implementará toda la lógica `Focal Loss` y compilación en Memoria asíncrona, capturando el mejor macro-F1 y volcando el modelo en .keras hacia tu Drive personal."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Montaje de Google Drive e Infraestructura\n",
    "from google.colab import drive\n",
    "import os\n",
    "\n",
    "drive.mount('/content/drive')\n",
    "\n",
    "# Test riguroso de Persistencia (Backup Zone)\n",
    "DRIVE_BACKUP_PATH = '/content/drive/MyDrive/ChestXRayProject'\n",
    "if not os.path.exists(DRIVE_BACKUP_PATH):\n",
    "    print(f\"\\u26A0\\uFE0F Directorio de backup {DRIVE_BACKUP_PATH} ausente. Creándolo...\")\n",
    "    os.makedirs(DRIVE_BACKUP_PATH, exist_ok=True)\n",
    "\n",
    "print(\"\\u2705 Drive montado y verificado. Modelos guardados aquí NO seran borrados al desconectar Colab.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2. Clonación y Actualización de Dependencias\n",
    "!rm -rf /content/project\n",
    "!git clone https://github.com/Many871027/Decoupled-Representation-Rebalanced-Classifier-for-X-Rays- /content/project\n",
    "%cd /content/project\n",
    "\n",
    "!pip install fastapi uvicorn pydantic scikit-learn pillow pandas tabulate > /dev/null\n",
    "print(\"\\u2705 Repositorio clonado y Dependencias Instaladas Nativamente.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3. Inyección Dinámica de Configuraciones (Medical-Safe & VRAM Optimizado)\n",
    "import sys\n",
    "import tensorflow as tf\n",
    "\n",
    "os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'\n",
    "gpus = tf.config.list_physical_devices('GPU')\n",
    "if gpus:\n",
    "    for gpu in gpus:\n",
    "        tf.config.experimental.set_memory_growth(gpu, True)\n",
    "\n",
    "sys.path.append('/content/project')\n",
    "import src.config as config\n",
    "\n",
    "# Limitantes Físicos de tu Hardware\n",
    "config.IMG_HEIGHT = 512\n",
    "config.IMG_WIDTH = 512\n",
    "config.BATCH_SIZE = 32\n",
    "\n",
    "# Vinculación hacia carpeta unificada local del Colab Data\n",
    "print(\"Por favor asegurate que tus imágenes estén montadas o extraídas. Asumiremos que están en /content/project/chest_xray o usarémos una ruta demo.\")\n",
    "DATA_TARGET = '/content/drive/MyDrive/ChestXRayProject/chest_xray'\n",
    "if os.path.exists(DATA_TARGET):\n",
    "    config.DATA_DIR = os.path.abspath(DATA_TARGET)\n",
    "    print(\"\\u2705 Chest X-Ray Data enlazada desde Google Drive\")\n",
    "else:\n",
    "    print(\"\\u26A0\\uFE0F Faltan las Imágenes en Drive. Usando la ruta relativa clonada.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 4. DataLoading Inteligente\n",
    "from src.data_pipeline import get_dataloaders\n",
    "try:\n",
    "    train_ds, val_ds, test_ds, class_weights = get_dataloaders({'batch_size': config.BATCH_SIZE})\n",
    "    print(\"\\u2705 Dataloaders Montados Estratificadamente.\")\n",
    "except Exception as e:\n",
    "    print(f\"\\u274C Error en Dataloaders (Probablemente no hay data en la ruta): {e}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 5. Grid Search Memory Workflow (Phase 1 Training)\n",
    "import itertools\n",
    "import pandas as pd\n",
    "from src.model_pipeline import build_custom_cnn_backbone, build_full_model, FocalLoss, MedicalReportCallback\n",
    "\n",
    "grid_dropouts = [0.4, 0.5]\n",
    "grid_lr = [1e-3, 5e-4]\n",
    "\n",
    "experiment_results = []\n",
    "best_overall_f1 = 0.0\n",
    "best_model_run = None\n",
    "\n",
    "for dropout, lr in itertools.product(grid_dropouts, grid_lr):\n",
    "    print(f\"\\n{'='*50}\\n\\u25B6\\uFE0F Iniciando Experimentaci\\u00f3n: Dropout={dropout} | LR={lr}\\n{'='*50}\")\n",
    "    \n",
    "    backbone = build_custom_cnn_backbone()\n",
    "    model = build_full_model(backbone, dropout_rate=dropout)\n",
    "    \n",
    "    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)\n",
    "    medical_report = MedicalReportCallback(val_ds)\n",
    "    \n",
    "    early_stopping = tf.keras.callbacks.EarlyStopping(\n",
    "        monitor='val_f1_score',\n",
    "        mode='max',\n",
    "        patience=8,\n",
    "        restore_best_weights=True,\n",
    "        verbose=1\n",
    "    )\n",
    "    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)\n",
    "\n",
    "    model.compile(optimizer=optimizer, loss=FocalLoss(), metrics=['accuracy'])\n",
    "    \n",
    "    history = model.fit(\n",
    "        train_ds,\n",
    "        validation_data=val_ds,\n",
    "        epochs=50,\n",
    "        callbacks=[lr_reducer, medical_report, early_stopping],\n",
    "        verbose=1\n",
    "    )\n",
    "    \n",
    "    # Extracci\\u00f3n asincrona nativa\n",
    "    val_acc = max(history.history['val_accuracy']) if 'val_accuracy' in history.history else 0\n",
    "    val_f1 = max(history.history.get('val_f1_score', [0]))\n",
    "    \n",
    "    # Evaluador del mejor modelo de todos los tiempos\n",
    "    is_best_run = False\n",
    "    if val_f1 > best_overall_f1:\n",
    "        best_overall_f1 = val_f1\n",
    "        best_model_run = model\n",
    "        is_best_run = True\n",
    "        print(f\"\\u2B50 Nuevo Campe\\u00f3n! F1: {val_f1:.4f}\")\n",
    "\n",
    "    experiment_results.append({\n",
    "        'Dropout': dropout,\n",
    "        'Learning Rate': lr,\n",
    "        'Best Epoch F1': round(val_f1, 4),\n",
    "        'Best Epoch Accuracy': round(val_acc, 4),\n",
    "        'Loss Alpha': 0.5,\n",
    "        'Is Best Run': is_best_run\n",
    "    })\n",
    "\n",
    "print(\"\\u2705 Ciclo Grid Search completado con \\u00e9xito.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## \ud83c\udfc6 Tabla Comparativa y Serialización\n",
    "El siguiente script genera la tabla de Pandas reportando las m\u00e9tricas y graba perpetuamente al ganador usando `.keras` directo a tu Drive."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from IPython.display import display\n",
    "\n",
    "df_results = pd.DataFrame(experiment_results)\n",
    "df_results = df_results.sort_values(by='Best Epoch F1', ascending=False).reset_index(drop=True)\n",
    "\n",
    "print(\"\\n\\n\" + \"*\"*60)\n",
    "print(\"\\uD83C\\uDFDF\\uFE0F AN\\u00C1LISIS COMPARATIVO DE HIPERPAR\\u00C1METROS (Top to Bottom) \\uD83C\\uDFDF\\uFE0F\")\n",
    "print(\"*\"*60 + \"\\n\")\n",
    "\n",
    "display(df_results.style.background_gradient(cmap='Blues', subset=['Best Epoch F1']))\n",
    "\n",
    "if best_model_run:\n",
    "    TARGET_SAVE = f\"{DRIVE_BACKUP_PATH}/Model_F1_{best_overall_f1:.4f}.keras\"\n",
    "    print(f\"\\n\\u26A1\\uFE0F Grabando pesos del modelo ganador directamente en tu Google Drive protegido...\")\n",
    "    best_model_run.save(TARGET_SAVE)\n",
    "    print(f\"\\u2705 MODELO PERSISTIDO EXITOSAMENTE a: {TARGET_SAVE}\")\n",
    "else:\n",
    "    print(\"\\u274C Fallo al extraer el modelo ganador.\")"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "gpuType": "T4",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {"name": "ipython", "version": 3},
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open("d:/2do4triMINAR/Final-Algoritmos/colab_runner.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
