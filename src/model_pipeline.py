import tensorflow as tf
from tensorflow.keras import layers, models
from src.config import NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH, CHANNELS, FOCAL_GAMMA, FOCAL_ALPHA, AUG_ROTATION, AUG_ZOOM, AUG_SHIFT, AUG_FLIP_H
import numpy as np
from sklearn.metrics import classification_report, f1_score

def build_custom_cnn_backbone(input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)):
    """
    Construye la arquitectura base convolucional (Feature Extractor).
    Se usa padding = 'same' y BatchNormalization para estabilidad.
    """
    inputs = tf.keras.Input(shape=input_shape)

    # --- Medical-Safe Augmentation Block ---
    # This block only operates during training
    x = inputs
    if AUG_FLIP_H:
        x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(AUG_ROTATION)(x)
    x = layers.RandomZoom(AUG_ZOOM)(x)
    x = layers.RandomTranslation(height_shift=AUG_SHIFT, width_shift=AUG_SHIFT)(x)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    # Block 2
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    # Global pooling is lighter than Flatten+Dense
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    
    # Creating a Keras Model for the Backbone guarantees it can be frozen later
    backbone = tf.keras.Model(inputs, x, name="cnn_backbone")
    return backbone

def build_full_model(backbone, dropout_rate=0.5):
    """
    Acopla el clasificador Dense linear al backbone.
    """
    inputs = backbone.input
    x = backbone.output
    
    # Classifier Head
    x = layers.Dense(128, activation='relu', name='classifier_dense_1')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', name='classifier_output')(x)
    
    model = tf.keras.Model(inputs, outputs, name="chest_xray_model")
    return model

class FocalLoss(tf.keras.losses.Loss):
    """
    Custom Focal Loss (Sparse Categorical CrossEntropy base).
    Penaliza errores en clases dificiles (como la clase minoritaria COVID).
    """
    def __init__(self, gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        # Aseguramos que y_pred no sea exactamente 0 ni 1
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

        # Sparse categorical encoding
        cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

        # Gather probabilities of the true classes
        # create one-hot representation to multiply
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=NUM_CLASSES)
        p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)

        # Focal Loss formula: -alpha * (1 - p_t)**gamma * log(p_t)
        # Using already calculated cross entropy which is -log(p_t)
        focal_loss = self.alpha * tf.pow((1.0 - p_t), self.gamma) * cross_entropy
        return tf.reduce_mean(focal_loss)

class MedicalReportCallback(tf.keras.callbacks.Callback):
    """
    Genera un reporte de clasificación detallado al final de cada época
    usando el dataset de validación.
    """
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        # Recolectar predicciones y etiquetas reales
        all_labels = []
        all_preds = []

        for images, labels in self.validation_data:
            preds = self.model.predict(images, verbose=0)
            all_preds.extend(np.argmax(preds, axis=1))
            all_labels.extend(labels.numpy() if hasattr(labels, 'numpy') else labels)

        # Generar reporte de sklearn
        from src.config import CLASS_NAMES
        report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES)

        macro_f1 = f1_score(all_labels, all_preds, average='macro')

        print(f"\n--- Medical Evaluation Report (Epoch {epoch+1}) ---")
        print(report)
        print(f"Macro-F1 Score: {macro_f1:.4f}")
        print("-" * 45)

        # Log to Keras logs for MLFlow tracking
        if logs is not None:
            logs['val_f1_score'] = macro_f1

def set_phase_2(model):
    """
    Congela (Freezes) el backbone para la fase 2.
    """
    backbone = model.get_layer("cnn_backbone")
    backbone.trainable = False
    print("Backbone congelado. Configurado para Phase 2 Training.")
    return model
