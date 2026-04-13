import os
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from pathlib import Path
from src.config import DATA_DIR, CLASS_NAMES, IMG_HEIGHT, IMG_WIDTH, CHANNELS, BATCH_SIZE

def get_image_paths_and_labels():
    """
    Recorre las tres carpetas consolidadas y retorna listas de rutas y etiquetas
    """
    paths = []
    labels = []
    
    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = Path(DATA_DIR) / class_name
        if not class_dir.exists():
            continue
            
        for file in class_dir.iterdir():
            if file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                paths.append(str(file))
                labels.append(label_idx)
                
    return paths, labels

def create_stratified_splits(paths, labels, test_size=0.15, val_size=0.15):
    """
    Crea particiones train, valid, test usando particionamiento estratificado
    garantizando representacion de la clase minoritaria (COVID).
    """
    # 1. Separar Test
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_val_idx, test_idx = next(sss1.split(paths, labels))
    
    paths_train_val = [paths[i] for i in train_val_idx]
    labels_train_val = [labels[i] for i in train_val_idx]
    
    paths_test = [paths[i] for i in test_idx]
    labels_test = [labels[i] for i in test_idx]
    
    # 2. Separar Train y Val
    # Val proportions is calculated against the remaining train_val subset
    relative_val_size = val_size / (1.0 - test_size)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=42)
    
    train_idx, val_idx = next(sss2.split(paths_train_val, labels_train_val))
    
    paths_train = [paths_train_val[i] for i in train_idx]
    labels_train = [labels_train_val[i] for i in train_idx]
    paths_val = [paths_train_val[i] for i in val_idx]
    labels_val = [labels_train_val[i] for i in val_idx]
    
    return (paths_train, labels_train), (paths_val, labels_val), (paths_test, labels_test)

def tf_parse_image(file_path, label):
    """
    Decodifica y redimensiona la imagen.
    Adapta escala a 0-1. Usa escalas de grises para ahorrar VRAM.
    """
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=CHANNELS)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])

    return img, label

def get_dataloaders(params=None):
    """
    Retorna Datasets asincronos de tf.data para entrenamiento pesado en cpu 
    sin bloquear la GPU.
    """
    batch_size = params.get('batch_size', BATCH_SIZE) if params else BATCH_SIZE
    
    paths, labels = get_image_paths_and_labels()
    
    # Calculate class weights for basic setups before doing phase 2
    from collections import Counter
    counts = Counter(labels)
    total = len(labels)
    class_weights = {i: (1 / counts[i]) * (total / len(counts)) for i in range(len(counts))}
    
    (ptr, ltr), (pva, lva), (pte, lte) = create_stratified_splits(paths, labels)
    
    train_ds = tf.data.Dataset.from_tensor_slices((ptr, ltr))
    val_ds = tf.data.Dataset.from_tensor_slices((pva, lva))
    test_ds = tf.data.Dataset.from_tensor_slices((pte, lte))
    
    # Pipeline optmization
    AUTOTUNE = tf.data.AUTOTUNE
    
    train_ds = train_ds.map(tf_parse_image, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    
    val_ds = val_ds.map(tf_parse_image, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.map(tf_parse_image, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, test_ds, class_weights

if __name__ == "__main__":
    train_ds, val_ds, test_ds, cw = get_dataloaders()
    print("Class Weights Calculated:", cw)
    for images, labels in train_ds.take(1):
        print("Tensor shape:", images.shape)
        print("Labels shape:", labels.shape)
