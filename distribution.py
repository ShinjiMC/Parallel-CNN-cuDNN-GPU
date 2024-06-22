import os
import shutil
import random

# Directorio donde se encuentran las carpetas de cada tipo de flor
flowers_dir = './flowers'

# Obtener la lista de subdirectorios (cada uno representa un tipo de flor)
flower_types = os.listdir(flowers_dir)

# Contar la cantidad de imágenes en cada subdirectorio
counts = {}
for flower_type in flower_types:
    count = len(os.listdir(os.path.join(flowers_dir, flower_type)))
    counts[flower_type] = count

# Mostrar la cantidad de imágenes por tipo de flor
for flower_type, count in counts.items():
    print(f'{flower_type}: {count} imágenes')

# Definir las rutas de los directorios de entrenamiento, prueba y validación
TRAIN_DIR = "./Train"
TEST_DIR = "./Test"
VAL_DIR = "./Validate"

# Crear los directorios si no existen
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# Porcentaje de datos para entrenamiento, prueba y validación
train_percent = 0.7
test_percent = 0.2
val_percent = 0.1

# Iterar sobre cada tipo de flor y mover imágenes a los directorios correspondientes
for flower_type in flower_types:
    # Obtener la lista de archivos en el directorio actual
    files = os.listdir(os.path.join(flowers_dir, flower_type))
    random.shuffle(files)  # Mezclar aleatoriamente los archivos

    # Calcular los índices para los conjuntos de entrenamiento, prueba y validación
    num_files = len(files)
    num_train = int(train_percent * num_files)
    num_test = int(test_percent * num_files)
    num_val = num_files - num_train - num_test

    # Crear las carpetas correspondientes para cada tipo de flor en TRAIN_DIR, TEST_DIR y VAL_DIR
    os.makedirs(os.path.join(TRAIN_DIR, flower_type), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, flower_type), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, flower_type), exist_ok=True)

    # Mover los archivos a los directorios correspondientes
    for i, file in enumerate(files):
        src = os.path.join(flowers_dir, flower_type, file)
        if i < num_train:
            dst = os.path.join(TRAIN_DIR, flower_type, file)
        elif i < num_train + num_test:
            dst = os.path.join(TEST_DIR, flower_type, file)
        else:
            dst = os.path.join(VAL_DIR, flower_type, file)
        shutil.move(src, dst)

print("Proceso completado. Archivos movidos a los directorios TRAIN_DIR, TEST_DIR y VAL_DIR.")
