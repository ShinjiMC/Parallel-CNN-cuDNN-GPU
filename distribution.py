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

# Iterar sobre cada tipo de flor y mover imágenes a los directorios correspondientes
for flower_type in flower_types:
    # Obtener la lista de archivos en el directorio actual
    files = os.listdir(os.path.join(flowers_dir, flower_type))
    random.shuffle(files)  # Mezclar aleatoriamente los archivos

    # Calcular el número de archivos para el conjunto de prueba (20%)
    num_files = len(files)
    num_test = int(0.2 * num_files)

    # Crear las carpetas correspondientes para cada tipo de flor en TRAIN_DIR, TEST_DIR y VAL_DIR
    os.makedirs(os.path.join(TRAIN_DIR, flower_type), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, flower_type), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, flower_type), exist_ok=True)

    # Mover/copiar los archivos a los directorios correspondientes
    for i, file in enumerate(files):
        src = os.path.join(flowers_dir, flower_type, file)
        
        # Copiar todas las imágenes a Train y Validate
        dst_train = os.path.join(TRAIN_DIR, flower_type, file)
        dst_val = os.path.join(VAL_DIR, flower_type, file)
        shutil.copy2(src, dst_train)
        shutil.copy2(src, dst_val)

        # Mover el 20% de las imágenes a Test
        if i < num_test:
            dst_test = os.path.join(TEST_DIR, flower_type, file)
            shutil.copy2(src, dst_test)

print("Proceso completado. Archivos copiados a los directorios TRAIN_DIR, TEST_DIR y VAL_DIR.")