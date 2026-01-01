# CNN Training Notebook

### Mount Google Drive

```yaml
from google.colab import drive
drive.mount('/content/drive')
```

### Setup paths pentru modele + output

```yaml
import os

MODEL_DIR = "/content/drive/MyDrive/ASL_Project/models"
os.makedirs(MODEL_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model_best.keras")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model_final.keras")
CLASSES_PATH = os.path.join(MODEL_DIR, "classes.json")
```

### Unzip dataset (cu train/val/test)

```yaml
import zipfile, os, shutil

zip_path = "/content/drive/MyDrive/ASL_Project/asl_dataset.zip"
extract_dir = "/content/data"

if os.path.exists(extract_dir):
    shutil.rmtree(extract_dir)

os.makedirs(extract_dir, exist_ok=True)

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(extract_dir)

print("Extracted to:", extract_dir)
```

### Import + GPU + seed
Setam seed pt ca toate operatiile aleatoare sa fie controlate si reproductibile

```yaml
import tensorflow as tf
import numpy as np
import random
import os

print("TensorFlow:", tf.__version__)
print("GPU available:", len(tf.config.list_physical_devices("GPU")) > 0)

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
```

### Dataset paths

```yaml
import os

BASE_DIR = "/content/data/asl_dataset"

TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR   = os.path.join(BASE_DIR, "val")
TEST_DIR  = os.path.join(BASE_DIR, "test")
```

### Parametri

```yaml
IMG_SIZE = (128, 128)   // dimensiunea la care vor fi redimensionate imaginile
BATCH_SIZE = 16         // cate imagini procesam simultan la fiecare pas de antrenare
EPOCHS = 40             // nr max de treceri complete prin dataset
LEARNING_RATE = 1e-4    // rata de invatare pt optimizer (cat de mari sunt pasii de actualizre) - 0.0001
```

### Data Generators (train cu augmentare, val/test fara)
Pregatim img pt a fi citite si procesate in batch-uri in timpul antrenarii, validarii si testarii

```yaml
from tensorflow.keras.preprocessing.image import ImageDataGenerator

// configureaza transformarile (rotatii, deplasari, zoom, normalizare) care se vor aplica pe img de antrenament
train_datagen = ImageDataGenerator(
    rescale=1.0/255,                 // normalizam pixelii de la [0,255] la [0,1]
    rotation_range=15,               // rotim aleator img pana la +- 15 grade
    width_shift_range=0.1,           // deplasam orizontal aleator pana la 10%
    height_shift_range=0.1,          // deplasam vertical aleator pana la 10%
    zoom_range=0.1                   // zoom in/out aleator pana la 10%
)

// configureaza doar normalizarea pt img de validate si testare
eval_datagen = ImageDataGenerator(rescale=1.0/255)

// citeste img din train, aplica augumentarile, le grupeaza in batch-uri si le amesteca
train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,                   // redimensionam la 128x128
    batch_size=BATCH_SIZE,                  // 16 img per batch
    class_mode="categorical",
    shuffle=True,                           // amestecam img la fiecare epoch
    seed=SEED
)

// citeste img din val, le normalizeaza si le grupeaza in batch-uri
val_gen = eval_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False                       // nu amestecam - ordine consistenta pt evaluare
)

// citeste img din test, le normalizeaza si le grupeaza in batch-uri
test_gen = eval_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)
```

### Salvez classes.json in Drive

```yaml
import json

classes = list(train_gen.class_indices.keys())
print("Classes:", classes)

with open(CLASSES_PATH, "w") as f:
    json.dump(classes, f)

print("Saved classes to:", CLASSES_PATH)
```

### Model CNN

```yaml
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

// model secvential, straturile se adauga unul dupa altul, in ordine
model = Sequential([
    Input(shape=(128, 128, 3)),                 // forma datelor de intrare

    Conv2D(32, (3, 3), activation="relu"),      // 32 filtre de 3x3 pt detectare patten-uri simple
    BatchNormalization(),                       // normalizeaza val intre straturi
    MaxPooling2D(2, 2),                         // reduce dimensiunea marginii la jum

    Conv2D(64, (3, 3), activation="relu"),      // aplica 64 filtre pt detectare pattern-uri mai complexe
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),                                  // transforma matricea 3D in vector 1D
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(len(classes), activation="softmax")   // strat final, da probabilitai pt fiecare clasa
])

// compilam modelul - specificam cum sa invete
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),    // learning rate 0.0001
    loss="categorical_crossentropy",                // functie pierdere pt clasificare multi-clasa
    metrics=["accuracy"]                            // masuram acuratetea in timpul antrenarii
)

// afiseaza structura completa a modelului - straturi, forme, nr parametri
model.summary()
```

Output:

```yaml
 Model: "sequential"

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 126, 126, 32)   │           896 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization             │ (None, 126, 126, 32)   │           128 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 63, 63, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 61, 61, 64)     │        18,496 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_1           │ (None, 61, 61, 64)     │           256 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 30, 30, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_2 (Conv2D)               │ (None, 28, 28, 128)    │        73,856 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_2           │ (None, 28, 28, 128)    │           512 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_2 (MaxPooling2D)  │ (None, 14, 14, 128)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 25088)          │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 256)            │     6,422,784 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 256)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 24)             │         6,168 │
└─────────────────────────────────┴────────────────────────┴───────────────┘

 Total params: 6,523,096 (24.88 MB)

 Trainable params: 6,522,648 (24.88 MB)

 Non-trainable params: 448 (1.75 KB)
```

### Callbacks

```yaml
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

// callback pt salvarea automata a celui mai bun model
checkpoint = ModelCheckpoint(
    filepath=BEST_MODEL_PATH,
    monitor="val_accuracy",     // monitorizam acuratetea pe setul de validare
    save_best_only=True,        // save numai cand modelul se imbunatateste
    verbose=1
)

// callback pt oprirea anticipata, cand modelul nu se mai imbunatateste
early_stop = EarlyStopping(
    monitor="val_loss",             // monitorizam pierderea pe setul de validare
    patience=6,                     // asteptam 6 epoch fara imbunatatire inainte de oprire
    restore_best_weights=True,      // la final, restauram greutatile celui mai bun model
    verbose=1
)

// callback pt reducerea automata a learning rate, cand progresul incetineste
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",     // monitorizam pierderea pe validare
    factor=0.5,             // reducem learning rate la jum
    patience=3,             // asteptam 3 epoch fara imbunatatire
    min_lr=1e-6,            // learning rate min = 0.000001
    verbose=1
)
```

### Training

```yaml
// antrenam modelul si salvam istoricul (pierderi, acuratete per epoca)
history = model.fit(
    train_gen,                  // datele de antrenament (cu augmentare)
    validation_data=val_gen,    // datele de validate (fara augmentare)
    epochs=EPOCHS,              // nar max epochs = 40
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)
```

Output:

```yaml
Epoch 1/40
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 0s 89ms/step - accuracy: 0.2013 - loss: 2.7536
Epoch 1: val_accuracy improved from -inf to 0.64010, saving model to /content/drive/MyDrive/ASL_Project/models/cnn_model_best.keras
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 572s 93ms/step - accuracy: 0.2014 - loss: 2.7536 - val_accuracy: 0.6401 - val_loss: 1.1312 - learning_rate: 1.0000e-04
Epoch 2/40
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 0s 83ms/step - accuracy: 0.4877 - loss: 1.5913
Epoch 2: val_accuracy improved from 0.64010 to 0.76578, saving model to /content/drive/MyDrive/ASL_Project/models/cnn_model_best.keras
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 526s 87ms/step - accuracy: 0.4877 - loss: 1.5912 - val_accuracy: 0.7658 - val_loss: 0.7188 - learning_rate: 1.0000e-04
Epoch 3/40
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 0s 82ms/step - accuracy: 0.6310 - loss: 1.1164
Epoch 3: val_accuracy improved from 0.76578 to 0.85370, saving model to /content/drive/MyDrive/ASL_Project/models/cnn_model_best.keras
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 519s 86ms/step - accuracy: 0.6310 - loss: 1.1164 - val_accuracy: 0.8537 - val_loss: 0.4380 - learning_rate: 1.0000e-04
Epoch 4/40
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 0s 82ms/step - accuracy: 0.7222 - loss: 0.8285
Epoch 4: val_accuracy improved from 0.85370 to 0.92240, saving model to /content/drive/MyDrive/ASL_Project/models/cnn_model_best.keras
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 519s 86ms/step - accuracy: 0.7222 - loss: 0.8285 - val_accuracy: 0.9224 - val_loss: 0.2569 - learning_rate: 1.0000e-04
Epoch 5/40
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 0s 82ms/step - accuracy: 0.7821 - loss: 0.6523
Epoch 5: val_accuracy improved from 0.92240 to 0.93600, saving model to /content/drive/MyDrive/ASL_Project/models/cnn_model_best.keras
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 518s 86ms/step - accuracy: 0.7821 - loss: 0.6523 - val_accuracy: 0.9360 - val_loss: 0.1890 - learning_rate: 1.0000e-04
Epoch 6/40
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 0s 82ms/step - accuracy: 0.8176 - loss: 0.5391
Epoch 6: val_accuracy improved from 0.93600 to 0.95667, saving model to /content/drive/MyDrive/ASL_Project/models/cnn_model_best.keras
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 516s 86ms/step - accuracy: 0.8176 - loss: 0.5391 - val_accuracy: 0.9567 - val_loss: 0.1459 - learning_rate: 1.0000e-04
Epoch 7/40
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 0s 81ms/step - accuracy: 0.8473 - loss: 0.4502
Epoch 7: val_accuracy improved from 0.95667 to 0.96626, saving model to /content/drive/MyDrive/ASL_Project/models/cnn_model_best.keras
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 511s 85ms/step - accuracy: 0.8474 - loss: 0.4502 - val_accuracy: 0.9663 - val_loss: 0.1073 - learning_rate: 1.0000e-04
Epoch 8/40
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 0s 82ms/step - accuracy: 0.8693 - loss: 0.3937
Epoch 8: val_accuracy improved from 0.96626 to 0.97599, saving model to /content/drive/MyDrive/ASL_Project/models/cnn_model_best.keras
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 513s 85ms/step - accuracy: 0.8693 - loss: 0.3937 - val_accuracy: 0.9760 - val_loss: 0.0781 - learning_rate: 1.0000e-04
Epoch 9/40
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 0s 82ms/step - accuracy: 0.8802 - loss: 0.3602
Epoch 9: val_accuracy did not improve from 0.97599
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 514s 85ms/step - accuracy: 0.8802 - loss: 0.3602 - val_accuracy: 0.9709 - val_loss: 0.0911 - learning_rate: 1.0000e-04
Epoch 10/40
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 0s 82ms/step - accuracy: 0.8930 - loss: 0.3238
Epoch 10: val_accuracy improved from 0.97599 to 0.98480, saving model to /content/drive/MyDrive/ASL_Project/models/cnn_model_best.keras
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 513s 85ms/step - accuracy: 0.8930 - loss: 0.3238 - val_accuracy: 0.9848 - val_loss: 0.0534 - learning_rate: 1.0000e-04
Epoch 11/40
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 0s 82ms/step - accuracy: 0.9064 - loss: 0.2892
Epoch 11: val_accuracy did not improve from 0.98480
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 515s 85ms/step - accuracy: 0.9064 - loss: 0.2892 - val_accuracy: 0.9750 - val_loss: 0.0715 - learning_rate: 1.0000e-04
Epoch 12/40
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 0s 81ms/step - accuracy: 0.9123 - loss: 0.2715
Epoch 12: val_accuracy did not improve from 0.98480
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 511s 85ms/step - accuracy: 0.9123 - loss: 0.2715 - val_accuracy: 0.9814 - val_loss: 0.0587 - learning_rate: 1.0000e-04
Epoch 13/40
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 0s 82ms/step - accuracy: 0.9179 - loss: 0.2590
Epoch 13: val_accuracy did not improve from 0.98480

Epoch 13: ReduceLROnPlateau reducing learning rate to 4.999999873689376e-05.
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 514s 85ms/step - accuracy: 0.9179 - loss: 0.2589 - val_accuracy: 0.9833 - val_loss: 0.0619 - learning_rate: 1.0000e-04
Epoch 14/40
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 0s 82ms/step - accuracy: 0.9344 - loss: 0.1997
Epoch 14: val_accuracy improved from 0.98480 to 0.99283, saving model to /content/drive/MyDrive/ASL_Project/models/cnn_model_best.keras
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 514s 85ms/step - accuracy: 0.9344 - loss: 0.1997 - val_accuracy: 0.9928 - val_loss: 0.0302 - learning_rate: 5.0000e-05
Epoch 15/40
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 0s 81ms/step - accuracy: 0.9422 - loss: 0.1748
Epoch 15: val_accuracy improved from 0.99283 to 0.99516, saving model to /content/drive/MyDrive/ASL_Project/models/cnn_model_best.keras
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 512s 85ms/step - accuracy: 0.9422 - loss: 0.1748 - val_accuracy: 0.9952 - val_loss: 0.0177 - learning_rate: 5.0000e-05
Epoch 16/40
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 0s 82ms/step - accuracy: 0.9458 - loss: 0.1710
Epoch 16: val_accuracy improved from 0.99516 to 0.99569, saving model to /content/drive/MyDrive/ASL_Project/models/cnn_model_best.keras
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 519s 86ms/step - accuracy: 0.9458 - loss: 0.1710 - val_accuracy: 0.9957 - val_loss: 0.0143 - learning_rate: 5.0000e-05
Epoch 17/40
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 0s 82ms/step - accuracy: 0.9526 - loss: 0.1512
Epoch 17: val_accuracy did not improve from 0.99569
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 515s 85ms/step - accuracy: 0.9526 - loss: 0.1512 - val_accuracy: 0.9935 - val_loss: 0.0190 - learning_rate: 5.0000e-05
Epoch 18/40
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 0s 81ms/step - accuracy: 0.9525 - loss: 0.1491
Epoch 18: val_accuracy improved from 0.99569 to 0.99748, saving model to /content/drive/MyDrive/ASL_Project/models/cnn_model_best.keras
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 509s 84ms/step - accuracy: 0.9525 - loss: 0.1491 - val_accuracy: 0.9975 - val_loss: 0.0103 - learning_rate: 5.0000e-05
Epoch 19/40
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 0s 79ms/step - accuracy: 0.9567 - loss: 0.1436
Epoch 19: val_accuracy did not improve from 0.99748
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 551s 83ms/step - accuracy: 0.9567 - loss: 0.1436 - val_accuracy: 0.9963 - val_loss: 0.0140 - learning_rate: 5.0000e-05
Epoch 20/40
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 0s 79ms/step - accuracy: 0.9563 - loss: 0.1438
Epoch 20: val_accuracy did not improve from 0.99748
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 494s 82ms/step - accuracy: 0.9563 - loss: 0.1438 - val_accuracy: 0.9964 - val_loss: 0.0144 - learning_rate: 5.0000e-05
Epoch 21/40
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 0s 79ms/step - accuracy: 0.9591 - loss: 0.1311
Epoch 21: val_accuracy did not improve from 0.99748

Epoch 21: ReduceLROnPlateau reducing learning rate to 2.499999936844688e-05.
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 498s 83ms/step - accuracy: 0.9591 - loss: 0.1311 - val_accuracy: 0.9943 - val_loss: 0.0188 - learning_rate: 5.0000e-05
Epoch 22/40
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 0s 78ms/step - accuracy: 0.9637 - loss: 0.1134
Epoch 22: val_accuracy did not improve from 0.99748
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 492s 82ms/step - accuracy: 0.9637 - loss: 0.1134 - val_accuracy: 0.9966 - val_loss: 0.0138 - learning_rate: 2.5000e-05
Epoch 23/40
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 0s 79ms/step - accuracy: 0.9678 - loss: 0.1068
Epoch 23: val_accuracy improved from 0.99748 to 0.99787, saving model to /content/drive/MyDrive/ASL_Project/models/cnn_model_best.keras
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 499s 83ms/step - accuracy: 0.9678 - loss: 0.1068 - val_accuracy: 0.9979 - val_loss: 0.0095 - learning_rate: 2.5000e-05
Epoch 24/40
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 0s 79ms/step - accuracy: 0.9680 - loss: 0.1043
Epoch 24: val_accuracy did not improve from 0.99787
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 497s 82ms/step - accuracy: 0.9680 - loss: 0.1043 - val_accuracy: 0.9978 - val_loss: 0.0087 - learning_rate: 2.5000e-05
Epoch 25/40
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 0s 79ms/step - accuracy: 0.9699 - loss: 0.0989
Epoch 25: val_accuracy improved from 0.99787 to 0.99831, saving model to /content/drive/MyDrive/ASL_Project/models/cnn_model_best.keras
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 497s 82ms/step - accuracy: 0.9699 - loss: 0.0989 - val_accuracy: 0.9983 - val_loss: 0.0069 - learning_rate: 2.5000e-05
Epoch 26/40
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 0s 79ms/step - accuracy: 0.9690 - loss: 0.1007
Epoch 26: val_accuracy did not improve from 0.99831
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 496s 82ms/step - accuracy: 0.9690 - loss: 0.1007 - val_accuracy: 0.9975 - val_loss: 0.0100 - learning_rate: 2.5000e-05
Epoch 27/40
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 0s 79ms/step - accuracy: 0.9712 - loss: 0.0944
Epoch 27: val_accuracy improved from 0.99831 to 0.99845, saving model to /content/drive/MyDrive/ASL_Project/models/cnn_model_best.keras
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 496s 82ms/step - accuracy: 0.9712 - loss: 0.0944 - val_accuracy: 0.9985 - val_loss: 0.0073 - learning_rate: 2.5000e-05
Epoch 28/40
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 0s 79ms/step - accuracy: 0.9703 - loss: 0.0948
Epoch 28: val_accuracy did not improve from 0.99845

Epoch 28: ReduceLROnPlateau reducing learning rate to 1.249999968422344e-05.
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 494s 82ms/step - accuracy: 0.9703 - loss: 0.0948 - val_accuracy: 0.9978 - val_loss: 0.0091 - learning_rate: 2.5000e-05
Epoch 29/40
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 0s 79ms/step - accuracy: 0.9735 - loss: 0.0873
Epoch 29: val_accuracy improved from 0.99845 to 0.99864, saving model to /content/drive/MyDrive/ASL_Project/models/cnn_model_best.keras
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 496s 82ms/step - accuracy: 0.9735 - loss: 0.0873 - val_accuracy: 0.9986 - val_loss: 0.0051 - learning_rate: 1.2500e-05
Epoch 30/40
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 0s 79ms/step - accuracy: 0.9755 - loss: 0.0780
Epoch 30: val_accuracy did not improve from 0.99864
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 496s 82ms/step - accuracy: 0.9755 - loss: 0.0780 - val_accuracy: 0.9984 - val_loss: 0.0062 - learning_rate: 1.2500e-05
Epoch 31/40
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 0s 79ms/step - accuracy: 0.9748 - loss: 0.0825
Epoch 31: val_accuracy did not improve from 0.99864
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 495s 82ms/step - accuracy: 0.9748 - loss: 0.0825 - val_accuracy: 0.9979 - val_loss: 0.0078 - learning_rate: 1.2500e-05
Epoch 32/40
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 0s 79ms/step - accuracy: 0.9774 - loss: 0.0749
Epoch 32: val_accuracy did not improve from 0.99864

Epoch 32: ReduceLROnPlateau reducing learning rate to 6.24999984211172e-06.
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 499s 83ms/step - accuracy: 0.9774 - loss: 0.0749 - val_accuracy: 0.9986 - val_loss: 0.0053 - learning_rate: 1.2500e-05
Epoch 33/40
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 0s 79ms/step - accuracy: 0.9773 - loss: 0.0718
Epoch 33: val_accuracy did not improve from 0.99864
6028/6028 ━━━━━━━━━━━━━━━━━━━━ 498s 83ms/step - accuracy: 0.9773 - loss: 0.0718 - val_accuracy: 0.9980 - val_loss: 0.0074 - learning_rate: 6.2500e-06
Epoch 34/40
3387/6028 ━━━━━━━━━━━━━━━━━━━━ 3:29 79ms/step - accuracy: 0.9787 - loss: 0.0706
```

### Runtime Interrupted
-> progres salvat in Google Drive

**Load best model**
```yaml
from tensorflow.keras.models import load_model

model = load_model(
    "/content/drive/MyDrive/ASL_Project/models/cnn_model_best.keras"
)

model.summary()
```

Output:

```yaml
 Model: "sequential"

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 126, 126, 32)   │           896 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization             │ (None, 126, 126, 32)   │           128 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 63, 63, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 61, 61, 64)     │        18,496 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_1           │ (None, 61, 61, 64)     │           256 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 30, 30, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_2 (Conv2D)               │ (None, 28, 28, 128)    │        73,856 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_2           │ (None, 28, 28, 128)    │           512 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_2 (MaxPooling2D)  │ (None, 14, 14, 128)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 25088)          │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 256)            │     6,422,784 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 256)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 24)             │         6,168 │
└─────────────────────────────────┴────────────────────────┴───────────────┘

 Total params: 19,568,394 (74.65 MB)

 Trainable params: 6,522,648 (24.88 MB)

 Non-trainable params: 448 (1.75 KB)

 Optimizer params: 13,045,298 (49.76 MB)
```

### Evaluate Model on TEST

```yaml
test_loss, test_acc = model.evaluate(test_gen, verbose=1)
print("FINAL TEST ACCURACY:", test_acc)
```

Output:

```yaml
1294/1294 ━━━━━━━━━━━━━━━━━━━━ 298s 230ms/step - accuracy: 0.9986 - loss: 0.0047
FINAL TEST ACCURACY: 0.9982601404190063
```

### Confusion Matrix

**Predictions on TEST**

```yaml
import numpy as np
from sklearn.metrics import confusion_matrix

test_gen.reset()

y_pred_probs = model.predict(test_gen, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

y_true = test_gen.classes
class_names = list(test_gen.class_indices.keys())
```

### Absolute Confusion Matrix

```yaml
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(14, 12))
plt.imshow(cm)
plt.title("Confusion Matrix – TEST SET", fontsize=16)
plt.xlabel("Predicted label")
plt.ylabel("True label")

plt.xticks(range(len(class_names)), class_names, rotation=45)
plt.yticks(range(len(class_names)), class_names)

plt.colorbar()

for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(
            j, i,
            cm[i, j],
            ha="center",
            va="center",
            color="white" if cm[i, j] > cm.max() * 0.6 else "black",
            fontsize=7
        )

plt.tight_layout()
plt.show()
```

![alt text](/models/confusion_matrix/absolute-confusion-matrix.png)

### Normalized Confusion Matrix

```yaml
cm_norm = confusion_matrix(y_true, y_pred, normalize="true")

plt.figure(figsize=(14, 12))
plt.imshow(cm_norm)
plt.title("Normalized Confusion Matrix – TEST SET", fontsize=16)
plt.xlabel("Predicted label")
plt.ylabel("True label")

plt.xticks(range(len(class_names)), class_names, rotation=45)
plt.yticks(range(len(class_names)), class_names)

plt.colorbar(label="Proportion")

for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(
            j, i,
            f"{cm_norm[i, j]:.2f}",
            ha="center",
            va="center",
            color="white" if cm_norm[i, j] > 0.6 else "black",
            fontsize=7
        )

plt.tight_layout()
plt.show()
```

![alt text](/models/confusion_matrix/normalized-confusion-matrix.png)
