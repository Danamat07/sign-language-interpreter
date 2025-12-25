## CNN Training Notebook - Execution Log
Documentation of the complete training process executed in Google Colab

---

### Mount Google Drive

  ```yaml
from google.colab import drive
drive.mount('/content/drive')
  ```

### Model Storage Setup

  ```yaml
import os

MODEL_DIR = "/content/drive/MyDrive/ASL_Project/models"
os.makedirs(MODEL_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model_best.keras")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model_final.keras")

print("Models will be saved to:", MODEL_DIR)
  ```

### Unzip Dataset

  ```yaml
import zipfile
import os

zip_path = '/content/drive/MyDrive/ASL_Project/asl_dataset.zip'

if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('/content/data')
  ```

### Import Libraries and Check GPU

  ```yaml
import tensorflow as tf
import os
import numpy as np

print("TensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")
print("GPU available:", len(gpus) > 0)

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass
  ```

Output:

  ```yaml
TensorFlow version: 2.19.0
GPU available: True
  ```

### Dataset Paths

  ```yaml
BASE_DIR = "data/asl_dataset"

TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
  ```

### Experiment Parameters

  ```yaml
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 0.0001
  ```

### Data Generators (train + test)

  ```yaml
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)
  ```

Output:

  ```yaml
Found 110216 images belonging to 24 classes.
Found 27566 images belonging to 24 classes.
  ```

### Saving Classes

  ```yaml
import json

classes = list(train_gen.class_indices.keys())

with open("classes.json", "w") as f:
    json.dump(classes, f)
  ```

### CNN Architecture from Scratch

  ```yaml
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

model = Sequential()

model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation="softmax"))

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
  ```

Output:

  ```yaml
 /usr/local/lib/python3.12/dist-packages/keras/src/layers/convolutional/base_conv.py:113: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
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

### Callbacks (early stopping + best model)

  ```yaml
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

checkpoint = ModelCheckpoint(
    filepath=BEST_MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)
  ```

### Model Training

  ```yaml
history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=10,
    callbacks=[checkpoint, early_stop],
    verbose=1
)
  ```

Training results:

  ```yaml
Epoch     Train Acc     Train Loss      Val Acc     Val Loss      Status

1/10      0.2152        2.6690          0.6875      0.9591        Saved
2/10      0.5355        1.4197          0.8420      0.4742        Saved
3/10      0.6784        0.9623          0.9075      0.2962        Saved
4/10      0.7614        0.7074          0.9511      0.1727        Saved
5/10      0.8101        0.5599          0.9601      0.1221        Saved
6/10      0.8478        0.4538          0.9654      0.1074        Saved
7/10      0.8711        0.3866          0.9819      0.0735        Best
8/10      0.8870        0.3389          0.9681      0.1151        No improvement
9/10      0.9027        0.2987          0.9768      0.0768        No improvement
10/10     0.9077        0.2563            -           -           Interrupted
  ```

### Runtime Interruption Recovery
The Colab runtime disconnected during epoch 10. All progress was preserved in Google Drive.

**Load Best Model**

  ```yaml
from tensorflow.keras.models import load_model

model = load_model("/content/drive/MyDrive/ASL_Project/models/cnn_model_best.keras")
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

**Reload Test Data**

  ```yaml
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = "data/asl_dataset"
TEST_DIR = os.path.join(BASE_DIR, "test")

IMG_SIZE = (128, 128)
BATCH_SIZE = 16

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)
  ```

Output:

  ```yaml
Found 27566 images belonging to 24 classes.
  ```

### Model Evaluation

**Final Test Accuracy**

  ```yaml
test_loss, test_acc = model.evaluate(test_gen)
print("Final test accuracy:", test_acc)
  ```

Output:

  ```yaml
1723/1723 ━━━━━━━━━━━━━━━━━━━━ 37s 20ms/step - accuracy: 0.9869 - loss: 0.0512
Final test accuracy: 0.9819342494010925
  ```

### Confusion Matrix

**Generate Predictions**

  ```yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

test_gen.reset()

y_pred_probs = model.predict(test_gen, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

y_true = test_gen.classes

cm = confusion_matrix(y_true, y_pred)

class_names = list(test_gen.class_indices.keys())
  ```

**Absolute Confusion Matrix**

  ```yaml
plt.figure(figsize=(14, 12))
plt.imshow(cm)
plt.title("Confusion Matrix – ASL CNN Model", fontsize=16)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(range(len(class_names)), class_names, rotation=45)
plt.yticks(range(len(class_names)), class_names)

plt.colorbar()

for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(
            j, i, cm[i, j],
            ha="center", va="center",
            color="white" if cm[i, j] > cm.max() * 0.6 else "black",
            fontsize=8
        )

plt.tight_layout()
plt.show()
  ```

![alt text](/src/matrix/absolute-confusion-matrix.png)

**Normalized Confusion Matrix**

  ```yaml
cm_normalized = confusion_matrix(y_true, y_pred, normalize="true")

plt.figure(figsize=(14, 12))
plt.imshow(cm_normalized)
plt.title("Normalized Confusion Matrix – ASL CNN Model", fontsize=16)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(range(len(class_names)), class_names, rotation=45)
plt.yticks(range(len(class_names)), class_names)

plt.colorbar(label="Proportion")

for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(
            j, i,
            f"{cm_normalized[i, j]:.2f}",
            ha="center",
            va="center",
            color="white" if cm_normalized[i, j] > 0.6 else "black",
            fontsize=8
        )

plt.tight_layout()
plt.show()
  ```

![alt text](/src/matrix/normalized-confusion-matrix.png)

### Save Final Model

  ```yaml
model.save("/content/drive/MyDrive/ASL_Project/models/cnn_model.keras")
  ```