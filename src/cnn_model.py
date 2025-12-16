"""
CNN training for ASL Alphabet (A-Z only)
Dataset: Kaggle ASL Alphabet
Enhanced version with augmentations and callbacks.
"""

import os
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Paths & Settings
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Folder must contain only A-Z folders
DATA_DIR = os.path.join(BASE_DIR, "data", "asl_alphabet", "asl_alphabet_train")
MODEL_DIR = os.path.join(BASE_DIR, "models")

IMG_SIZE = (128, 128)  # increased size for better accuracy
BATCH_SIZE = 32
EPOCHS = 30  # can adjust depending on hardware

# Data Generator with Augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.8,1.2],
    horizontal_flip=True  # helps with mirrored signs
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# Classes
class_indices = train_gen.class_indices
idx_to_class = {v: k for k, v in class_indices.items()}
CLASSES = [idx_to_class[i] for i in range(len(idx_to_class))]

print("\nClasses used for training:")
print(CLASSES)
print("Total classes:", len(CLASSES))
assert len(CLASSES) == 26, "Dataset must contain exactly 26 classes (A-Z)"

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(len(CLASSES), activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Callbacks
os.makedirs(MODEL_DIR, exist_ok=True)

checkpoint = ModelCheckpoint(
    os.path.join(MODEL_DIR, "cnn_model_best.h5"),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

earlystop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Train
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint, earlystop],
    verbose=1
)

# Save final model and classes
model.save(os.path.join(MODEL_DIR, "cnn_model.h5"))

with open(os.path.join(MODEL_DIR, "classes.json"), "w") as f:
    json.dump(CLASSES, f)

print("\nModel and classes saved successfully")
