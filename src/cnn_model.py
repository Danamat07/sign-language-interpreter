import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# 1. Setări directoare
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "asl_alphabet", "asl_alphabet_train")
MODEL_DIR = os.path.join(BASE_DIR, "models")
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 15

# 2. Generator pentru preprocesare + augmentare
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2   # 20% pentru validare
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# 3. Construire CNN simplu
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1],3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_gen.num_classes, activation='softmax')
])

# 4. Compilare
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Antrenare
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# 6. Salvare model
os.makedirs(MODEL_DIR, exist_ok=True)
model.save(os.path.join(MODEL_DIR, "cnn_model.h5"))
print("Model salvat în folderul models/")
