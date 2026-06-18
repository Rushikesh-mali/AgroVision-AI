import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json

IMG_SIZE = 224
BATCH_SIZE = 32
DATASET_PATH = "dataset"

# =====================================
# DATA AUGMENTATION
# =====================================
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# =====================================
# SAVE CLASS MAPPING
# =====================================
print("\n🔥 Class Mapping:")
print(train_data.class_indices)

with open("class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f, indent=4)

print("✅ class_indices.json saved successfully")

# =====================================
# LOAD MOBILENETV2
# =====================================
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze base layers initially
base_model.trainable = False

# ====================================
# BUILD MODEL
# =====================================
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(train_data.num_classes, activation="softmax")
])

# =====================================
# PHASE 1 TRAINING
# =====================================
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    patience=2,
    factor=0.3,
    verbose=1
)

print("\n🚀 Phase 1 Training Started")

model.fit(
    train_data,
    validation_data=val_data,
    epochs=5,
    callbacks=[early_stop, reduce_lr]
)


# =====================================
# SAVE MODEL
# =====================================
model.save("agrovision_model.h5")

print("\n🎉 Training Completed Successfully!")
print("✅ Model Saved -> agrovision_model.h5")
print("✅ Class Mapping Saved -> class_indices.json")
print(f"✅ Total Classes Trained -> {train_data.num_classes}")