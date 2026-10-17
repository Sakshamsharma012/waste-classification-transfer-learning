import os, time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

data_dir = "trashnet_data/dataset-resized"
IMG = (224, 224)
BS = 32
SEED = 42

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.3, subset="training", seed=SEED,
    image_size=IMG, batch_size=BS)
rest_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.3, subset="validation", seed=SEED,
    image_size=IMG, batch_size=BS)

class_names = train_ds.class_names
num_classes = len(class_names)
n = tf.data.experimental.cardinality(rest_ds).numpy()

val_ds = rest_ds.take(n // 2).cache().prefetch(tf.data.AUTOTUNE)
test_ds = rest_ds.skip(n // 2).cache().prefetch(tf.data.AUTOTUNE)
train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)

def build_model():
    aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
    ], name="augmentation")

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG + (3,), include_top=False, weights="imagenet")
    base_model.trainable = False

    inputs = layers.Input(IMG + (3,))
    x = aug(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs), base_model

# Run Experiment 1: Stage 2 lr=3e-5, patience=4
print("--- Experiment 1: Stage 2 lr=3e-5, patience=4 ---")
model, base = build_model()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
cb1 = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.3, verbose=0),
]
model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=cb1, verbose=0)
loss1, acc1 = model.evaluate(test_ds, verbose=0)
print(f"Exp 1 Before FT: {acc1:.4f}")

base.trainable = True
for layer in base.layers[:-40]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(3e-5),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
cb2 = [
    tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.3, verbose=0),
]
model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=cb2, verbose=0)
loss2, acc2 = model.evaluate(test_ds, verbose=0)
print(f"Exp 1 After FT (lr=3e-5, patience=4): {acc2:.4f}")

# Run Experiment 2: Stage 2 lr=5e-5, top 50 layers, patience=4
print("--- Experiment 2: Stage 2 lr=5e-5, top 50 layers, patience=4 ---")
model, base = build_model()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=cb1, verbose=0)
loss1_2, acc1_2 = model.evaluate(test_ds, verbose=0)
base.trainable = True
for layer in base.layers[:-50]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(5e-5),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=cb2, verbose=0)
loss2_2, acc2_2 = model.evaluate(test_ds, verbose=0)
print(f"Exp 2 After FT (lr=5e-5, top 50 layers): {acc2_2:.4f}")
