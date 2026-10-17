import os, time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix

print("TF:", tf.__version__, "| GPU:", tf.config.list_physical_devices('GPU'))

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

print("Classes:", class_names)
print("Val batches:", n // 2, "Test batches:", n - (n // 2))

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
model = models.Model(inputs, outputs)

# Stage 1
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

callbacks1 = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.3, verbose=1),
]

t0 = time.time()
print("Starting Stage 1...")
h1 = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks1, verbose=1)
loss1, acc1 = model.evaluate(test_ds, verbose=0)
print(f"Test accuracy BEFORE fine-tuning: {acc1:.4f}")

# Stage 2
base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

callbacks2 = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.3, verbose=1),
]

print("Starting Stage 2...")
h2 = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks2, verbose=1)
loss2, acc2 = model.evaluate(test_ds, verbose=0)
total_time = time.time() - t0
print(f"Test accuracy AFTER fine-tuning: {acc2:.4f}")
print(f"Total training time: {total_time:.2f} s")

# Evaluation
y_true, y_pred = [], []
for imgs, labels in test_ds:
    p = model.predict(imgs, verbose=0)
    y_pred += list(np.argmax(p, axis=1))
    y_true += list(labels.numpy())

y_true, y_pred = np.array(y_true), np.array(y_pred)
print(classification_report(y_true, y_pred, target_names=class_names))
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)
