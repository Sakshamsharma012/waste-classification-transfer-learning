import time
import tensorflow as tf
from tensorflow.keras import layers, models

IMG = (224, 224)
BS = 32
SEED = 42
data_dir = "trashnet_data/dataset-resized"

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.3, subset="training", seed=SEED,
    image_size=IMG, batch_size=BS)
rest_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.3, subset="validation", seed=SEED,
    image_size=IMG, batch_size=BS)

n = tf.data.experimental.cardinality(rest_ds).numpy()
val_ds = rest_ds.take(n // 2)
test_ds = rest_ds.skip(n // 2)

train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

# Benchmark EfficientNetB0
print("Testing EfficientNetB0 1 epoch...")
base_eff = tf.keras.applications.EfficientNetB0(input_shape=IMG + (3,), include_top=False, weights="imagenet")
base_eff.trainable = False
inp = layers.Input(IMG + (3,))
x = tf.keras.applications.efficientnet.preprocess_input(inp)
x = base_eff(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(6, activation="softmax")(x)
m_eff = models.Model(inp, out)
m_eff.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

t0 = time.time()
m_eff.fit(train_ds, validation_data=val_ds, epochs=1, verbose=1)
t1 = time.time()
print(f"EfficientNetB0 1 epoch time: {t1 - t0:.2f} s")
