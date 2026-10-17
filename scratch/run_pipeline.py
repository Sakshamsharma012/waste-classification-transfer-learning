import os, sys, time, json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell, new_output
from PIL import Image
import io, base64

print("Starting end-to-end Waste Classification execution...")
print(f"TensorFlow: {tf.__version__} | Keras: {tf.keras.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")

# Step 1: Ensure dataset unzipped cleanly
data_dest = "trashnet_data"
data_dir = os.path.join(data_dest, "dataset-resized")
zip_path = "dataset-resized.zip"

if not os.path.exists(data_dir):
    print(f"Extracting {zip_path}...")
    import zipfile, shutil
    os.makedirs(data_dest, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(data_dest)
    for root, dirs, files in os.walk(data_dest, topdown=False):
        for f in files:
            if f.startswith('._') or f == '.DS_Store':
                try: os.remove(os.path.join(root, f))
                except: pass
        for d in dirs:
            if d == '__MACOSX':
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)

# Confirm counts
classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
print(f"Classes ({len(classes)}): {classes}")
class_counts = {}
for c in classes:
    cf = os.path.join(data_dir, c)
    cnt = len([f for f in os.listdir(cf) if os.path.isfile(os.path.join(cf, f))])
    class_counts[c] = cnt
    print(f"  {c:12s}: {cnt}")
print(f"Total: {sum(class_counts.values())}")

# Step 2: Data splits
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
n_rest = tf.data.experimental.cardinality(rest_ds).numpy()
val_batches = n_rest // 2
test_batches = n_rest - val_batches

AUTOTUNE = tf.data.AUTOTUNE
val_ds = rest_ds.take(val_batches).cache().prefetch(AUTOTUNE)
test_ds = rest_ds.skip(val_batches).cache().prefetch(AUTOTUNE)
train_ds = train_ds.cache().prefetch(AUTOTUNE)

print(f"Train batches: {tf.data.experimental.cardinality(train_ds).numpy()}, Val batches: {val_batches}, Test batches: {test_batches}")

# Class distribution plot
train_counts = np.zeros(num_classes, dtype=int)
for _, y in train_ds:
    for label in y.numpy():
        train_counts[int(label)] += 1

plt.figure(figsize=(8, 4))
sns.barplot(x=class_names, y=train_counts, hue=class_names, palette="viridis", legend=False)
plt.title("Images per Class (Training Set)", fontsize=12)
plt.xlabel("Class", fontsize=10)
plt.ylabel("Number of Images", fontsize=10)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=200)
plt.close()
print("Saved class_distribution.png")

# Sample images plot
plt.figure(figsize=(10, 5))
for imgs, labels in train_ds.take(1):
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.imshow(imgs[i].numpy().astype("uint8"))
        plt.title(f"{class_names[int(labels[i])]}", fontsize=10)
        plt.axis("off")
plt.tight_layout()
plt.savefig("sample_images.png", dpi=200)
plt.close()
print("Saved sample_images.png")

# Step 3: Model construction
aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2),
], name="augmentation")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG + (3,), include_top=False, weights="imagenet")
base_model.trainable = False

inputs = layers.Input(shape=IMG + (3,))
x = aug(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs, name="waste_mobilenetv2")
model.summary()

# Step 4: Stage 1 training
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

cb1 = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.3, verbose=1),
]

t0_s1 = time.time()
print("\n--- Training Stage 1 (Frozen Base) ---")
h1 = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=cb1, verbose=1)
time_s1 = time.time() - t0_s1
loss1, acc1 = model.evaluate(test_ds, verbose=0)
print(f"Stage 1 Time: {time_s1:.2f} s")
print(f"Test accuracy BEFORE fine-tuning: {acc1:.4f} (loss: {loss1:.4f})")

# Step 5: Stage 2 training (Fine-tuning top 40 layers)
base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

print(f"\nTrainable layers in base model: {sum(1 for l in base_model.layers if l.trainable)}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

cb2 = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.3, verbose=1),
]

t0_s2 = time.time()
print("\n--- Training Stage 2 (Fine-Tuning Top 40 Layers) ---")
h2 = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=cb2, verbose=1)
time_s2 = time.time() - t0_s2
total_time = time_s1 + time_s2
loss2, acc2 = model.evaluate(test_ds, verbose=0)
print(f"Stage 2 Time: {time_s2:.2f} s")
print(f"Total training time: {total_time:.2f} s")
print(f"Test accuracy AFTER fine-tuning: {acc2:.4f} (loss: {loss2:.4f})")

# Step 6: Learning curves
def merge_hist(a, b, k):
    return a.history[k] + b.history[k]

acc = merge_hist(h1, h2, "accuracy")
val_acc = merge_hist(h1, h2, "val_accuracy")
loss = merge_hist(h1, h2, "loss")
val_loss = merge_hist(h1, h2, "val_loss")
split_epoch = len(h1.history["accuracy"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

ax1.plot(range(1, len(acc) + 1), acc, label="Train Accuracy", marker='o', markersize=3)
ax1.plot(range(1, len(val_acc) + 1), val_acc, label="Val Accuracy", marker='s', markersize=3)
ax1.axvline(split_epoch + 0.5, ls="--", color="gray", label="Fine-tuning start")
ax1.set_title("Training & Validation Accuracy", fontsize=12)
ax1.set_xlabel("Epoch", fontsize=10)
ax1.set_ylabel("Accuracy", fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.legend(loc="lower right")

ax2.plot(range(1, len(loss) + 1), loss, label="Train Loss", marker='o', markersize=3)
ax2.plot(range(1, len(val_loss) + 1), val_loss, label="Val Loss", marker='s', markersize=3)
ax2.axvline(split_epoch + 0.5, ls="--", color="gray", label="Fine-tuning start")
ax2.set_title("Training & Validation Loss", fontsize=12)
ax2.set_xlabel("Epoch", fontsize=10)
ax2.set_ylabel("Loss", fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.legend(loc="upper right")

plt.tight_layout()
plt.savefig("training_curves.png", dpi=200)
plt.close()
print("Saved training_curves.png")

# Step 7: Test Evaluation
y_true = []
y_pred = []
for imgs, labels in test_ds:
    preds = model.predict(imgs, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1).tolist())
    y_true.extend(labels.numpy().tolist())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

clf_rep = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print("\nClassification Report:\n", clf_rep)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            cbar=True)
plt.title("Confusion Matrix (Test Set)", fontsize=12)
plt.xlabel("Predicted Class", fontsize=10)
plt.ylabel("True Class", fontsize=10)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200)
plt.close()
print("Saved confusion_matrix.png")

# Step 8: Misclassified Examples
wrong = []
for imgs, labels in test_ds:
    preds = np.argmax(model.predict(imgs, verbose=0), axis=1)
    for im, t, q in zip(imgs, labels.numpy(), preds):
        if t != q:
            wrong.append((im.numpy().astype("uint8"), int(t), int(q)))
    if len(wrong) >= 8:
        break

plt.figure(figsize=(14, 6))
for i, (im, t, q) in enumerate(wrong[:8]):
    plt.subplot(2, 4, i + 1)
    plt.imshow(im)
    plt.title(f"True: {class_names[t]}\nPred: {class_names[q]}", fontsize=9, color="darkred")
    plt.axis("off")
plt.suptitle("Sample Misclassified Test Images", fontsize=13)
plt.tight_layout()
plt.savefig("wrong_predictions.png", dpi=200)
plt.close()
print("Saved wrong_predictions.png")

# Step 9: Save model
model.save("waste_model.keras")
model_size = os.path.getsize("waste_model.keras") / (1024 * 1024)
print(f"Saved waste_model.keras ({model_size:.2f} MB)")

# Step 10: Gradio demo snapshot
sample_test_img = None
for imgs, labels in test_ds.take(1):
    sample_test_img = imgs[0].numpy().astype("uint8")
    sample_true_label = class_names[int(labels[0])]
    break

# Run prediction
img_tensor = tf.convert_to_tensor(sample_test_img, dtype=tf.float32)
img_resized = tf.image.resize(img_tensor, IMG)[None]
p_sample = model.predict(img_resized, verbose=0)[0]
top3_idx = np.argsort(p_sample)[::-1][:3]
top3_classes = [class_names[i] for i in top3_idx]
top3_probs = [float(p_sample[i]) for i in top3_idx]

# Render Gradio demo UI mock visualization
fig, (ax_im, ax_bar) = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={'width_ratios': [1, 1.3]})
ax_im.imshow(sample_test_img)
ax_im.set_title(f"Input Image (Ground Truth: {sample_true_label})", fontsize=11)
ax_im.axis("off")

y_pos = np.arange(len(top3_classes))
ax_bar.barh(y_pos, top3_probs, color=['#2b5c8f', '#4682b4', '#87ceeb'])
ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(top3_classes, fontsize=11)
ax_bar.invert_yaxis()
ax_bar.set_xlim(0, 1.0)
ax_bar.set_xlabel("Confidence Score", fontsize=10)
ax_bar.set_title("Gradio Output: Top-3 Predicted Classes", fontsize=11)
for i, v in enumerate(top3_probs):
    ax_bar.text(v + 0.02, i, f"{v*100:.1f}%", va='center', fontweight='bold', fontsize=10)
plt.tight_layout()
plt.savefig("gradio_demo.png", dpi=200)
plt.close()
print("Saved gradio_demo.png")

# Save all metrics to json for notebook embedding
results = {
    "acc_before": float(acc1),
    "loss_before": float(loss1),
    "acc_after": float(acc2),
    "loss_after": float(loss2),
    "time_s1": float(time_s1),
    "time_s2": float(time_s2),
    "total_time": float(total_time),
    "model_size_mb": float(model_size),
    "classification_report": clf_rep,
    "class_counts": class_counts,
    "confusion_matrix": cm.tolist()
}
with open("scratch/run_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nAll pipeline execution and assets successfully completed!")
