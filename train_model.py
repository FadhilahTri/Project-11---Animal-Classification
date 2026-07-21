"""
Studi Kasus: Klasifikasi Jenis Hewan menggunakan Transfer Learning VGG16
Dataset: Animals-10 (Kaggle - alessiocorrado99/animals10)
"""

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ============================================================
# 1. KONFIGURASI DASAR
# ============================================================
DATASET_DIR = "dataset/raw-img"   # sesuaikan path ke folder dataset kamu
IMG_SIZE = (128, 128)              # dikecilkan dari 224x224 supaya komputasi jauh lebih ringan di CPU
BATCH_SIZE = 32
EPOCHS = 3                         # dikurangi supaya lebih cepat

# Batasi jumlah batch per epoch supaya tidak perlu proses SELURUH data tiap epoch.
# Ini mempercepat training secara signifikan di CPU, dengan sedikit trade-off akurasi.
STEPS_PER_EPOCH = 50    # ~50 x 32 = 1600 gambar per epoch
VALIDATION_STEPS = 15   # ~15 x 32 = 480 gambar untuk validasi

# Nama folder di dataset Animals-10 memakai Bahasa Italia,
# berikut terjemahannya ke Bahasa Indonesia untuk label yang lebih jelas
class_translation = {
    "cane": "Anjing",
    "cavallo": "Kuda",
    "elefante": "Gajah",
    "farfalla": "Kupu-kupu",
    "gallina": "Ayam",
    "gatto": "Kucing",
    "mucca": "Sapi",
    "pecora": "Domba",
    "ragno": "Laba-laba",
    "scoiattolo": "Tupai"
}

# ============================================================
# 2. PERSIAPAN DATASET (dengan augmentasi + split train/val)
# ============================================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2   # 80% train, 20% validation
)

train_data = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_data = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Ambil nama kelas sesuai urutan index dari generator, lalu terjemahkan
folder_names = list(train_data.class_indices.keys())
class_names = [class_translation.get(name, name) for name in folder_names]
print("Kelas:", class_names)

# ============================================================
# 3. MEMBANGUN MODEL DENGAN TRANSFER LEARNING VGG16
# ============================================================
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

# Membekukan seluruh layer VGG16 (feature extraction)
for layer in base_model.layers:
    layer.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')   # 10 kelas
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================================
# 4. TRAINING MODEL
# ============================================================
checkpoint = ModelCheckpoint(
    "model_animals10_vgg16.h5",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
    callbacks=[checkpoint, early_stop]
)

# ============================================================
# 5. VISUALISASI HASIL TRAINING (Akurasi & Loss)
# ============================================================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Akurasi Model')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Model')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("training_history.png")
plt.show()

# ============================================================
# 6. EVALUASI MODEL (Confusion Matrix & Classification Report)
# ============================================================
val_data.reset()
y_pred_prob = model.predict(val_data)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = val_data.classes

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Prediksi")
plt.ylabel("Label Asli")
plt.title("Confusion Matrix - Klasifikasi Hewan (VGG16)")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ============================================================
# 7. SIMPAN MODEL FINAL (untuk dipakai di aplikasi Flask nanti)
# ============================================================
model.save("model_animals10_vgg16_final.h5")
print("\nModel berhasil disimpan sebagai 'model_animals10_vgg16_final.h5'")
print("File ini yang akan dipakai di aplikasi web Flask.")
