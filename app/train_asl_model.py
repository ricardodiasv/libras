"""
Treina uma CNN para o Sign Language MNIST (ASL letras A-Z, sem J e Z).
Dataset: sign_mnist_train.csv (CSV com label + 784 pixels grayscale 28x28)
Salva o modelo treinado em reconhecimento_libras/modelo/asl_model.h5
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization, Input
)
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ── Caminhos ──────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, 'assets', 'sign_mnist_train.csv')
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, '..', 'reconhecimento_libras', 'modelo', 'asl_model.h5')
MODEL_SAVE_PATH = os.path.normpath(MODEL_SAVE_PATH)

NUM_CLASSES = 24   # A-Y sem J(9) e Z(25)

# ── Carregar dados ────────────────────────────────────────────
print(f"Carregando dataset de {CSV_PATH} ...")
df = pd.read_csv(CSV_PATH)

labels = df['label'].values
pixels = df.drop('label', axis=1).values.astype('float32') / 255.0
pixels = pixels.reshape(-1, 28, 28, 1)

# Remapear labels: os labels originais pulam o 9 (J) e 25 (Z)
# Precisamos mapear para 0-23 contíguos
unique_labels = sorted(set(labels))
label_map = {old: new for new, old in enumerate(unique_labels)}
mapped_labels = np.array([label_map[l] for l in labels])
labels_cat = to_categorical(mapped_labels, num_classes=NUM_CLASSES)

print(f"Total de amostras: {len(labels)}")
print(f"Labels únicos: {unique_labels}")
print(f"Shape pixels: {pixels.shape}")

# ── Split treino/validação ────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    pixels, labels_cat, test_size=0.15, random_state=42, stratify=mapped_labels
)
print(f"Treino: {X_train.shape[0]} | Validação: {X_val.shape[0]}")

# ── Modelo CNN ────────────────────────────────────────────────
model = Sequential([
    Input(shape=(28, 28, 1)),

    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax'),
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ── Callbacks ─────────────────────────────────────────────────
early_stop = EarlyStopping(
    monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
)

# ── Treino ────────────────────────────────────────────────────
print("\nIniciando treinamento...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=128,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)

# ── Avaliação ─────────────────────────────────────────────────
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\n✅ Acurácia na validação: {val_acc:.2%}")
print(f"   Loss na validação:    {val_loss:.4f}")

# ── Salvar modelo ─────────────────────────────────────────────
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"\n💾 Modelo salvo em: {MODEL_SAVE_PATH}")
print(f"   Input shape:  {model.input_shape}")
print(f"   Output shape: {model.output_shape}")

# Salvar mapa de labels para referência
label_to_letter = {}
alphabet = 'ABCDEFGHIKLMNOPQRSTUVWXY'  # sem J e Z
for i, letter in enumerate(alphabet):
    label_to_letter[i] = letter
print(f"\n   Mapa de labels: {label_to_letter}")
