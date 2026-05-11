import os, glob, json, random
import numpy as np
import cv2

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Input
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR      = os.path.join(SCRIPT_DIR, 'assets')
HAND_MODEL_PATH = os.path.join(SCRIPT_DIR, 'hand_landmarker.task')

MODEL_SAVE_PATH = os.path.normpath(os.path.join(
    SCRIPT_DIR, '..', 'reconhecimento_libras', 'modelo', 'libras_alfabeto.h5'))
LABELS_SAVE_PATH = os.path.normpath(os.path.join(
    SCRIPT_DIR, '..', 'reconhecimento_libras', 'modelo', 'libras_alfabeto_labels.json'))

N_HANDS   = 2
N_LAND    = 21
N_COORDS  = 3
FEAT_DIM  = N_HANDS * N_LAND * N_COORDS  # 2*21*3 = 126

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def create_landmarker():
    base_options = mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=N_HANDS,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return mp_vision.HandLandmarker.create_from_options(options)

def extract_features_from_image(img_path: str, landmarker) -> np.ndarray:
    feat = np.zeros(FEAT_DIM, dtype=np.float32)
    frame = cv2.imread(img_path)
    if frame is not None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        try:
            result = landmarker.detect(mp_img)
            for h_idx, hand_lm in enumerate(result.hand_landmarks[:N_HANDS]):
                offset = h_idx * N_LAND * N_COORDS
                for l_idx, lm in enumerate(hand_lm):
                    base = offset + l_idx * N_COORDS
                    feat[base]     = lm.x
                    feat[base + 1] = lm.y
                    feat[base + 2] = lm.z
        except Exception:
            pass
    return feat

def load_dataset():
    rows = []
    for split in ['training', 'test']:
        split_dir = os.path.join(ASSETS_DIR, split)
        if not os.path.exists(split_dir):
            continue
        classes = os.listdir(split_dir)
        for cls in classes:
            cls_dir = os.path.join(split_dir, cls)
            if os.path.isdir(cls_dir):
                for ext in ['*.png', '*.jpg', '*.jpeg']:
                    for img_path in glob.glob(os.path.join(cls_dir, ext)):
                        rows.append({'path': img_path, 'class': cls})
    return rows

if __name__ == '__main__':
    print("=" * 60)
    print("  Treinamento LIBRAS Alfabeto (Imagens Estaticas)")
    print("=" * 60)

    rows = load_dataset()
    if not rows:
        print("Nenhuma imagem encontrada nas pastas assets/training e assets/test.")
        exit(1)

    classes_sorted = sorted(set(r['class'] for r in rows))
    class_to_idx   = {cls: i for i, cls in enumerate(classes_sorted)}
    idx_to_class   = {i: cls for cls, i in class_to_idx.items()}
    NUM_CLASSES    = len(classes_sorted)

    print(f"Imagens encontradas: {len(rows)}")
    print(f"Classes unicas:      {NUM_CLASSES}")

    landmarker = create_landmarker()

    CACHE_FILE = os.path.join(SCRIPT_DIR, 'features_alfabeto_cache.npz')
    if os.path.exists(CACHE_FILE):
        print(f"Carregando features do cache: {CACHE_FILE}")
        data = np.load(CACHE_FILE)
        X = data['X']
        y = data['y']
    else:
        print("Extraindo landmarks...")
        X_list, y_list = [], []
        for i, row in enumerate(rows):
            if i % 500 == 0:
                print(f"   [{i}/{len(rows)}] {row['class']}")
            feat = extract_features_from_image(row['path'], landmarker)
            # Ignorar imagens onde nenhuma mao foi detectada (vetor de zeros)
            if np.sum(feat) != 0:
                X_list.append(feat)
                y_list.append(class_to_idx[row['class']])

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)
        
        print("Salvando cache...")
        np.savez_compressed(CACHE_FILE, X=X, y=y)

    print(f"Shape X: {X.shape}")
    print(f"Shape y: {y.shape}")

    y_cat = to_categorical(y, num_classes=NUM_CLASSES)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=0.15, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Treino: {X_train.shape[0]} | Validacao: {X_val.shape[0]}")

    model = Sequential([
        Input(shape=(FEAT_DIM,)),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n[OK] Acuracia na validacao: {val_acc:.2%}")

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    with open(LABELS_SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(idx_to_class, f, ensure_ascii=False, indent=2)

    print(f"[SAVE] Modelo salvo em:  {MODEL_SAVE_PATH}")
    print("\nTreinamento concluido! [DONE]")
