"""
Treina um modelo para reconhecimento de LIBRAS usando o dataset V-LIBRASIL.
Pipeline:
  1. Lê os vídeos MP4 do dataset.
  2. Extrai landmarks de mãos via MediaPipe HandLandmarker em N frames amostrados.
  3. Agrega os landmarks em um vetor de características (média temporal).
  4. Treina um MLP (Dense) para classificar os sinais.
  5. Salva o modelo em reconhecimento_libras/modelo/libras_vlibrasil.h5
     e o mapeamento de labels em reconhecimento_libras/modelo/libras_vlibrasil_labels.json
Uso:
  cd app
  python train_vlibrasil_model.py
Opções ajustáveis no topo do script:
  FRAMES_PER_VIDEO  – quantos frames amostrar por vídeo (default 10)
  MIN_VIDEOS_CLASS  – mínimo de vídeos por classe para incluí-la (default 2)
  MAX_CLASSES       – limitar a N classes mais frequentes (None = todas)
"""
import os, csv, json, random, math
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
DATASET_DIR     = os.path.join(SCRIPT_DIR, 'assets', 'videos UFPE (V-LIBRASIL)', 'data')
ANNOTATIONS_CSV = os.path.join(SCRIPT_DIR, 'assets', 'videos UFPE (V-LIBRASIL)', 'annotations.csv')
HAND_MODEL_PATH = os.path.join(SCRIPT_DIR, 'hand_landmarker.task')
MODEL_SAVE_PATH = os.path.normpath(os.path.join(
    SCRIPT_DIR, '..', 'reconhecimento_libras', 'modelo', 'libras_vlibrasil.h5'))
LABELS_SAVE_PATH = os.path.normpath(os.path.join(
    SCRIPT_DIR, '..', 'reconhecimento_libras', 'modelo', 'libras_vlibrasil_labels.json'))
FRAMES_PER_VIDEO  = 10    
MIN_VIDEOS_CLASS  = 2     
MAX_CLASSES       = None  
N_HANDS   = 2
N_LAND    = 21
N_COORDS  = 3
FEAT_DIM  = FRAMES_PER_VIDEO * N_HANDS * N_LAND * N_COORDS  
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
def extract_features(video_path: str, landmarker) -> np.ndarray:
    """
    Retorna array de forma (FRAMES_PER_VIDEO, N_HANDS*N_LAND*N_COORDS).
    Frames sem mãos detectadas ficam zerados.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    indices = sorted(set(
        int(i * (total - 1) / max(FRAMES_PER_VIDEO - 1, 1))
        for i in range(FRAMES_PER_VIDEO)
    ))
    while len(indices) < FRAMES_PER_VIDEO:
        indices.append(indices[-1])
    indices = indices[:FRAMES_PER_VIDEO]
    frame_feats = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        feat = np.zeros(N_HANDS * N_LAND * N_COORDS, dtype=np.float32)
        if ret and frame is not None:
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
        frame_feats.append(feat)
    cap.release()
    return np.stack(frame_feats, axis=0)  
def load_annotations():
    rows = []
    with open(ANNOTATIONS_CSV, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            video_path = os.path.join(DATASET_DIR, r['video_name'])
            if os.path.exists(video_path):
                rows.append({'path': video_path, 'class': r['class']})
    return rows
def filter_classes(rows, min_videos=2, max_classes=None):
    from collections import Counter
    counts = Counter(r['class'] for r in rows)
    valid = {cls for cls, cnt in counts.items() if cnt >= min_videos}
    rows = [r for r in rows if r['class'] in valid]
    if max_classes and len(valid) > max_classes:
        top = [cls for cls, _ in counts.most_common(max_classes) if cls in valid]
        valid = set(top)
        rows = [r for r in rows if r['class'] in valid]
    return rows
if __name__ == '__main__':
    print("=" * 60)
    print("  Treinamento V-LIBRASIL")
    print("=" * 60)
    print("\n[1/6] Carregando anotações...")
    rows = load_annotations()
    rows = filter_classes(rows, MIN_VIDEOS_CLASS, MAX_CLASSES)
    classes_sorted = sorted(set(r['class'] for r in rows))
    class_to_idx   = {cls: i for i, cls in enumerate(classes_sorted)}
    idx_to_class   = {i: cls for cls, i in class_to_idx.items()}
    NUM_CLASSES    = len(classes_sorted)
    print(f"   Vídeos encontrados:  {len(rows)}")
    print(f"   Classes únicas:      {NUM_CLASSES}")
    print("\n[2/6] Inicializando MediaPipe HandLandmarker...")
    landmarker = create_landmarker()
    CACHE_FILE = os.path.join(SCRIPT_DIR, 'features_cache.npz')
    if os.path.exists(CACHE_FILE):
        print(f"\n[3/6] Carregando features do cache: {CACHE_FILE}")
        data = np.load(CACHE_FILE)
        X = data['X']
        y = data['y']
    else:
        print(f"\n[3/6] Extraindo landmarks de {len(rows)} vídeos "
              f"({FRAMES_PER_VIDEO} frames cada)...")
        print("      Isso pode demorar vários minutos.\n")
        X_list, y_list = [], []
        for i, row in enumerate(rows):
            if i % 100 == 0:
                print(f"   [{i}/{len(rows)}] {os.path.basename(row['path'])}")
            feats = extract_features(row['path'], landmarker)  
            X_list.append(feats.flatten())                     
            y_list.append(class_to_idx[row['class']])
        X = np.array(X_list, dtype=np.float32)  
        y = np.array(y_list, dtype=np.int32)
        print("Salvando features em cache para rodadas futuras...")
        np.savez_compressed(CACHE_FILE, X=X, y=y)
    print(f"\n   Shape X: {X.shape}")
    print(f"   Shape y: {y.shape}")
    print("\n[4/6] Dividindo treino/validação (85%/15%)...")
    y_cat = to_categorical(y, num_classes=NUM_CLASSES)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=0.15, random_state=RANDOM_SEED
    )
    print(f"   Treino: {X_train.shape[0]} | Validação: {X_val.shape[0]}")
    print("\n[5/6] Construindo e treinando modelo MLP...")
    model = Sequential([
        Input(shape=(FEAT_DIM,)),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax'),
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=4, min_lr=1e-6, verbose=1),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n[OK] Acurácia na validação: {val_acc:.2%}")
    print(f"   Loss na validação:    {val_loss:.4f}")
    print("\n[6/6] Salvando modelo e mapeamento de labels...")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    with open(LABELS_SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(idx_to_class, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVE] Modelo salvo em:  {MODEL_SAVE_PATH}")
    print(f"   Labels salvo em: {LABELS_SAVE_PATH}")
    print(f"   Input shape:     {model.input_shape}")
    print(f"   Output shape:    {model.output_shape}")
    print(f"   Número de classes: {NUM_CLASSES}")
    print(f"\n   FEAT_DIM = {FRAMES_PER_VIDEO} frames × {N_HANDS} mãos × "
          f"{N_LAND} landmarks × {N_COORDS} coords = {FEAT_DIM}")
    print("\nTreinamento concluído! [DONE]")