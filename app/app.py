import cv2
import json
import numpy as np
import streamlit as st
from keras.models import load_model
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import tempfile
import os
import urllib.request
import av
from collections import deque

current_dir = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL_PATH = os.path.join(current_dir, 'hand_landmarker.task')

if not os.path.exists(HAND_MODEL_PATH):
    HAND_MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )
    with st.spinner("Baixando modelo de detecção de mãos..."):
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)

# ──────────────────────────────────────────────────────────────
# HandLandmarker
# ──────────────────────────────────────────────────────────────
def create_hand_landmarker_image():
    base_options = mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)

def create_hand_landmarker_video():
    base_options = mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)

# ──────────────────────────────────────────────────────────────
# Modelos de classificação
# ──────────────────────────────────────────────────────────────

# --- V-LIBRASIL (palavras em português - novo dataset) ---
VLIBRASIL_MODEL_PATH  = os.path.normpath(os.path.join(
    current_dir, '..', 'reconhecimento_libras', 'modelo', 'libras_vlibrasil.h5'))
VLIBRASIL_LABELS_PATH = os.path.normpath(os.path.join(
    current_dir, '..', 'reconhecimento_libras', 'modelo', 'libras_vlibrasil_labels.json'))

# Parâmetros de extração de features (deve coincidir com train_vlibrasil_model.py)
FRAMES_PER_VIDEO = 10
N_HANDS   = 2
N_LAND    = 21
N_COORDS  = 3
FEAT_DIM  = FRAMES_PER_VIDEO * N_HANDS * N_LAND * N_COORDS  # 1260

vlibrasil_model  = None
VLIBRASIL_LABELS = {}

if os.path.exists(VLIBRASIL_MODEL_PATH) and os.path.exists(VLIBRASIL_LABELS_PATH):
    vlibrasil_model = load_model(VLIBRASIL_MODEL_PATH)
    with open(VLIBRASIL_LABELS_PATH, encoding='utf-8') as f:
        raw = json.load(f)
        VLIBRASIL_LABELS = {int(k): v for k, v in raw.items()}
else:
    st.warning(
        "⚠️ Modelo V-LIBRASIL não encontrado. "
        "Execute `python train_vlibrasil_model.py` na pasta `app/` para treinar."
    )

# --- LIBRAS Alfabeto (Imagens Estáticas) ---
ALFABETO_MODEL_PATH = os.path.normpath(os.path.join(
    current_dir, '..', 'reconhecimento_libras', 'modelo', 'libras_alfabeto.h5'))
ALFABETO_LABELS_PATH = os.path.normpath(os.path.join(
    current_dir, '..', 'reconhecimento_libras', 'modelo', 'libras_alfabeto_labels.json'))

alfabeto_model = None
ALFABETO_LABELS = {}
if os.path.exists(ALFABETO_MODEL_PATH) and os.path.exists(ALFABETO_LABELS_PATH):
    alfabeto_model = load_model(ALFABETO_MODEL_PATH)
    with open(ALFABETO_LABELS_PATH, encoding='utf-8') as f:
        raw = json.load(f)
        ALFABETO_LABELS = {int(k): v for k, v in raw.items()}


# --- LIBRAS legado (palavras em inglês - modelo antigo) ---
libras_path = os.path.normpath(os.path.join(
    current_dir, '..', 'reconhecimento_libras', 'modelo', 'NewModel.h5'))
libras_model = None
LIBRAS_LABELS = {
    0: 'bus', 1: 'bank', 2: 'car', 3: 'formation', 4: 'hospital',
    5: 'I', 6: 'man', 7: 'motorcycle', 8: 'my', 9: 'supermarket',
    10: 'we', 11: 'woman', 12: 'you', 13: 'you (plural)', 14: 'your'
}
if os.path.exists(libras_path):
    libras_model = load_model(libras_path)

# --- ASL (letras A-Y, sem J e Z) ---
asl_path = os.path.normpath(os.path.join(
    current_dir, '..', 'reconhecimento_libras', 'modelo', 'asl_model.h5'))
asl_model = None
ASL_LABELS = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P',
    15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W',
    22: 'X', 23: 'Y'
}
if os.path.exists(asl_path):
    asl_model = load_model(asl_path)

# ──────────────────────────────────────────────────────────────
# Buffer de landmarks para V-LIBRASIL (janela deslizante)
# ──────────────────────────────────────────────────────────────
class LandmarkBuffer:
    """Mantém os últimos FRAMES_PER_VIDEO vetores de landmarks para predição."""
    def __init__(self):
        self.buffer = deque(maxlen=FRAMES_PER_VIDEO)

    def add(self, hand_landmarks_list):
        feat = np.zeros(N_HANDS * N_LAND * N_COORDS, dtype=np.float32)
        for h_idx, hand_lm in enumerate(hand_landmarks_list[:N_HANDS]):
            offset = h_idx * N_LAND * N_COORDS
            for l_idx, lm in enumerate(hand_lm):
                base = offset + l_idx * N_COORDS
                feat[base]     = lm.x
                feat[base + 1] = lm.y
                feat[base + 2] = lm.z
        self.buffer.append(feat)

    def get_feature_vector(self):
        """Retorna vetor (FEAT_DIM,) com padding à esquerda se necessário."""
        frames = list(self.buffer)
        while len(frames) < FRAMES_PER_VIDEO:
            frames.insert(0, np.zeros(N_HANDS * N_LAND * N_COORDS, dtype=np.float32))
        return np.concatenate(frames[:FRAMES_PER_VIDEO])

    def ready(self):
        return len(self.buffer) >= FRAMES_PER_VIDEO // 2


# ──────────────────────────────────────────────────────────────
# Pré-processamento e predição
# ──────────────────────────────────────────────────────────────
def _preprocess_libras(hand_roi_bgr):
    """Resize para (213, 120, 3) com padding, BGR→RGB."""
    TARGET_H, TARGET_W = 213, 120
    src = cv2.cvtColor(hand_roi_bgr, cv2.COLOR_BGR2RGB)
    h, w = src.shape[:2]
    scale = min(TARGET_H / h, TARGET_W / w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(src, (nw, nh))
    canvas = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
    y0, x0 = (TARGET_H - nh) // 2, (TARGET_W - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas.astype('float32') / 255.0

def _preprocess_asl(hand_roi_bgr):
    """Resize para (28, 28, 1) grayscale."""
    gray = cv2.cvtColor(hand_roi_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    return resized.astype('float32').reshape(28, 28, 1) / 255.0

def predict_from_roi(hand_roi_bgr, model_choice, confidence_threshold=0.70):
    """Predição baseada em ROI de imagem (ASL e LIBRAS legado)."""
    if model_choice == 'LIBRAS' and libras_model is not None:
        inp = _preprocess_libras(hand_roi_bgr)
        mdl, labels = libras_model, LIBRAS_LABELS
    elif model_choice == 'ASL' and asl_model is not None:
        inp = _preprocess_asl(hand_roi_bgr)
        mdl, labels = asl_model, ASL_LABELS
    else:
        return None, 0.0, []

    probs = mdl(np.expand_dims(inp, axis=0), training=False).numpy()[0]
    best_idx  = int(probs.argmax())
    best_conf = float(probs[best_idx])
    if best_conf >= confidence_threshold:
        return labels[best_idx], best_conf, probs
    return None, best_conf, probs

def predict_vlibrasil(feat_vector, confidence_threshold=0.50):
    """Predição V-LIBRASIL a partir do vetor de landmarks agregado."""
    if vlibrasil_model is None:
        return None, 0.0
    probs = vlibrasil_model(
        np.expand_dims(feat_vector, axis=0), training=False
    ).numpy()[0]
    best_idx  = int(probs.argmax())
    best_conf = float(probs[best_idx])
    if best_conf >= confidence_threshold:
        return VLIBRASIL_LABELS.get(best_idx, str(best_idx)), best_conf
    return None, best_conf

def predict_alfabeto(feat_vector, confidence_threshold=0.50):
    if alfabeto_model is None:
        return None, 0.0
    probs = alfabeto_model(
        np.expand_dims(feat_vector, axis=0), training=False
    ).numpy()[0]
    best_idx = int(probs.argmax())
    best_conf = float(probs[best_idx])
    if best_conf >= confidence_threshold:
        return ALFABETO_LABELS.get(best_idx, str(best_idx)), best_conf
    return None, best_conf

# ──────────────────────────────────────────────────────────────
# Desenho e processamento de frames
# ──────────────────────────────────────────────────────────────
def draw_coordinates(hand_landmarks, img, model_choice, confidence_threshold=0.70):
    offset = 20
    xs = [lm.x * img.shape[1] for lm in hand_landmarks]
    ys = [lm.y * img.shape[0] for lm in hand_landmarks]
    x_min = max(0, int(min(xs) - offset))
    x_max = min(img.shape[1], int(max(xs) + offset))
    y_min = max(0, int(min(ys) - offset))
    y_max = min(img.shape[0], int(max(ys) + offset))
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    hand_roi = img[y_min:y_max, x_min:x_max]
    label = None
    if hand_roi.size > 0 and hand_roi.shape[0] > 0 and hand_roi.shape[1] > 0:
        label, conf, _ = predict_from_roi(hand_roi, model_choice, confidence_threshold)
        if label is not None:
            text = f"{label} ({conf:.0%})"
            text_y = max(0, y_min - 10)
            cv2.putText(img, text, (x_min, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
    return img, label

def draw_subtitle_bar(img, subtitle_text):
    if not subtitle_text:
        return img
    h, w = img.shape[:2]
    bar_h = 50
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(subtitle_text, font, 1.0, 2)[0]
    text_x = (w - text_size[0]) // 2
    text_y = h - (bar_h - text_size[1]) // 2
    cv2.putText(img, subtitle_text, (text_x, text_y), font, 1.0, (255, 255, 255), 2)
    return img

def process_frame_image(img, landmarker, model_choice, confidence_threshold=0.70,
                        lm_buffer=None):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = landmarker.detect(mp_img)
    detected_labels = []

    if results.hand_landmarks:
        # Atualizar buffer para V-LIBRASIL
        if lm_buffer is not None:
            lm_buffer.add(results.hand_landmarks)

        if model_choice == 'VLIBRASIL':
            # Predição com buffer de sequência
            if lm_buffer is not None and lm_buffer.ready():
                feat = lm_buffer.get_feature_vector()
                label, conf = predict_vlibrasil(feat, confidence_threshold)
                if label:
                    detected_labels.append(label)
        elif model_choice == 'LIBRAS_ALFABETO':
            # Predição com 1 frame apenas
            feat = np.zeros(2 * 21 * 3, dtype=np.float32)
            for h_idx, hand_lm in enumerate(results.hand_landmarks[:2]):
                offset = h_idx * 21 * 3
                for l_idx, lm in enumerate(hand_lm):
                    base = offset + l_idx * 3
                    feat[base] = lm.x
                    feat[base + 1] = lm.y
                    feat[base + 2] = lm.z
            label, conf = predict_alfabeto(feat, confidence_threshold)
            if label:
                detected_labels.append(label)
                # Desenhar bounding box
            for hl in results.hand_landmarks:
                xs = [lm.x * img.shape[1] for lm in hl]
                ys = [lm.y * img.shape[0] for lm in hl]
                x_min = max(0, int(min(xs) - 20))
                x_max = min(img.shape[1], int(max(xs) + 20))
                y_min = max(0, int(min(ys) - 20))
                y_max = min(img.shape[0], int(max(ys) + 20))
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 200, 255), 2)
            if detected_labels:
                cv2.putText(img, f"{detected_labels[0]} ({conf:.0%})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 200, 255), 2)
        else:
            for hl in results.hand_landmarks:
                img, label = draw_coordinates(hl, img, model_choice, confidence_threshold)
                if label:
                    detected_labels.append(label)

    return img, detected_labels

def process_frame_video(img, landmarker, timestamp_ms, model_choice,
                        confidence_threshold=0.70, lm_buffer=None):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = landmarker.detect_for_video(mp_img, timestamp_ms)
    detected_labels = []

    if results.hand_landmarks:
        if lm_buffer is not None:
            lm_buffer.add(results.hand_landmarks)

        if model_choice == 'VLIBRASIL':
            if lm_buffer is not None and lm_buffer.ready():
                feat = lm_buffer.get_feature_vector()
                label, conf = predict_vlibrasil(feat, confidence_threshold)
                if label:
                    detected_labels.append(label)
            for hl in results.hand_landmarks:
                xs = [lm.x * img.shape[1] for lm in hl]
                ys = [lm.y * img.shape[0] for lm in hl]
                x_min = max(0, int(min(xs) - 20))
                x_max = min(img.shape[1], int(max(xs) + 20))
                y_min = max(0, int(min(ys) - 20))
                y_max = min(img.shape[0], int(max(ys) + 20))
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 200, 255), 2)
        else:
            for hl in results.hand_landmarks:
                img, label = draw_coordinates(hl, img, model_choice, confidence_threshold)
                if label:
                    detected_labels.append(label)

    return img, detected_labels


# ──────────────────────────────────────────────────────────────
# Estabilizador de legendas
# ──────────────────────────────────────────────────────────────
class SubtitleStabilizer:
    def __init__(self, stability_frames=5):
        self.stability_frames = stability_frames
        self.history = deque(maxlen=stability_frames)
        self.current_text = ""

    def update(self, labels):
        text = " + ".join(sorted(labels)) if labels else ""
        self.history.append(text)
        if len(self.history) == self.stability_frames and len(set(self.history)) == 1:
            self.current_text = text
        return self.current_text


# ──────────────────────────────────────────────────────────────
# Transformer para Webcam
# ──────────────────────────────────────────────────────────────
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.landmarker   = create_hand_landmarker_image()
        self.model_choice = 'ASL'
        self.confidence_threshold = 0.70
        self.stabilizer   = SubtitleStabilizer(stability_frames=3)
        self.lm_buffer    = LandmarkBuffer()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        img, labels = process_frame_image(
            img, self.landmarker, self.model_choice,
            self.confidence_threshold, self.lm_buffer
        )
        subtitle = self.stabilizer.update(labels)
        img = draw_subtitle_bar(img, subtitle)
        return img


# ──────────────────────────────────────────────────────────────
# Interface Streamlit
# ──────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://www.mjvinnovation.com/wp-content/uploads/2021/07/"
    "mjv_blogpost_redes_neurais_ilustracao_cerebro-01-1024x1020.png"
)
st.sidebar.title('Reconhecimento de :red[Sinais] :wave:')

st.sidebar.info("""\
## Reconhecimento de Mãos - Projeto

Este projeto visa desenvolver um programa capaz de utilizar uma rede neural
treinada para detectar mãos em tempo real por meio da câmera do usuário, ou
através do upload de um vídeo. Usando um modelo de rede neural CNN.
""")

# ── Seletor de modelo ─────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### Modelo de reconhecimento")

# Opções disponíveis dependem dos modelos carregados
_options = ["ASL (Letras A-Y)"]
if libras_model is not None:
    _options.append("LIBRAS Legado (15 palavras)")
if vlibrasil_model is not None:
    _options.append("V-LIBRASIL (Palavras em Português)")
if alfabeto_model is not None:
    _options.append("LIBRAS Alfabeto (Letras Estáticas)")

MODEL_CHOICE_STR = st.sidebar.radio(
    "Selecione o modelo:",
    _options,
    help=(
        "ASL = alfabeto americano (A-Y). "
        "LIBRAS Legado = 15 palavras (modelo antigo). "
        "V-LIBRASIL = dataset completo da UFPE."
    )
)

if "V-LIBRASIL" in MODEL_CHOICE_STR:
    MODEL_KEY = 'VLIBRASIL'
elif "Alfabeto" in MODEL_CHOICE_STR:
    MODEL_KEY = 'LIBRAS_ALFABETO'
elif "Legado" in MODEL_CHOICE_STR:
    MODEL_KEY = 'LIBRAS'
else:
    MODEL_KEY = 'ASL'

# Status dos modelos
st.sidebar.markdown("---")
st.sidebar.markdown("### Status dos modelos")
st.sidebar.markdown(f"{'✅' if asl_model else '❌'} ASL")
st.sidebar.markdown(f"{'✅' if libras_model else '❌'} LIBRAS Legado")
st.sidebar.markdown(f"{'✅' if alfabeto_model else '❌'} LIBRAS Alfabeto")
st.sidebar.markdown(
    f"{'✅' if vlibrasil_model else '⚠️'} V-LIBRASIL "
    f"({'treinado' if vlibrasil_model else 'execute train_vlibrasil_model.py'})"
)
if vlibrasil_model:
    st.sidebar.caption(f"{len(VLIBRASIL_LABELS)} classes carregadas")


def exibir_imagem():
    st.subheader("Imagem dos sinais (dataset legado)")
    num_colunas = 5
    nomes = [
        'bank_1605967468_148.jpeg', 'bus_1605967420_87.jpeg',
        'car_1605967469_166.jpeg', 'formation_1605967420_969.jpeg',
        'hospital_1605967420_62.jpeg', 'I_1605967469_110.jpeg',
        'man_1605967420_82.jpeg', 'motorcycle_1605967415_6.jpeg',
        'my_1605967420_99.jpeg', 'supermarket_1605967420_70.jpeg',
        'we_1605967420_78.jpeg', 'woman_1605967469_87.jpeg',
        'you (plural)_1605967420_55.jpeg', 'you_1605967420_63.jpeg',
        'your_1605967420_70.jpeg',
    ]
    legendas = [
        'banco', 'onibus', 'carro', 'formacao', 'hospital',
        'eu', 'homem', 'motocicleta', 'Meu', 'supermercado',
        'nos', 'mulher', 'voces', 'voce', 'sua',
    ]
    imagens = [os.path.join(current_dir, 'assets', nome) for nome in nomes]
    colunas = st.columns(num_colunas)
    for i, (imagem_path, legenda) in enumerate(zip(imagens, legendas)):
        with colunas[i % num_colunas]:
            if os.path.exists(imagem_path):
                st.image(imagem_path, caption=legenda, width=150)
            else:
                st.caption(f"[{legenda}] (imagem nao encontrada)")


# ──────────────────────────────────────────────────────────────
# Pagina principal
# ──────────────────────────────────────────────────────────────
st.title("Sistema de Leitura de Gestos em LIBRAS")

if MODEL_KEY == 'VLIBRASIL' and vlibrasil_model is None:
    st.error(
        "🚫 Modelo V-LIBRASIL não está treinado ainda.\n\n"
        "Para treinar, execute no terminal:\n"
        "```\ncd app\npython train_vlibrasil_model.py\n```\n\n"
        "O treinamento pode levar 30-60 minutos dependendo do hardware."
    )
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("### Configuracoes de predicao")
CONFIDENCE = st.sidebar.slider(
    "Confianca minima (%)",
    min_value=1, max_value=99,
    value=50 if MODEL_KEY == 'VLIBRASIL' else 70,
    help="Aumentar reduz falsos positivos"
) / 100.0
STABILITY = st.sidebar.slider(
    "Estabilidade da legenda (frames)",
    min_value=1, max_value=10, value=3,
    help="Frames consecutivos necessarios para mudar a legenda"
)

modo = st.radio("Selecione a fonte de entrada:", ("Webcam", "Upload de Video"))

if modo == "Webcam":
    ctx = webrtc_streamer(key="hand-recognition-1", video_processor_factory=VideoTransformer)
    if ctx.video_processor:
        ctx.video_processor.model_choice = MODEL_KEY
        ctx.video_processor.confidence_threshold = CONFIDENCE
        ctx.video_processor.stabilizer.stability_frames = STABILITY

elif modo == "Upload de Video":
    FRAME_SKIP = st.sidebar.slider(
        "Processar 1 a cada N frames",
        min_value=1, max_value=6, value=2,
        help="Valores maiores = processamento mais rapido"
    )

    video_file = st.file_uploader(
        "Faca o upload do seu video (MP4, AVI, MOV)", type=['mp4', 'avi', 'mov']
    )

    if video_file is not None:
        tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile_in.write(video_file.read())
        tfile_in.close()

        cap = cv2.VideoCapture(tfile_in.name)
        fps_val    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        tfile_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile_out.close()

        output_container = av.open(tfile_out.name, mode='w')
        video_stream = output_container.add_stream('h264', rate=int(fps_val))
        video_stream.width   = width
        video_stream.height  = height
        video_stream.pix_fmt = 'yuv420p'
        video_stream.options = {'preset': 'fast', 'crf': '23'}

        progress_bar = st.progress(0, text="Processando video...")

        landmarker  = create_hand_landmarker_video()
        stabilizer  = SubtitleStabilizer(stability_frames=STABILITY)
        lm_buffer   = LandmarkBuffer()
        frame_count = 0
        last_annotated = None
        last_labels    = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            if frame_count % FRAME_SKIP == 0:
                last_annotated, last_labels = process_frame_video(
                    frame.copy(), landmarker, timestamp_ms,
                    MODEL_KEY, CONFIDENCE, lm_buffer
                )

            display_frame = last_annotated if last_annotated is not None else frame
            subtitle = stabilizer.update(last_labels)
            display_frame = draw_subtitle_bar(display_frame.copy(), subtitle)

            av_frame = av.VideoFrame.from_ndarray(display_frame, format='bgr24')
            for packet in video_stream.encode(av_frame):
                output_container.mux(packet)

            if total_frames > 0:
                pct = min(frame_count / total_frames, 1.0)
                progress_bar.progress(pct,
                    text=f"Processando... frame {frame_count}/{total_frames}")

        cap.release()
        for packet in video_stream.encode():
            output_container.mux(packet)
        output_container.close()

        progress_bar.empty()
        st.success("Processamento concluido! Reproduzindo o video anotado:")

        with open(tfile_out.name, 'rb') as vf:
            st.video(vf.read())

        for p in [tfile_in.name, tfile_out.name]:
            try:
                os.remove(p)
            except Exception:
                pass

if st.button("Clique aqui para exibir as imagens do dataset legado"):
    exibir_imagem()

st.sidebar.write("""\
## Integrantes

- Lorrayne
- Libhinny
- Samira
- Ytalo
""")
