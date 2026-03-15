import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import numpy as np
import base64
import cv2
import librosa
import soundfile as sf
import tensorflow as tf
from deepface import DeepFace
import tempfile, io

app = FastAPI(title="Tone AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

VOICE_MODEL_PATH = "models/Emotion_Voice_Detection_Model.h5"
VOICE_LABELS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
EMOJI_MAP = {
    "neutral": "😐", "calm": "😌", "happy": "😊", "sad": "😢",
    "angry": "😠", "fearful": "😨", "disgust": "🤢", "surprised": "😲",
}

# ─── Load voice model ──────────────────
voice_model = None
try:
    import tf_keras
    voice_model = tf_keras.models.load_model(VOICE_MODEL_PATH, compile=False)
    print("✅ Voice model loaded via tf_keras")
except Exception as e1:
    try:
        voice_model = tf.keras.models.load_model(VOICE_MODEL_PATH, compile=False)
        print("✅ Voice model loaded via tf.keras")
    except Exception as e2:
        print(f"❌ Voice model failed: {e2}")

def extract_mfcc(audio_bytes: bytes) -> np.ndarray:
    """Try multiple methods to load audio and extract MFCC."""
    # Method 1: direct librosa load from bytes
    try:
        buf = io.BytesIO(audio_bytes)
        X, sr = librosa.load(buf, sr=22050, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T, axis=0)
        return mfccs.reshape(1, 40, 1)
    except Exception:
        pass

    # Method 2: write to temp file and load
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(audio_bytes)
        tmp = f.name
    try:
        X, sr = librosa.load(tmp, sr=22050, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T, axis=0)
        return mfccs.reshape(1, 40, 1)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.post("/analyze/face")
async def analyze_face(image: str = Form(...)):
    try:
        # Decode base64 image
        if "," in image:
            image = image.split(",", 1)[1]
        img_bytes = base64.b64decode(image)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return JSONResponse({"success": False, "error": "Could not decode image"})

        result = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
            detector_backend="opencv"
        )

        emotions = {k: round(float(v), 2) for k, v in result[0]["emotion"].items()}
        dominant = str(result[0]["dominant_emotion"])
        region = {}
        for k, v in result[0]["region"].items():
            try:
                region[k] = int(v) if v is not None else 0
            except (TypeError, ValueError):
                region[k] = 0

        return JSONResponse({
            "success": True,
            "dominant": dominant,
            "emoji": EMOJI_MAP.get(dominant, "🤔"),
            "emotions": emotions,
            "face_box": region,
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})

@app.post("/analyze/voice")
async def analyze_voice(audio: UploadFile = File(...)):
    if voice_model is None:
        return JSONResponse({"success": False, "error": "Voice model not loaded."})
    try:
        audio_bytes = await audio.read()
        features = extract_mfcc(audio_bytes)
        preds = voice_model.predict(features, verbose=0)[0]
        idx = int(np.argmax(preds))
        label = VOICE_LABELS[idx]
        all_emotions = {VOICE_LABELS[i]: round(float(preds[i]) * 100, 1) for i in range(len(preds))}
        return JSONResponse({
            "success": True,
            "dominant": label,
            "emoji": EMOJI_MAP.get(label, "🤔"),
            "confidence": round(float(preds[idx]) * 100, 1),
            "emotions": all_emotions,
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})

@app.post("/analyze/combined")
async def analyze_combined(image: str = Form(...), audio: UploadFile = File(...)):
    face_res = await analyze_face(image=image)
    voice_res = await analyze_voice(audio=audio)
    face_data = face_res.body
    voice_data = voice_res.body
    import json
    fd = json.loads(face_data)
    vd = json.loads(voice_data)
    if fd.get("success") and vd.get("success"):
        fused = {}
        for lbl in VOICE_LABELS:
            f_score = fd["emotions"].get(lbl, 0) / 100
            v_score = vd["emotions"].get(lbl, 0) / 100
            fused[lbl] = round((f_score + v_score) / 2 * 100, 1)
        dominant = max(fused, key=fused.get)
        return JSONResponse({
            "success": True,
            "dominant": dominant,
            "emoji": EMOJI_MAP.get(dominant, "🤔"),
            "emotions": fused,
            "face": fd,
            "voice": vd,
        })
    return JSONResponse({"success": False, "face": fd, "voice": vd})

app.mount("/static", StaticFiles(directory="static"), name="static")