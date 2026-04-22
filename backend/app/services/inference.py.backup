import logging

import numpy as np

from app.config import IMAGENET_MEAN, IMAGENET_STD, THRESHOLD
from app.schemas.prediction import PredictionResponse

AUDIO_SAMPLE_RATE = 16000
AUDIO_CLIP_SAMPLES = 64000   # 4 seconds at 16kHz


logger = logging.getLogger(__name__)

FAKE_TYPE_NAMES = {
    0: 'Real',
    1: 'Deepfakes (face swap)',
    2: 'Face2Face (reenactment)',
    3: 'FaceSwap',
    4: 'NeuralTextures',
    5: 'DeepFakeDetection',
    6: 'FaceShifter',
    7: 'GAN-generated',
    8: 'Diffusion-generated',
    9: 'Unknown fake',
}


def _preprocess_imagenet(img_rgb_uint8: np.ndarray) -> np.ndarray:
    """Normalize RGB image to ImageNet stats. Returns (1, 3, H, W) float32."""
    import cv2
    img = cv2.resize(img_rgb_uint8, (224, 224)).astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std  = np.array(IMAGENET_STD,  dtype=np.float32).reshape(1, 1, 3)
    img  = (img - mean) / std
    # HWC -> CHW -> BCHW
    return img.transpose(2, 0, 1)[np.newaxis, ...]


def predict_image(model_bundle: dict | object, model_input: np.ndarray,
                  original_rgb: np.ndarray | None = None) -> dict:
    """Run image inference. Accepts PyTorch, ONNX, or legacy TF model."""

    # --- PyTorch path ---
    if isinstance(model_bundle, dict) and model_bundle.get('type') == 'pytorch':
        import torch
        model  = model_bundle['model']
        device = model_bundle['device']

        img = original_rgb if original_rgb is not None else None
        if img is None:
            # model_input is (1,3,224,224) float32 numpy — convert directly
            tensor = torch.from_numpy(model_input).to(device)
        else:
            tensor = torch.from_numpy(_preprocess_imagenet(img)).to(device)

        with torch.no_grad():
            out = model(tensor)

        logit = out['fake_logit']
        if logit.dim() == 2 and logit.shape[1] == 2:
            probs = torch.softmax(logit, dim=1)[0]
            logger.info("DEBUG probs[0]=%.4f probs[1]=%.4f logits=%s", float(probs[0]), float(probs[1]), logit.tolist())
            fake_prob = float(probs[1].cpu())  # probs[1]=P(fake), probs[0]=P(real)
        else:
            fake_prob = float(torch.sigmoid(logit.squeeze()).cpu())

        result = {'raw_prediction': fake_prob}
        return result

    # --- ONNX path ---
    if isinstance(model_bundle, dict) and model_bundle.get('type', '').startswith('onnx'):
        session = model_bundle['session']
        temperature = model_bundle.get('temperature', 1.0)

        # Re-preprocess with ImageNet normalization if raw TF input was passed
        if original_rgb is not None:
            onnx_input = _preprocess_imagenet(original_rgb)
        else:
            onnx_input = model_input  # assume caller already did ImageNet norm

        outputs = session.run(None, {session.get_inputs()[0].name: onnx_input})
        probs = outputs[0][0]  # (2,) — [P(real), P(fake)]
        fake_prob = float(probs[1])

        result = {'raw_prediction': fake_prob}

        # Full model also returns type probs
        if model_bundle['type'] == 'onnx_full' and len(outputs) > 1:
            type_probs = outputs[1][0]  # (10,)
            fake_type_idx = int(np.argmax(type_probs))
            result['fake_type'] = FAKE_TYPE_NAMES.get(fake_type_idx, 'Unknown')
            result['fake_type_confidence'] = float(type_probs[fake_type_idx]) * 100.0
            result['type_probabilities'] = {
                FAKE_TYPE_NAMES[i]: float(type_probs[i]) * 100.0
                for i in range(len(type_probs))
            }

        return result

    # --- Legacy TF path ---
    if isinstance(model_bundle, dict):
        tf_model = model_bundle['session']
    else:
        tf_model = model_bundle  # bare TF model (old callers)

    # TF Xception expects (1, H, W, 3) HWC normalized to [0, 1].
    # When original_rgb (face crop) is available use it directly.
    # Otherwise fall back to converting the CHW model_input back to HWC.
    import cv2 as _cv2
    if original_rgb is not None:
        tf_input = _cv2.resize(original_rgb, (224, 224)).astype('float32') / 255.0
        tf_input = np.expand_dims(tf_input, axis=0)   # (1, 224, 224, 3)
    else:
        # model_input is (1, 3, 224, 224) CHW — transpose to HWC
        tf_input = model_input.transpose(0, 2, 3, 1)  # (1, 224, 224, 3)

    raw = float(tf_model.predict(tf_input, verbose=0)[0][0])
    return {'raw_prediction': raw}


def predict_video(model_bundle: dict | None, frames_input: np.ndarray) -> dict:
    """Run video inference. frames_input: (1, T, 3, H, W) for ONNX or (1, T, H, W, C) for TF."""
    if model_bundle is None:
        raise RuntimeError("Video model not loaded.")

    if isinstance(model_bundle, dict) and model_bundle.get('type') == 'onnx':
        session = model_bundle['session']
        outputs = session.run(None, {session.get_inputs()[0].name: frames_input})
        probs = outputs[0][0]  # (2,)
        fake_prob = float(probs[1])
        result = {'raw_prediction': fake_prob}

        if len(outputs) > 1:
            type_probs = outputs[1][0]
            fake_type_idx = int(np.argmax(type_probs))
            result['fake_type'] = FAKE_TYPE_NAMES.get(fake_type_idx, 'Unknown')
            result['fake_type_confidence'] = float(type_probs[fake_type_idx]) * 100.0

        return result

    # Legacy TF LipNet
    if isinstance(model_bundle, dict):
        tf_model = model_bundle['session']
    else:
        tf_model = model_bundle
    raw = float(tf_model.predict(frames_input, verbose=0)[0][0])
    return {'raw_prediction': raw}


def build_explanation(raw_prediction: float, threshold: float,
                      fake_type: str | None = None) -> str:
    if raw_prediction > threshold:
        base = "The model predicts this is likely fake."
        return f"{base} Detected manipulation type: {fake_type}." if fake_type and fake_type != 'Real' else base
    return "The model predicts this is likely real."


def predict_all_faces(model_bundle: dict | object, face_crops: list) -> dict:
    """
    Run model on every face crop and aggregate results.
    If any face is fake the whole image is flagged as fake.
    Returns the worst-case (highest fake probability) result.

    Args:
        model_bundle: loaded model bundle
        face_crops:   list of (224, 224, 3) uint8 RGB numpy arrays

    Returns dict with keys: raw_prediction, flagged_face_index, faces_detected,
                             fake_type, fake_type_confidence, type_probabilities
    """
    results = []
    for i, crop in enumerate(face_crops):
        inp = _preprocess_imagenet(crop)
        res = predict_image(model_bundle, inp, original_rgb=crop)
        results.append(res)
        logger.info("Face %d/%d: raw=%.6f (inp shape=%s mean=%.4f)",
                    i + 1, len(face_crops), res['raw_prediction'],
                    inp.shape, float(inp.mean()))

    # Worst-case aggregation: highest fake probability wins
    flagged_idx = int(np.argmax([r['raw_prediction'] for r in results]))
    worst = results[flagged_idx]

    return {
        'raw_prediction':       worst['raw_prediction'],
        'flagged_face_index':   flagged_idx if len(face_crops) > 1 else None,
        'faces_detected':       len(face_crops),
        'fake_type':            worst.get('fake_type'),
        'fake_type_confidence': worst.get('fake_type_confidence'),
        'type_probabilities':   worst.get('type_probabilities'),
    }


AUDIO_FAKE_TYPE_NAMES = {
    0: 'Real',
    1: 'TTS (Tacotron)',
    2: 'TTS (WaveGlow)',
    3: 'TTS (HiFi-GAN)',
    4: 'TTS (VITS)',
    5: 'Voice Conversion',
    6: 'GAN Vocoder',
    7: 'Diffusion Vocoder',
    8: 'WaveFake',
    9: 'Unknown Fake',
}


def _preprocess_audio(wav_bytes: bytes, clip_samples: int = AUDIO_CLIP_SAMPLES) -> np.ndarray:
    """
    Load audio bytes, resample to 16kHz, crop/pad to clip_samples.
    Returns (1, clip_samples) float32 numpy array.
    """
    import io
    import soundfile as sf
    import librosa

    buf = io.BytesIO(wav_bytes)
    try:
        wav, sr = sf.read(buf, dtype='float32', always_2d=False)
    except Exception:
        buf.seek(0)
        wav, sr = librosa.load(buf, sr=None, mono=True)

    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != AUDIO_SAMPLE_RATE:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=AUDIO_SAMPLE_RATE)

    if len(wav) >= clip_samples:
        wav = wav[:clip_samples]
    else:
        wav = np.pad(wav, (0, clip_samples - len(wav)))

    return wav[np.newaxis, :].astype(np.float32)   # (1, clip_samples)


def predict_audio(model_bundle: dict, wav_bytes: bytes) -> dict:
    """
    Run audio deepfake inference.
    Returns dict with raw_prediction (float in [0,1]), fake_type, fake_type_confidence.
    """
    if model_bundle is None:
        raise RuntimeError("Audio model not loaded.")

    if not (isinstance(model_bundle, dict) and model_bundle.get('type') == 'pytorch_audio'):
        raise RuntimeError("Unsupported audio model bundle type.")

    import torch
    model  = model_bundle['model']
    device = model_bundle['device']
    clip_samples = model_bundle.get('clip_samples', AUDIO_CLIP_SAMPLES)

    wav_np = _preprocess_audio(wav_bytes, clip_samples=clip_samples)
    tensor = torch.from_numpy(wav_np).to(device)   # (1, T)

    with torch.no_grad():
        out = model(tensor)

    logit = out['fake_logit']
    if logit.dim() == 2 and logit.shape[1] == 2:
        probs = torch.softmax(logit, dim=1)[0]
        fake_prob = float(probs[1].cpu())
    else:
        fake_prob = float(torch.sigmoid(logit.squeeze()).cpu())

    result = {'raw_prediction': fake_prob}

    if 'type_logits' in out:
        type_probs = torch.softmax(out['type_logits'], dim=1)[0].cpu().numpy()
        fake_type_idx = int(np.argmax(type_probs))
        result['fake_type'] = AUDIO_FAKE_TYPE_NAMES.get(fake_type_idx, 'Unknown')
        result['fake_type_confidence'] = float(type_probs[fake_type_idx]) * 100.0
        result['type_probabilities'] = {
            AUDIO_FAKE_TYPE_NAMES[i]: float(type_probs[i]) * 100.0
            for i in range(len(type_probs))
        }

    return result


def format_prediction_response(
    *,
    filename: str,
    raw_prediction: float,
    threshold: float = THRESHOLD,
    heatmap_path: str | None = None,
    fake_type: str | None = None,
    fake_type_confidence: float | None = None,
    type_probabilities: dict | None = None,
    faces_detected: int = 1,
    flagged_face_index: int | None = None,
) -> PredictionResponse:
    raw_prediction = float(raw_prediction)
    prediction_label = "FAKE" if raw_prediction > threshold else "REAL"

    fake_probability = round(raw_prediction * 100.0, 2)
    real_probability = round((1.0 - raw_prediction) * 100.0, 2)
    confidence = max(fake_probability, real_probability)
    explanation = build_explanation(raw_prediction, threshold, fake_type)
    if faces_detected > 1 and prediction_label == "FAKE" and flagged_face_index is not None:
        explanation += f" (detected in face #{flagged_face_index + 1} of {faces_detected})"

    return PredictionResponse(
        success=True,
        filename=filename,
        prediction_label=prediction_label,
        raw_prediction=round(raw_prediction, 6),
        fake_probability=fake_probability,
        real_probability=real_probability,
        confidence=confidence,
        threshold_used=threshold,
        explanation=explanation,
        heatmap_path=heatmap_path,
        fake_type=fake_type,
        fake_type_confidence=fake_type_confidence,
        type_probabilities=type_probabilities,
        faces_detected=faces_detected,
        flagged_face_index=flagged_face_index,
    )
