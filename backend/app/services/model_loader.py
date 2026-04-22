import json
import logging
from pathlib import Path

from app.config import (
    LIPNET_MODEL_PATH,
    MODEL_PATH,
    ONNX_DEPLOYMENT_META_PATH,
    ONNX_IMAGE_FULL_MODEL_PATH,
    ONNX_IMAGE_MODEL_PATH,
    ONNX_VIDEO_MODEL_PATH,
    PYTORCH_IMAGE_MODEL_PATH,
    PYTORCH_AUDIO_MODEL_PATH,
)


logger = logging.getLogger(__name__)


def _load_onnx_session(path: Path):
    """Load an onnxruntime InferenceSession. Returns None if unavailable."""
    try:
        import onnxruntime as ort

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = 2

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
            if _cuda_available() else ['CPUExecutionProvider']

        session = ort.InferenceSession(str(path), sess_options=sess_opts, providers=providers)
        logger.info("ONNX session loaded: %s (providers: %s)", path, session.get_providers())
        return session
    except Exception as exc:
        logger.warning("Failed to load ONNX session from %s: %s", path, exc)
        return None


def _cuda_available() -> bool:
    try:
        import onnxruntime as ort
        return 'CUDAExecutionProvider' in ort.get_available_providers()
    except Exception:
        return False


def _remap_checkpoint_keys(state_dict: dict) -> dict:
    """Remap checkpoint keys from old naming convention to current model_arch names."""
    remap = {
        'stream1_clip':   'clip_stream',
        'stream2_effnet': 'effnet_stream',
        'stream3_freq':   'freq_stream',
        'stream4_srm':    'srm_stream',
        'stream5_gram':   'gram_stream',
        'fusion':         'image_fusion',
        'head':           'image_head',
        'binary_out':     'image_binary_out',
        'type_out':       'image_type_out',
        'temporal':          'temporal_transformer',
        'video_binary_out':  'video_binary_head',
        'video_type_out':    'video_type_head',
    }
    new_state = {}
    for k, v in state_dict.items():
        new_k = k
        for old, new in remap.items():
            if k.startswith(old + '.'):
                new_k = new + k[len(old):]
                break
        new_state[new_k] = v
    return new_state


def _load_pytorch_model(path: Path):
    """Load DeepTruthHybridV2 PyTorch checkpoint."""
    try:
        import torch
        import sys
        sys.path.insert(0, str(path.parent.parent))  # project root for model_arch
        from model_arch import DeepTruthHybridV2

        ckpt = torch.load(path, map_location='cpu')
        num_fake_types = ckpt.get('num_fake_types', 10)
        # Set seed before model init so any randomly-initialised missing weights
        # are always identical across restarts (from_pretrained resets rng state)
        torch.manual_seed(42)
        model = DeepTruthHybridV2(num_fake_types=num_fake_types, dropout=0.0)
        # Remap old key names to current architecture names
        state_dict = _remap_checkpoint_keys(ckpt['model_state'])
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        loaded = len(state_dict) - len(unexpected)
        logger.info("Loaded %d / %d checkpoint keys (strict=False)", loaded, len(state_dict))
        if missing:
            logger.warning("Missing keys (%d): %s", len(missing), missing[:10])
        if unexpected:
            logger.warning("Unexpected keys (%d): %s", len(unexpected), unexpected[:10])
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        logger.info("PyTorch model loaded from %s on %s", path, device)
        return {
            'type': 'pytorch',
            'model': model,
            'device': device,
            'fake_type_map': ckpt.get('fake_type_map', {}),
            'num_fake_types': num_fake_types,
        }
    except Exception as exc:
        logger.warning("Failed to load PyTorch model from %s: %s", path, exc)
        return None


def load_model():
    """Load the image model. Prefers PyTorch, then ONNX, then TF Keras."""
    # --- Try PyTorch first ---
    if PYTORCH_IMAGE_MODEL_PATH.exists():
        bundle = _load_pytorch_model(PYTORCH_IMAGE_MODEL_PATH)
        if bundle:
            return bundle

    # --- Try ONNX ---
    if ONNX_IMAGE_FULL_MODEL_PATH.exists():
        session = _load_onnx_session(ONNX_IMAGE_FULL_MODEL_PATH)
        if session:
            meta = _load_deployment_meta()
            temperature = meta.get('image_model', {}).get('temperature', 1.0)
            return {'type': 'onnx_full', 'session': session, 'temperature': temperature, 'meta': meta}

    if ONNX_IMAGE_MODEL_PATH.exists():
        session = _load_onnx_session(ONNX_IMAGE_MODEL_PATH)
        if session:
            meta = _load_deployment_meta()
            temperature = meta.get('image_model', {}).get('temperature', 1.0)
            return {'type': 'onnx', 'session': session, 'temperature': temperature, 'meta': meta}

    # --- Fall back to TF ---
    if MODEL_PATH.exists():
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(MODEL_PATH)
            logger.info("TF image model loaded from: %s", MODEL_PATH.resolve())
            return {'type': 'tensorflow', 'session': model}
        except Exception as exc:
            logger.error("Failed to load TF image model: %s", exc)

    logger.warning("No image model found at %s or %s", ONNX_IMAGE_MODEL_PATH, MODEL_PATH)
    return None


def load_lipnet_model():
    """Load the video model. Prefers ONNX; falls back to TF Keras."""
    # --- Try ONNX first ---
    if ONNX_VIDEO_MODEL_PATH.exists():
        session = _load_onnx_session(ONNX_VIDEO_MODEL_PATH)
        if session:
            meta = _load_deployment_meta()
            temperature = meta.get('video_model', {}).get('temperature', 1.0)
            n_frames    = meta.get('video_model', {}).get('n_frames', 32)
            return {'type': 'onnx', 'session': session, 'temperature': temperature,
                    'n_frames': n_frames, 'meta': meta}

    # --- Fall back to TF ---
    if LIPNET_MODEL_PATH.exists():
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(LIPNET_MODEL_PATH)
            logger.info("TF video model loaded from: %s", LIPNET_MODEL_PATH.resolve())
            return {'type': 'tensorflow', 'session': model}
        except Exception as exc:
            logger.error("Failed to load TF video model: %s", exc)

    logger.warning("No video model found at %s or %s", ONNX_VIDEO_MODEL_PATH, LIPNET_MODEL_PATH)
    return None


def load_audio_model():
    """Load the PyTorch audio model (SimpleAudioDetector). Returns None if not found."""
    if not PYTORCH_AUDIO_MODEL_PATH.exists():
        logger.warning("No audio model found at %s", PYTORCH_AUDIO_MODEL_PATH)
        return None
    try:
        import torch
        import torch.nn as nn

        class SimpleAudioDetector(nn.Module):
            def __init__(self):
                super().__init__()
                from transformers import Wav2Vec2Model
                self.wav2vec = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')
                for p in self.wav2vec.feature_extractor.parameters():
                    p.requires_grad = False
                self.classifier = nn.Sequential(
                    nn.Linear(768, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 2)
                )

            def forward(self, x):
                out = self.wav2vec(x.float()).last_hidden_state
                pooled = out.mean(dim=1)
                logit = self.classifier(pooled)
                return {'fake_logit': logit, 'embedding': pooled}

        ckpt = torch.load(PYTORCH_AUDIO_MODEL_PATH, map_location='cpu', weights_only=False)
        num_fake_types = ckpt.get('num_fake_types', 10)
        model = SimpleAudioDetector()
        model.load_state_dict(ckpt['model_state'], strict=False)
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        logger.info("Audio model loaded from %s on %s", PYTORCH_AUDIO_MODEL_PATH, device)
        return {
            'type': 'pytorch_audio',
            'model': model,
            'device': device,
            'fake_type_map': ckpt.get('fake_type_map', {}),
            'num_fake_types': num_fake_types,
            'sample_rate': ckpt.get('sample_rate', 16000),
            'clip_samples': ckpt.get('clip_samples', 64000),
        }
    except Exception as exc:
        logger.warning("Failed to load audio model from %s: %s", PYTORCH_AUDIO_MODEL_PATH, exc)
        return None


def _load_deployment_meta() -> dict:
    if ONNX_DEPLOYMENT_META_PATH.exists():
        try:
            with open(ONNX_DEPLOYMENT_META_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}
