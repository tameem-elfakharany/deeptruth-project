import logging
import uuid
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from fastapi import HTTPException

from app.config import HEATMAPS_DIR


logger = logging.getLogger(__name__)


def _find_last_conv_layer_name(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        try:
            output_shape = layer.output_shape
        except Exception:
            continue
        if isinstance(output_shape, tuple) and len(output_shape) == 4:
            return layer.name
    raise ValueError("Could not find a 4D convolutional feature map layer for Grad-CAM.")


def generate_gradcam_heatmap(
    *,
    model: tf.keras.Model,
    model_input: np.ndarray,
    original_rgb: np.ndarray,
    filename_stem: str,
    output_dir: Path = HEATMAPS_DIR,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        last_conv_layer_name = _find_last_conv_layer_name(model)
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM layer selection failed: {e}")

    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(tf.convert_to_tensor(model_input))
        class_score = predictions[:, 0]

    grads = tape.gradient(class_score, conv_outputs)
    if grads is None:
        raise HTTPException(status_code=500, detail="Grad-CAM gradient computation failed.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    heatmap_resized = cv2.resize(heatmap, (original_rgb.shape[1], original_rgb.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color_bgr, cv2.COLOR_BGR2RGB)

    overlay_rgb = cv2.addWeighted(original_rgb, 0.6, heatmap_color_rgb, 0.4, 0.0)
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

    out_name = f"{filename_stem}_{uuid.uuid4().hex}.png"
    out_path = output_dir / out_name

    ok = cv2.imwrite(str(out_path), overlay_bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to write heatmap image to disk.")

    logger.info("Grad-CAM heatmap saved to: %s", out_path.resolve())
    return f"/outputs/heatmaps/{out_name}"
