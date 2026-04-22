# debug_image.py - Run this from the DeepTruth root directory.
# Mimics the exact server pipeline to diagnose image model discrepancies.
# Usage:
#   cd DeepTruth
#   python debug_image.py <path_to_image>
import sys
import os

# Add backend to path so we can import from app.*
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

import cv2
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_image.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"Image path: {image_path}")
    print(f"File exists: {os.path.exists(image_path)}")
    print(f"File size: {os.path.getsize(image_path)} bytes")
    print()

    # --- Step 1: Load image bytes (same as server) ---
    with open(image_path, 'rb') as f:
        file_bytes = f.read()
    print(f"Loaded {len(file_bytes)} bytes")

    # --- Step 2: preprocess_image_bytes (same as server) ---
    from app.utils.image_processing import preprocess_image_bytes
    model_input, original_rgb = preprocess_image_bytes(file_bytes)
    print(f"original_rgb shape: {original_rgb.shape}, dtype: {original_rgb.dtype}")
    print(f"original_rgb min/max: {original_rgb.min()}/{original_rgb.max()}")
    print()

    # --- Step 3: detect_all_faces (same as server) ---
    from app.services.face_detector import detect_all_faces
    face_crops, face_boxes = detect_all_faces(original_rgb)
    print(f"Faces detected: {len(face_crops)}")
    for i, (crop, box) in enumerate(zip(face_crops, face_boxes if face_boxes else [None]*len(face_crops))):
        print(f"  Face {i}: shape={crop.shape}, box={box}")
        print(f"    crop min/max: {crop.min()}/{crop.max()}")
    print()

    # --- Step 4: Load model (same as server) ---
    print("Loading image model...")
    from app.services.model_loader import load_model
    model_bundle = load_model()
    if model_bundle is None:
        print("ERROR: model_bundle is None — no model loaded!")
        sys.exit(1)
    print(f"Model type: {model_bundle.get('type', 'unknown')}")
    print()

    # --- Step 5: predict_all_faces (same as server) ---
    from app.services.inference import predict_image, _preprocess_imagenet, FAKE_TYPE_NAMES

    print("Running inference on each face crop:")
    results = []
    for i, crop in enumerate(face_crops):
        inp = _preprocess_imagenet(crop)
        print(f"  Face {i}: inp shape={inp.shape}, min={inp.min():.4f}, max={inp.max():.4f}, mean={inp.mean():.4f}")
        res = predict_image(model_bundle, inp, original_rgb=crop)
        results.append(res)
        print(f"  Face {i}: raw_prediction={res['raw_prediction']:.6f} ({res['raw_prediction']*100:.2f}% fake)")
        if 'fake_type' in res:
            print(f"           fake_type={res['fake_type']}, confidence={res.get('fake_type_confidence', 0):.1f}%")

    print()

    # --- Worst-case aggregation ---
    flagged_idx = int(np.argmax([r['raw_prediction'] for r in results]))
    worst = results[flagged_idx]
    raw = worst['raw_prediction']
    label = "FAKE" if raw > 0.5 else "REAL"
    print(f"FINAL RESULT: raw={raw:.6f} -> {label}")
    print(f"  Fake probability: {raw*100:.2f}%")
    print(f"  Real probability: {(1-raw)*100:.2f}%")
    print(f"  Confidence: {max(raw, 1-raw)*100:.2f}%")


if __name__ == "__main__":
    main()
