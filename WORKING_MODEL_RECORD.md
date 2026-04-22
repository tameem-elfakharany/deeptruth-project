# Working Model Record

## Current Working Model (as of 2026-04-18)

**File:** `models/deeptruth_image_model_final.pth`  
**Size:** 507,067 KB (≈519 MB)  
**MD5:** `1f07bc973f7079374f91c8ca84a77f42`  
**Training:** Phase 1 only, 5 epochs  
**Metrics:** real=98.7%, fake=97.2%, avg=98.0% (best at epoch 3)

### Trained on fake types:
- Deepfakes (face swap), Face2Face, FaceSwap, NeuralTextures
- DeepFakeDetection, FaceShifter, GAN-generated, Diffusion-generated
- StyleGAN2, Stable Diffusion, DALL-E, Midjourney, DeepFaceLab

---

## Critical Inference Convention

In `backend/app/services/inference.py`:

```python
# probs[1] = P(fake),  probs[0] = P(real)
fake_prob = float(probs[1].cpu())
```

**DO NOT change this to probs[0]** — the model was trained with fake=1, real=0,  
so class index 1 is always the fake class.

---

## How to verify the model is correct

Run this in the project root:
```
certutil -hashfile models\deeptruth_image_model_final.pth MD5
```
Expected MD5: `1f07bc973f7079374f91c8ca84a77f42`

If the hash differs, the model file was replaced with a different/wrong version.
