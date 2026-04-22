"""
model_arch.py — DeepTruth full model architecture.
Upload this file to Google Drive/DeepTruth/, then in Colab run:
    !cp /content/drive/MyDrive/DeepTruth/model_arch.py /content/
    from model_arch import *
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from transformers import CLIPVisionModel
import timm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MANIPULATION_TYPES = ['real', 'face_swap', 'face_reenactment', 'gan_generated', 'diffusion_generated']
N_TYPES = len(MANIPULATION_TYPES)


# ── Stream 1: CLIP ViT-B/16 ───────────────────────────────────────────
class CLIPStream(nn.Module):
    def __init__(self, freeze_layers=8, out_dim=512):
        super().__init__()
        self.clip = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch16')
        for i, layer in enumerate(self.clip.vision_model.encoder.layers):
            if i < freeze_layers:
                for p in layer.parameters():
                    p.requires_grad = False
        for p in self.clip.vision_model.embeddings.parameters():
            p.requires_grad = False
        self.proj = nn.Sequential(nn.Linear(768, out_dim), nn.LayerNorm(out_dim), nn.GELU())

    def forward(self, x):
        cls_feat = self.clip(pixel_values=x).last_hidden_state[:, 0, :]
        return self.proj(cls_feat)


# ── Stream 2: EfficientNet-B4 ─────────────────────────────────────────
class EfficientNetStream(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0, global_pool='avg')
        self.proj = nn.Sequential(nn.Linear(1792, out_dim), nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(0.3))
        for name, param in self.backbone.named_parameters():
            if any(f'blocks.{i}.' in name for i in range(4)):
                param.requires_grad = False

    def forward(self, x):
        return self.proj(self.backbone(x))


# ── Stream 3: Frequency (FFT + Wavelet) ──────────────────────────────
class FrequencyStream(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.freq_cnn = nn.Sequential(
            nn.Conv2d(9, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(128*4*4, out_dim), nn.LayerNorm(out_dim), nn.GELU())

    def compute_fft(self, x):
        fft = torch.fft.fft2(x, norm='ortho')
        fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))
        return torch.log1p(torch.abs(fft_shift)), torch.angle(fft_shift) / torch.pi

    def compute_wavelet_ll(self, x):
        ll = F.avg_pool2d(x, kernel_size=2, stride=1, padding=1)
        return F.interpolate(ll, size=x.shape[-2:], mode='bilinear', align_corners=False)

    def forward(self, x):
        mag, phase = self.compute_fft(x)
        wavelet_ll = self.compute_wavelet_ll(x)
        def norm(t):
            mn = t.amin(dim=(-3,-2,-1), keepdim=True)
            mx = t.amax(dim=(-3,-2,-1), keepdim=True)
            return (t - mn) / (mx - mn + 1e-4)   # 1e-8 underflows in fp16
        freq_map = torch.cat([norm(mag), norm(phase), norm(wavelet_ll)], dim=1)
        return self.proj(self.freq_cnn(freq_map))


# ── Stream 4: SRM Noise Residual ─────────────────────────────────────
def get_srm_kernels():
    srm1 = np.array([[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]], dtype=np.float32) / 4.0
    srm2 = np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]], dtype=np.float32) / 12.0
    srm3 = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]], dtype=np.float32) / 2.0
    kernels = []
    for base in [srm1, srm2, srm3]:
        for k in range(4):
            rotated = np.rot90(base, k)
            kernels.append(rotated)
            if k < 2:
                kernels.append(np.flip(rotated, axis=0).copy())
    while len(kernels) < 30:
        kernels.append(srm1)
    return torch.tensor(np.stack(kernels[:30])[:, np.newaxis, :, :])

class SRMStream(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.register_buffer('srm_weight', get_srm_kernels())
        self.cnn = nn.Sequential(
            nn.Conv2d(90, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(4),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(2),
        )
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(128*4, out_dim), nn.LayerNorm(out_dim), nn.GELU())

    def forward(self, x):
        channels = []
        for c in range(3):
            ch = x[:, c:c+1, :, :]
            channels.append(F.conv2d(ch, self.srm_weight, padding=2))
        noise = torch.tanh(torch.cat(channels, dim=1) / 10.0)
        return self.proj(self.cnn(noise))


# ── Stream 5: Gram Matrix Style ───────────────────────────────────────
class GramStyleStream(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.proj = nn.Sequential(
            nn.Linear(128*128, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, out_dim), nn.LayerNorm(out_dim), nn.GELU(),
        )

    def gram_matrix(self, feat):
        B, C, H, W = feat.shape
        f = feat.view(B, C, H*W)
        return torch.bmm(f, f.transpose(1, 2)) / (C*H*W)

    def forward(self, x):
        gram = self.gram_matrix(self.features(x))
        return self.proj(gram.view(gram.shape[0], -1))


# ── Cross-Attention Fusion ────────────────────────────────────────────
class CrossAttentionFusion(nn.Module):
    def __init__(self, stream_dims, n_heads=8, dropout=0.1):
        super().__init__()
        self.attn_dim = 256
        self.projections = nn.ModuleList([nn.Linear(d, self.attn_dim) for d in stream_dims])
        self.attention = nn.MultiheadAttention(self.attn_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(self.attn_dim)

    def forward(self, stream_outputs):
        tokens = [proj(s) for proj, s in zip(self.projections, stream_outputs)]
        token_seq = torch.stack(tokens, dim=1)
        attended, attn_weights = self.attention(token_seq, token_seq, token_seq)
        attended = self.norm(attended + token_seq)
        return attended.flatten(1), attn_weights


# ── Temporal Transformer (video path) ────────────────────────────────
class TemporalTransformer(nn.Module):
    def __init__(self, in_dim=512, n_frames=8, n_heads=8, n_layers=4, out_dim=512):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, n_frames, in_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim, nhead=n_heads, dim_feedforward=in_dim*4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, in_dim) * 0.02)
        self.proj = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU())

    def forward(self, frame_embeds):
        B, T, D = frame_embeds.shape
        x = frame_embeds + self.pos_embed[:, :T, :]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        out = self.transformer(x)
        return self.proj(out[:, 0, :])


# ── Full Unified Model ────────────────────────────────────────────────
class DeepTruthHybridV2(nn.Module):
    def __init__(self, num_fake_types=10, n_frames=8, clip_freeze_layers=8, dropout=0.3):
        super().__init__()
        # Stream names match notebook 04 layer-freeze/unfreeze logic
        self.clip_stream   = CLIPStream(freeze_layers=clip_freeze_layers, out_dim=512)
        self.effnet_stream = EfficientNetStream(out_dim=512)
        self.freq_stream   = FrequencyStream(out_dim=256)
        self.srm_stream    = SRMStream(out_dim=128)
        self.gram_stream   = GramStyleStream(out_dim=128)

        stream_dims = [512, 512, 256, 128, 128]
        self.image_fusion = CrossAttentionFusion(stream_dims)
        fused_dim = 5 * 256  # 1280

        # temporal_transformer name matches notebook 04 stage-4a freeze logic
        self.temporal_transformer = TemporalTransformer(in_dim=512, n_frames=n_frames, out_dim=512)
        video_fused_dim = fused_dim + 512

        self.image_head = nn.Sequential(
            nn.Linear(fused_dim, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout * 0.75),
        )
        self.image_binary_out = nn.Linear(256, 2)
        self.image_type_out   = nn.Linear(256, num_fake_types)

        self.video_head = nn.Sequential(
            nn.Linear(video_fused_dim, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout * 0.75),
        )
        # video_binary_head / video_type_head names match notebook 04 stage-4a freeze logic
        self.video_binary_head = nn.Linear(256, 2)
        self.video_type_head   = nn.Linear(256, num_fake_types)

    def extract_image_features(self, x):
        s1 = self.clip_stream(x)
        s2 = self.effnet_stream(x)
        s3 = self.freq_stream(x)
        s4 = self.srm_stream(x)
        s5 = self.gram_stream(x)
        fused, attn_weights = self.image_fusion([s1, s2, s3, s4, s5])
        return fused, {'clip': s1, 'efficientnet': s2, 'frequency': s3,
                       'srm_noise': s4, 'gram_style': s5, 'attn_weights': attn_weights}

    def forward_image(self, x):
        fused, stream_feats = self.extract_image_features(x)
        h = self.image_head(fused)
        logit = self.image_binary_out(h)
        type_logit = self.image_type_out(h)
        return {
            'fake_logit':   logit,
            'binary_logit': logit,
            'type_logits':  type_logit,
            'type_logit':   type_logit,
            'embedding':    fused,
            'stream_feats': stream_feats,
        }

    def forward_video(self, x):
        B, T, C, H, W = x.shape
        x_flat = x.view(B*T, C, H, W)
        fused_frames, stream_feats = self.extract_image_features(x_flat)
        clip_feats = stream_feats['clip'].view(B, T, -1)
        temporal_feat = self.temporal_transformer(clip_feats)
        fused_mean = fused_frames.view(B, T, -1).mean(dim=1)
        video_feat = torch.cat([fused_mean, temporal_feat], dim=1)
        h = self.video_head(video_feat)
        logit = self.video_binary_head(h)
        type_logit = self.video_type_head(h)
        return {
            'fake_logit':   logit,
            'binary_logit': logit,
            'type_logits':  type_logit,
            'type_logit':   type_logit,
            'stream_feats': stream_feats,
        }

    def forward(self, x):
        if x.dim() == 4:
            return self.forward_image(x)
        elif x.dim() == 5:
            return self.forward_video(x)
        raise ValueError(f'Expected 4D or 5D input, got {x.dim()}D')


# ── Loss Functions ────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class DeepTruthLoss(nn.Module):
    def __init__(self, type_weight=0.3, gamma=2.0, label_smoothing=0.1,
                 num_types=None, focal_gamma=None):
        super().__init__()
        gamma = focal_gamma if focal_gamma is not None else gamma
        self.focal = FocalLoss(gamma=gamma)
        self.type_ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.type_weight = type_weight

    def forward(self, binary_logit, binary_labels, type_logit=None, type_labels=None):
        # Called as: criterion(out['binary_logit'], labels, out['type_logit'], type_ids)
        main_loss = self.focal(binary_logit, binary_labels)
        if type_logit is not None and type_labels is not None:
            type_loss = self.type_ce(type_logit, type_labels)
            total = main_loss + self.type_weight * type_loss
        else:
            total = main_loss
        return total


print('DeepTruth model architecture loaded.')
print(f'  Streams: CLIP + EfficientNet-B4 + Frequency + SRM + GramStyle')
print(f'  Device: {DEVICE}')
