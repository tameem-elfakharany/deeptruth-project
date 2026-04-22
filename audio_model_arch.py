"""
DeepTruth Audio Model Architecture
5-stream hybrid model for audio deepfake detection.

Streams:
  1. Mel Spectrogram CNN     — captures spectral patterns
  2. Wav2Vec2 embeddings     — pretrained speech representations
  3. LFCC features           — linear frequency cepstral coefficients
  4. Phase spectrum          — phase inconsistencies in fakes
  5. RawNet waveform         — raw waveform artifacts

Usage in Colab:
    import sys; sys.path.insert(0, '/content')
    from audio_model_arch import *
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

AUDIO_MANIPULATION_TYPES = [
    'real',
    'tts_tacotron',
    'tts_waveglow',
    'tts_hifigan',
    'tts_vits',
    'voice_conversion',
    'gan_vocoder',
    'diffusion_vocoder',
    'wavefake',
    'unknown_fake',
]
N_AUDIO_TYPES = len(AUDIO_MANIPULATION_TYPES)

SAMPLE_RATE   = 16000
CLIP_DURATION = 4       # seconds
CLIP_SAMPLES  = SAMPLE_RATE * CLIP_DURATION   # 64000 samples


# ── Stream 1: Mel Spectrogram CNN ─────────────────────────────────────────────
class MelSpectrogramStream(nn.Module):
    """Learns spectral artifact patterns from Mel spectrograms."""

    def __init__(self, out_dim=256, n_mels=128, hop_length=160, n_fft=512):
        super().__init__()
        self.n_mels     = n_mels
        self.hop_length = hop_length
        self.n_fft      = n_fft

        # Mel filterbank — fixed, not learned
        self.register_buffer('mel_fb', self._build_mel_filterbank(n_mels, n_fft, SAMPLE_RATE))

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def _build_mel_filterbank(self, n_mels, n_fft, sr):
        """Simple triangular mel filterbank."""
        n_freqs = n_fft // 2 + 1
        mel_min = 2595 * math.log10(1 + 0 / 700)
        mel_max = 2595 * math.log10(1 + (sr / 2) / 700)
        mel_pts = torch.linspace(mel_min, mel_max, n_mels + 2)
        hz_pts  = 700 * (10 ** (mel_pts / 2595) - 1)
        bins    = torch.floor((n_fft + 1) * hz_pts / sr).long()
        fb      = torch.zeros(n_mels, n_freqs)
        for m in range(1, n_mels + 1):
            f_m_minus = bins[m - 1]
            f_m       = bins[m]
            f_m_plus  = bins[m + 1]
            for k in range(f_m_minus, f_m):
                if f_m != f_m_minus:
                    fb[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                if f_m_plus != f_m:
                    fb[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
        return fb  # (n_mels, n_freqs)

    def compute_mel(self, x):
        """x: (B, T) waveform -> (B, 1, n_mels, T') spectrogram"""
        x = x.float()
        # STFT
        window = torch.hann_window(self.n_fft, device=x.device)
        stft   = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                            win_length=self.n_fft, window=window,
                            return_complex=True)          # (B, n_fft//2+1, T')
        mag    = stft.abs().clamp(min=1e-9)
        mel    = torch.matmul(self.mel_fb.to(x.device), mag)   # (B, n_mels, T')
        log_mel = torch.log(mel + 1e-9)
        # Normalize
        mn  = log_mel.amin(dim=(-2, -1), keepdim=True)
        mx  = log_mel.amax(dim=(-2, -1), keepdim=True)
        log_mel = (log_mel - mn) / (mx - mn + 1e-4)
        return log_mel.unsqueeze(1)   # (B, 1, n_mels, T')

    def forward(self, x):
        mel = self.compute_mel(x)
        return self.proj(self.cnn(mel))


# ── Stream 2: Wav2Vec2 embeddings ─────────────────────────────────────────────
class Wav2Vec2Stream(nn.Module):
    """Pretrained Wav2Vec2-base speech representations."""

    def __init__(self, out_dim=512, freeze_layers=8):
        super().__init__()
        from transformers import Wav2Vec2Model
        self.wav2vec = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')

        # Freeze early layers
        for i, layer in enumerate(self.wav2vec.encoder.layers):
            if i < freeze_layers:
                for p in layer.parameters():
                    p.requires_grad = False
        for p in self.wav2vec.feature_extractor.parameters():
            p.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(768, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        # x: (B, T) normalized waveform
        out    = self.wav2vec(x.float()).last_hidden_state  # (B, T', 768)
        pooled = out.mean(dim=1)                             # (B, 768)
        return self.proj(pooled)


# ── Stream 3: LFCC features ───────────────────────────────────────────────────
class LFCCStream(nn.Module):
    """Linear Frequency Cepstral Coefficients — effective for TTS detection."""

    def __init__(self, out_dim=128, n_lfcc=60, n_fft=512, hop_length=160):
        super().__init__()
        self.n_lfcc     = n_lfcc
        self.n_fft      = n_fft
        self.hop_length = hop_length

        # Linear filterbank (unlike mel which is log-spaced)
        n_freqs = n_fft // 2 + 1
        self.register_buffer('linear_fb', self._build_linear_filterbank(n_lfcc, n_freqs))

        self.cnn = nn.Sequential(
            nn.Conv1d(n_lfcc, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def _build_linear_filterbank(self, n_filters, n_freqs):
        fb = torch.zeros(n_filters, n_freqs)
        freq_pts = torch.linspace(0, n_freqs - 1, n_filters + 2).long()
        for m in range(1, n_filters + 1):
            f_minus = freq_pts[m - 1]
            f_center = freq_pts[m]
            f_plus   = freq_pts[m + 1]
            for k in range(f_minus, f_center):
                if f_center != f_minus:
                    fb[m - 1, k] = (k - f_minus) / (f_center - f_minus)
            for k in range(f_center, f_plus):
                if f_plus != f_center:
                    fb[m - 1, k] = (f_plus - k) / (f_plus - f_center)
        return fb

    def forward(self, x):
        x = x.float()
        window = torch.hann_window(self.n_fft, device=x.device)
        stft   = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                            win_length=self.n_fft, window=window,
                            return_complex=True)
        mag    = stft.abs().clamp(min=1e-9)                      # (B, F, T')
        lfcc   = torch.matmul(self.linear_fb.to(x.device), mag)  # (B, n_lfcc, T')
        lfcc   = torch.log(lfcc + 1e-9)
        # DCT approximation via matmul
        n      = lfcc.shape[1]
        dct_mat = torch.cos(math.pi / n * (torch.arange(n, device=x.device).float() + 0.5)
                            .unsqueeze(0) * torch.arange(n, device=x.device).float().unsqueeze(1))
        lfcc   = torch.matmul(dct_mat.unsqueeze(0), lfcc)        # (B, n_lfcc, T')
        return self.proj(self.cnn(lfcc))


# ── Stream 4: Phase Spectrum ──────────────────────────────────────────────────
class PhaseStream(nn.Module):
    """Phase spectrum inconsistencies — TTS systems often have unnatural phase."""

    def __init__(self, out_dim=128, n_fft=512, hop_length=160):
        super().__init__()
        self.n_fft      = n_fft
        self.hop_length = hop_length

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        x = x.float()
        window = torch.hann_window(self.n_fft, device=x.device)
        stft   = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                            win_length=self.n_fft, window=window,
                            return_complex=True)
        phase  = torch.angle(stft) / math.pi              # (B, F, T') in [-1, 1]
        phase  = phase.unsqueeze(1)                        # (B, 1, F, T')
        return self.proj(self.cnn(phase))


# ── Stream 5: RawNet Waveform ─────────────────────────────────────────────────
class RawNetStream(nn.Module):
    """Raw waveform SincNet-style CNN — detects low-level codec/vocoder artifacts."""

    def __init__(self, out_dim=128):
        super().__init__()

        # SincNet-inspired learned bandpass filters
        self.sinc_conv = nn.Conv1d(1, 64, kernel_size=251, stride=1, padding=125, bias=False)
        self.bn0 = nn.BatchNorm1d(64)

        self.residual_blocks = nn.Sequential(
            self._res_block(64, 128, stride=4),
            self._res_block(128, 128, stride=4),
            self._res_block(128, 256, stride=4),
            self._res_block(256, 256, stride=4),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def _res_block(self, in_ch, out_ch, stride):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.3),
            nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.3),
        )

    def forward(self, x):
        # x: (B, T)
        x = x.float().unsqueeze(1)          # (B, 1, T)
        x = F.leaky_relu(self.bn0(self.sinc_conv(x)), 0.3)
        x = self.residual_blocks(x)
        x = self.pool(x)
        return self.proj(x)


# ── Cross-Attention Fusion ────────────────────────────────────────────────────
class AudioCrossAttentionFusion(nn.Module):
    """Fuses 5 audio streams with cross-attention."""

    def __init__(self, stream_dims, out_dim=1024, n_heads=8):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Linear(d, out_dim) for d in stream_dims
        ])
        self.attention = nn.MultiheadAttention(out_dim, n_heads, batch_first=True, dropout=0.1)
        self.norm      = nn.LayerNorm(out_dim)
        self.out_dim   = out_dim

    def forward(self, stream_list):
        # Project each stream to common dim
        tokens = torch.stack([proj(s) for proj, s in zip(self.projections, stream_list)], dim=1)
        # (B, 5, out_dim)
        attended, weights = self.attention(tokens, tokens, tokens)
        fused = self.norm(attended + tokens)
        pooled = fused.mean(dim=1)   # (B, out_dim)
        return pooled, weights


# ── Main Model ────────────────────────────────────────────────────────────────
class DeepTruthAudioV1(nn.Module):
    """
    5-stream hybrid audio deepfake detector.

    Streams: Mel CNN + Wav2Vec2 + LFCC + Phase + RawNet
    Training: SupCon pre-train -> supervised warm-up -> full fine-tune
    """

    STREAM_DIMS = [256, 512, 128, 128, 128]   # output dims of each stream
    FUSED_DIM   = 1024

    def __init__(self, num_fake_types=N_AUDIO_TYPES, dropout=0.3,
                 wav2vec_freeze_layers=8):
        super().__init__()

        # ── Streams ───────────────────────────────────────────────────────────
        self.stream1_mel    = MelSpectrogramStream(out_dim=256)
        self.stream2_wav2vec = Wav2Vec2Stream(out_dim=512, freeze_layers=wav2vec_freeze_layers)
        self.stream3_lfcc   = LFCCStream(out_dim=128)
        self.stream4_phase  = PhaseStream(out_dim=128)
        self.stream5_rawnet = RawNetStream(out_dim=128)

        # ── Fusion ────────────────────────────────────────────────────────────
        self.audio_fusion = AudioCrossAttentionFusion(
            stream_dims=self.STREAM_DIMS,
            out_dim=self.FUSED_DIM,
        )

        # ── Classification head ───────────────────────────────────────────────
        self.audio_head = nn.Sequential(
            nn.Linear(self.FUSED_DIM, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
        )
        self.audio_binary_out = nn.Linear(256, 2)          # real vs fake
        self.audio_type_out   = nn.Linear(256, num_fake_types)

        print('DeepTruth Audio model architecture loaded.')
        print('  Streams: MelCNN + Wav2Vec2 + LFCC + Phase + RawNet')
        print(f'  Device: {DEVICE}')

    def extract_audio_features(self, x):
        """x: (B, T) waveform at 16kHz"""
        s1 = self.stream1_mel(x)
        s2 = self.stream2_wav2vec(x)
        s3 = self.stream3_lfcc(x)
        s4 = self.stream4_phase(x)
        s5 = self.stream5_rawnet(x)
        fused, attn_weights = self.audio_fusion([s1, s2, s3, s4, s5])
        return fused, {'mel': s1, 'wav2vec': s2, 'lfcc': s3,
                       'phase': s4, 'rawnet': s5, 'attn_weights': attn_weights}

    def forward(self, x):
        return self.forward_audio(x)

    def forward_audio(self, x):
        fused, stream_feats = self.extract_audio_features(x)
        h = self.audio_head(fused)
        return {
            'fake_logit':   self.audio_binary_out(h),
            'binary_logit': self.audio_binary_out(h),
            'type_logits':  self.audio_type_out(h),
            'type_logit':   self.audio_type_out(h),
            'embedding':    fused,
            'stream_feats': stream_feats,
        }


# ── Focal Loss ────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce   = F.cross_entropy(logits, targets, reduction='none')
        pt   = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        return loss.mean() if self.reduction == 'mean' else loss


# ── Combined Loss ─────────────────────────────────────────────────────────────
class DeepTruthAudioLoss(nn.Module):
    def __init__(self, type_weight=0.3, focal_gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.focal      = FocalLoss(gamma=focal_gamma)
        self.type_ce    = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.type_weight = type_weight

    def forward(self, outputs, binary_labels, type_labels=None):
        main_loss = self.focal(outputs['fake_logit'], binary_labels)
        if type_labels is not None:
            type_loss = self.type_ce(outputs['type_logits'], type_labels)
            total     = main_loss + self.type_weight * type_loss
        else:
            type_loss = torch.tensor(0.0, device=binary_labels.device)
            total     = main_loss
        return total, {'main': main_loss.item(), 'type': type_loss.item()}


if __name__ == '__main__':
    print('Testing DeepTruthAudioV1 ...')
    model = DeepTruthAudioV1().to(DEVICE)
    x     = torch.randn(2, CLIP_SAMPLES).to(DEVICE)
    with torch.no_grad():
        out = model(x)
    print('Output keys:', list(out.keys()))
    print('fake_logit shape:', out['fake_logit'].shape)
    print('embedding shape: ', out['embedding'].shape)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Total params: {total:.1f}M')
