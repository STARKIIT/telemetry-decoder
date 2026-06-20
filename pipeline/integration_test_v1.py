# ============================================================
# END-TO-END INTEGRATION TEST  (integration_test_v1.py)
#
# First time the whole receiver runs as ONE CHAIN:
#
#   synthesized continuous stream (unknown offset, phase, CFO,
#   AWGN, OFDM interferer)
#        -> sync_frontend_v1.find_frame_start()   [alignment + f_hat]
#        -> derotate by -f_hat
#        -> extract 2560-sample frame
#        -> v8.2 model
#        -> 256 bit decisions
#        -> compare vs ground truth
#
# Reported per (SNR x CFO x SJR) cell:
#   sync acquisition accuracy, END-TO-END %perfect / BER, and the
#   ORACLE %perfect (true offset + true CFO used) — the gap between
#   end-to-end and oracle is exactly what the front-end costs.
#
# Frames here are 32-bit sync word + 224 random payload bits.
# v8.2 is content-agnostic, so it decodes the sync bits like any
# others — no retraining needed for this test.
#
# IMPORTANT SCOPE NOTE: the synthetic channel here is simpler than
# the v16 chain (no LNA/IM3/multipath/AGC/ADC). This test validates
# CHAIN MECHANICS (sync -> derotation -> model handoff), not channel
# generalization — that's what val/test on v16 and later sim-to-real
# are for. Expect results close to the clean/easy rows of the
# diagnostic.
#
# Usage (Colab, same folder as sync_frontend_v1.py):
#   !python integration_test_v1.py
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import math
import importlib.util

# ────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────
CHECKPOINT   = "/content/drive/MyDrive/pcmfm_dataset_v16_5G_May/telemetry_v8_2.pth"
SYNC_MODULE  = "sync_frontend_v1.py"      # path to the sync front-end file
N_TRIALS     = 60                          # frames per grid cell
PAD          = 600                         # stream samples before/after frame
SEED         = 11

GRID_SNR_DB  = (15, 10, 5)
GRID_CFO_KHZ = (0, 100)
GRID_SJR_DB  = (None, 0)                  # None = no interferer

# "v16"      -> the exact narrowband LTE/NR tones from pcmfm_dataset_v16
#               (in-distribution: expect SJR-0 rows to look like the
#                easy rows of the diagnostic — confirms chain + model)
# "wideband" -> 48-subcarrier OFDM proxy (out-of-distribution: closer
#               to real 15 kHz-SCS LTE/NR; expect the collapse)
INTERFERER   = "v16"

# Receiver RF front-end: the v16 chain passes the composite through a
# deeply saturated Rapp LNA (mean compression ~0.03 per the generator).
# A saturated LNA is a LIMITER, and limiting confers the classic FM
# capture effect — it suppresses interference relative to the
# constant-envelope telemetry signal. v16 SJR labels are PRE-LNA, so
# the effective post-limiter SJR the model trained on is several dB
# better than labeled. A test without the LNA is therefore HARDER than
# any same-labeled v16 condition. Real receivers have this LNA; keep
# it on for like-for-like comparison.
APPLY_LNA    = True
P1DB_RNG     = (-25.0, -5.0)               # dBm, as in v16

NUM_BITS, SIG_LEN = 256, 2560

CFG = {
	"num_bits": NUM_BITS, "dim": 192, "n_conformer": 4,
	"n_heads": 4, "conv_kernel": 15, "dropout": 0.1,
}

# ────────────────────────────────────────────────────────────
# import sync front-end
# ────────────────────────────────────────────────────────────
def load_sync_module():
    for cand in (SYNC_MODULE, f"/content/{SYNC_MODULE}",
                 f"/content/drive/MyDrive/{SYNC_MODULE}"):
        if os.path.exists(cand):
            spec = importlib.util.spec_from_file_location("syncfe", cand)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["syncfe"] = mod
            spec.loader.exec_module(mod)
            print(f"  sync front-end loaded from {cand}")
            return mod
    raise FileNotFoundError(
        f"{SYNC_MODULE} not found — put it next to this script")


# ────────────────────────────────────────────────────────────
# MODEL BLUEPRINT — verbatim from telemetry_v8_1.py
# ────────────────────────────────────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        bottleneck = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, bottleneck, bias=False), nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _ = x.shape
        return x * self.fc(self.pool(x).view(b, c)).view(b, c, 1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dilation=1):
        super().__init__()
        pad1 = 3 * dilation
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 7, stride=stride, padding=pad1,
                      dilation=dilation, bias=False),
            nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 5, padding=2, bias=False),
            nn.BatchNorm1d(out_ch))
        self.se = SEBlock(out_ch)
        self.skip = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_ch)) if (in_ch != out_ch or stride != 1) else nn.Identity()

    def forward(self, x):
        return F.relu(self.se(self.conv(x)) + self.skip(x), inplace=True)


class SpectralGate(nn.Module):
    def __init__(self, hidden=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, hidden, 31, padding=15), nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, 15, padding=7), nn.ReLU(inplace=True),
            nn.Conv1d(hidden, 1, 7, padding=3))

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            z = torch.complex(xf[:, 0, :], xf[:, 1, :])
            Z = torch.fft.fft(z, dim=-1)
            logmag = torch.log1p(torch.abs(Z)).unsqueeze(1)
            logmag = (logmag - logmag.mean(dim=-1, keepdim=True)) / \
                     (logmag.std(dim=-1, keepdim=True) + 1e-6)
            phase_bin = torch.angle(Z).unsqueeze(1)
            feats = torch.cat([logmag, torch.cos(phase_bin)], dim=1)
            mask = torch.sigmoid(self.net(feats)).squeeze(1)
            zf = torch.fft.ifft(Z * mask, dim=-1)
            out = torch.stack([zf.real, zf.imag], dim=1)
        return out.to(x.dtype)


class ConformerBlock(nn.Module):
    def __init__(self, d, heads, conv_kernel=15, ff_mult=4, drop=0.1):
        super().__init__()
        def ffn():
            return nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, d * ff_mult), nn.SiLU(), nn.Dropout(drop),
                nn.Linear(d * ff_mult, d), nn.Dropout(drop))
        self.ff1 = ffn()
        self.attn_ln = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.conv_ln = nn.LayerNorm(d)
        self.conv = nn.Sequential(
            nn.Conv1d(d, 2 * d, 1), nn.GLU(dim=1),
            nn.Conv1d(d, d, conv_kernel, padding=(conv_kernel - 1) // 2, groups=d),
            nn.BatchNorm1d(d), nn.SiLU(),
            nn.Conv1d(d, d, 1), nn.Dropout(drop))
        self.ff2 = ffn()
        self.final_ln = nn.LayerNorm(d)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        a = self.attn_ln(x)
        x = x + self.attn(a, a, a, need_weights=False)[0]
        c = self.conv_ln(x).permute(0, 2, 1)
        x = x + self.conv(c).permute(0, 2, 1)
        x = x + 0.5 * self.ff2(x)
        return self.final_ln(x)


class TelemetryPipelineV8(nn.Module):
    def __init__(self, cfg=CFG):
        super().__init__()
        d = cfg["dim"]
        self.spectral_gate = SpectralGate()
        self.enc1 = ResBlock(6, 64)
        self.enc2 = ResBlock(64, 128, stride=2)
        self.enc3 = ResBlock(128, 192, stride=2)
        self.mid  = ResBlock(192, 192, dilation=2)
        self.up2  = nn.ConvTranspose1d(192, 128, 4, stride=2, padding=1)
        self.dec2 = ResBlock(256, 128)
        self.up1  = nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1)
        self.dec1 = ResBlock(128, 64)
        self.out_conv = nn.Conv1d(64, 2, 7, padding=3)
        self.feature_cnn = nn.Sequential(
            ResBlock(10, 128),
            ResBlock(128, d, stride=5),
            ResBlock(d, d, stride=2))
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg["num_bits"], d))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.conformer = nn.ModuleList([
            ConformerBlock(d, cfg["n_heads"], cfg["conv_kernel"], drop=cfg["dropout"])
            for _ in range(cfg["n_conformer"])])
        self.bit_head = nn.Linear(d, 1)
        self.aux_head = nn.Sequential(
            nn.Linear(d, 64), nn.SiLU(), nn.Linear(64, 4))

    @staticmethod
    def _phase_features(x):
        xf = x.float()
        z = torch.complex(xf[:, 0, :], xf[:, 1, :])
        dz = z[:, 1:] * torch.conj(z[:, :-1])
        dz = dz / torch.abs(dz).clamp(min=1e-6)
        d_phi = F.pad(torch.angle(dz), (1, 0))
        return torch.stack([torch.sin(d_phi), torch.cos(d_phi)], dim=1).to(x.dtype)

    def forward(self, x_noisy):
        x_filt = self.spectral_gate(x_noisy)
        dphi_n = self._phase_features(x_noisy)
        u_in = torch.cat([x_noisy, x_filt, dphi_n], dim=1)
        e1 = self.enc1(u_in)
        e2 = self.enc2(e1)
        e3 = self.mid(self.enc3(e2))
        h  = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        h  = self.dec1(torch.cat([self.up1(h), e1], dim=1))
        x_recon = self.out_conv(h)
        dphi_r = self._phase_features(x_recon)
        t = self.feature_cnn(torch.cat(
            [x_recon, x_noisy, x_filt, dphi_r, dphi_n], dim=1)).permute(0, 2, 1)
        t = t + self.pos_emb
        for blk in self.conformer:
            t = blk(t)
        return x_recon, self.bit_head(t).squeeze(-1), self.aux_head(t.mean(dim=1))


# ────────────────────────────────────────────────────────────
# stream synthesis + receiver chain
# ────────────────────────────────────────────────────────────
def apply_lna_v16(composite, p1db_dbm, p_rapp=2.0):
    """Rapp AM-AM limiter, ported verbatim from pcmfm_dataset_v16
    (length-general). Returns unit-power output like the v16 chain."""
    v_p1db = math.sqrt(10 ** (p1db_dbm / 10) * 1e-3 * 50)
    v_sat = v_p1db * (2 * p_rapp / (2 * p_rapp - 1)) ** (1 / (2 * p_rapp))
    v_rms = math.sqrt(np.mean(np.abs(composite) ** 2) * 50e-3)
    if v_rms < 1e-15:
        return composite
    scale = v_rms / (math.sqrt(np.mean(np.abs(composite) ** 2)) + 1e-15)
    sv = composite * scale
    amp = np.abs(sv)
    comp = 1 / (1 + (amp / v_sat) ** (2 * p_rapp)) ** (1 / (2 * p_rapp))
    out = sv * comp
    return out / (np.sqrt(np.mean(np.abs(out) ** 2)) + 1e-12)


def make_trial(sf, rng, snr_db, cfo_khz, sjr_db):
    """Returns (rx_stream, true_start, true_cfo_hz, bits).
    Stage order mirrors v16: interference -> LNA -> CFO -> AWGN."""
    payload = rng.integers(0, 2, NUM_BITS - len(sf.SYNC_BITS))
    bits = np.concatenate([sf.SYNC_BITS, payload]).astype(np.int64)
    sig = sf.pcmfm_iq(bits, phi0=rng.uniform(0, 2 * np.pi))
    pre  = sf.pcmfm_iq(rng.integers(0, 2, PAD // sf.SPB + 1))[:PAD]
    post = sf.pcmfm_iq(rng.integers(0, 2, PAD // sf.SPB + 1))[:PAD]
    rx = np.concatenate([pre, sig, post])
    if sjr_db is not None:
        j = make_interferer(len(rx), rng)
        rx = rx + j * 10 ** (-sjr_db / 20)
    if APPLY_LNA:
        rx = apply_lna_v16(rx, float(rng.uniform(*P1DB_RNG)))
    f_true = cfo_khz * 1e3
    if f_true:
        rx = sf.apply_cfo(rx, f_true)
    # noise referenced to post-LNA composite power (v16 convention:
    # AWGN refs AGC output power, i.e. the composite, not the signal)
    p_ref = float(np.mean(np.abs(rx) ** 2))
    npow = p_ref / (10 ** (snr_db / 10))
    rx = rx + (rng.standard_normal(len(rx)) +
               1j * rng.standard_normal(len(rx))) * np.sqrt(npow / 2)
    return rx, PAD, f_true, bits


def _ofdm(n, rng, n_sc=48, nfft=512):
    n_sym = int(np.ceil(n / nfft)) + 1
    sc = rng.choice(np.arange(-nfft // 4, nfft // 4), size=n_sc, replace=False)
    x = []
    for _ in range(n_sym):
        S = np.zeros(nfft, dtype=complex)
        S[sc % nfft] = (rng.choice([-1, 1], n_sc) +
                        1j * rng.choice([-1, 1], n_sc)) / np.sqrt(2)
        x.append(np.fft.ifft(S) * np.sqrt(nfft))
    x = np.concatenate(x)[:n]
    return x / np.sqrt(np.mean(np.abs(x) ** 2))


# ── v16 interferers, ported verbatim from pcmfm_dataset_v16 (length-
#    parameterized so they cover the whole stream, not just one frame) ──
def _lte_v16(n, rng):
    fft_size = 32; cp_len = max(1, int(0.0694 * fft_size))
    sym_len = fft_size + cp_len; n_sym = n // sym_len + 2
    offset = rng.integers(-2, 3)
    sc_pos = np.clip([fft_size // 2 + offset - 1, fft_size // 2 + offset + 1],
                     1, fft_size - 1)
    lvl = np.array([-3, -1, 1, 3]) / np.sqrt(10)
    pts = np.array([a + 1j * b for a in lvl for b in lvl])
    stream = np.zeros(n_sym * sym_len, dtype=np.complex128)
    for s in range(n_sym):
        fd = np.zeros(fft_size, dtype=np.complex128)
        fd[sc_pos] = rng.choice(pts, size=len(sc_pos))
        td = np.fft.ifft(np.fft.ifftshift(fd))
        stream[s * sym_len:s * sym_len + sym_len] = \
            np.concatenate([td[-cp_len:], td])
    stream = stream[:n]
    duty = float(rng.uniform(0.70, 0.98))
    blen = max(1, int(duty * n)); bst = rng.integers(0, max(1, n - blen))
    mask = np.zeros(n, dtype=np.float32); mask[bst:bst + blen] = 1.0
    out = stream * (mask + 10 ** (-10 / 20) * (1 - mask))
    p = np.mean(np.abs(out) ** 2)
    return out / (np.sqrt(p) + 1e-15) if p > 1e-20 else out


def _nr_v16(n, rng):
    fft_size = 16; cp_len = max(1, int(0.0694 * fft_size))
    sym_len = fft_size + cp_len; n_sym = n // sym_len + 2
    mini = int(rng.integers(2, 8))
    sc_pos = np.array([fft_size // 2 + int(rng.integers(1, 3))])
    lvl = np.arange(-15, 16, 2) / np.sqrt(170)
    pts = np.array([a + 1j * b for a in lvl for b in lvl],
                   dtype=np.complex128)
    stream = np.zeros(n_sym * sym_len, dtype=np.complex128)
    for s in range(n_sym):
        if (s % 14) < mini:
            fd = np.zeros(fft_size, dtype=np.complex128)
            fd[sc_pos] = rng.choice(pts, size=len(sc_pos))
            td = np.fft.ifft(np.fft.ifftshift(fd))
            stream[s * sym_len:s * sym_len + sym_len] = \
                np.concatenate([td[-cp_len:], td])
    out = stream[:n]
    p = np.mean(np.abs(out) ** 2)
    return out / (np.sqrt(p) + 1e-15) if p > 1e-20 else out


def make_interferer(n, rng):
    if INTERFERER == "wideband":
        return _ofdm(n, rng)
    # v16: LTE always + NR half the time, equal-power split like the
    # composite SJR convention
    j = _lte_v16(n, rng)
    if rng.random() < 0.5:
        j = (j + _nr_v16(n, rng)) / np.sqrt(2)
    return j


def extract_frame(sf, rx, start, f_hz):
    """Derotate by -f_hz, slice SIG_LEN samples at start, scale to the
    v16 INPUT CONTRACT, return (2, SIG_LEN) float32.

    SCALE MATTERS: the v16 chain ends with AGC at -10 dBm, so training
    frames sit at ~1e-4 W power (RMS 0.01). Phase features are
    scale-invariant but the SpectralGate's log1p spectral features are
    NOT — unit-power input (+40 dB off-contract) breaks interference
    excision while leaving clean decoding intact. Discovered by this
    test on 2026-06; deployment receivers must AGC to this point."""
    z = sf.apply_cfo(rx, -f_hz)
    seg = z[start:start + SIG_LEN]
    if len(seg) < SIG_LEN:                       # clamp at stream edge
        seg = np.pad(seg, (0, SIG_LEN - len(seg)))
    TARGET_RMS = 1e-2                            # -10 dBm, v16 convention
    p = np.sqrt(np.mean(np.abs(seg) ** 2)) + 1e-12
    seg = seg * (TARGET_RMS / p)
    return np.stack([seg.real, seg.imag]).astype(np.float32)


@torch.no_grad()
def predict_bits(model, frames, device):
    x = torch.from_numpy(np.stack(frames)).to(device)
    _, logits, _ = model(x)
    return (logits > 0).long().cpu().numpy()


# ────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(SEED)
    print(f"Device: {device}")
    sf = load_sync_module()

    ckpt = torch.load(CHECKPOINT, map_location=device)
    model = TelemetryPipelineV8(CFG).to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded EMA weights (epoch {ckpt.get('epoch', '?')})\n")

    print("=" * 86)
    print(f" END-TO-END INTEGRATION TEST  (sync -> derotate -> v8.2)  | interferer = {INTERFERER}")
    print(f" {N_TRIALS} frames/cell | frame = 32-bit sync 0x{sf.SYNC_HEX:08X}"
          f" + 224 payload bits")
    print("=" * 86)
    print(f"  {'SNR':>4} {'CFO kHz':>8} {'SJR':>5} | {'sync<=1':>8} "
          f"{'sync<=3':>8} | {'E2E %perf':>9} {'E2E BER':>9} | "
          f"{'oracle %perf':>12}")
    print("  " + "-" * 78)

    for snr in GRID_SNR_DB:
        for cfo in GRID_CFO_KHZ:
            for sjr in GRID_SJR_DB:
                e2e_frames, ora_frames, truths = [], [], []
                sync_err = []
                for _ in range(N_TRIALS):
                    rx, t0, f_true, bits = make_trial(sf, rng, snr, cfo, sjr)
                    est, f_hat, _, _ = sf.find_frame_start(rx)
                    sync_err.append(abs(est - t0))
                    e2e_frames.append(extract_frame(sf, rx, est, f_hat))
                    ora_frames.append(extract_frame(sf, rx, t0, f_true))
                    truths.append(bits)
                truths = np.stack(truths)
                pred_e = predict_bits(model, e2e_frames, device)
                pred_o = predict_bits(model, ora_frames, device)
                err_e = (pred_e != truths).sum(axis=1)
                err_o = (pred_o != truths).sum(axis=1)
                sync_err = np.array(sync_err)
                sj = "  --" if sjr is None else f"{sjr:>4}"
                print(f"  {snr:>4} {cfo:>8} {sj:>5} |"
                      f" {np.mean(sync_err <= 1)*100:>7.1f}%"
                      f" {np.mean(sync_err <= 3)*100:>7.1f}% |"
                      f" {np.mean(err_e == 0)*100:>8.1f}%"
                      f" {err_e.sum()/(N_TRIALS*NUM_BITS):>9.5f} |"
                      f" {np.mean(err_o == 0)*100:>11.1f}%")

    print("""
  READING THIS
  - 'E2E' uses ONLY what a real receiver has (estimated start +
    estimated CFO). 'oracle' uses ground truth. The E2E-vs-oracle gap
    is the total cost of the classical front-end.
  - Synthetic channel is simpler than v16 (no LNA/multipath/AGC chain),
    so absolute numbers should look like the easy rows of the
    diagnostic; the CHAIN handoff is what is being validated.
  - Success criterion: E2E within ~2 pts of oracle in every cell.
    Larger gaps localize the fault: low sync<=3 -> front-end;
    sync good but E2E low -> extraction/derotation handoff.
""")
    print("DONE")


if __name__ == "__main__":
    main()
