# ============================================================
# PHASE-2 LLR EXPORT + LDPC HARNESS — v8.1
#
# Turns the model into the front half of a real receiver chain
# and reports the metric deployments are judged on: POST-FEC
# frame error rate.
#
#   1. Run inference on VAL  -> fit a temperature T (focal loss
#      typically mis-calibrates confidence; LDPC decoders need
#      honest LLRs). Reports ECE before/after.
#   2. Run inference on TEST -> export calibrated per-bit LLRs.
#   3. Decode through a rate-1/2 / 2/3 / 4/5 LDPC proxy
#      (regular Gallager construction + normalized min-sum,
#      50 iterations, GPU-vectorized across blocks).
#   4. Report post-FEC BER / FER overall, in the failure box,
#      and for lte_and_nr — with and without interleaving.
#
# FEC evaluation method (standard "extracted channel" trick):
# the dataset bits are random, not codewords, so we map the
# observed per-bit channel onto the all-zeros codeword:
#     llr_i = +|L_i|  if the model's hard decision was correct
#             -|L_i|  otherwise
# Valid given output symmetry (the conjugation/bit-inversion
# augmentation explicitly trains 0/1 symmetry into the model).
#
# CAVEAT printed in the report: the code here is a regular
# Gallager proxy, not the exact IRIG-106 AR4JA codes — expect
# the true codes to be slightly BETTER (a few tenths of a dB).
# If the proxy already gives FER ~ 0, the conclusion stands.
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import math

# ────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────
DATA_DIR   = "/content/drive/MyDrive/pcmfm_dataset_v16_5G_May"
CHECKPOINT = "/content/drive/MyDrive/pcmfm_dataset_v16_5G_May/telemetry_v8_1.pth"
OUT_NPZ    = "llr_export_v8_1.npz"
NUM_BITS   = 256
BATCH_SIZE = 256
LLR_CLIP   = 24.0
LDPC_ITERS = 50
MINSUM_ALPHA = 0.75          # normalized min-sum scaling
SEED       = 7

# Proxy codes ~ IRIG-106 k=1024 family: (rate, dv, dc, n)
# n chosen divisible by dc, k = n*(1 - dv/dc) ~= 1024
CODES = [("r1/2", 3, 6, 2046),
         ("r2/3", 3, 9, 1539),
         ("r4/5", 3, 15, 1275)]

CFG = {
    "num_bits": NUM_BITS, "dim": 192, "n_conformer": 4,
    "n_heads": 4, "conv_kernel": 15, "dropout": 0.1,
}
LBL_LTE, LBL_SJR_LTE, LBL_NR = 2, 3, 4

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
        bit_logits = self.bit_head(t).squeeze(-1)
        aux = self.aux_head(t.mean(dim=1))
        return x_recon, bit_logits, aux


# ────────────────────────────────────────────────────────────
# INFERENCE — collect logits
# ────────────────────────────────────────────────────────────
@torch.no_grad()
def collect_logits(model, split, device, limit=None):
    sd = os.path.join(DATA_DIR, split)
    X = np.load(f"{sd}/X_noisy.npy", mmap_mode="r")
    y = np.load(f"{sd}/y_bits.npy").astype(np.int64)
    lbl = np.load(f"{sd}/labels.npy")
    N = len(X) if limit is None else min(limit, len(X))
    logits = np.zeros((N, NUM_BITS), dtype=np.float32)
    model.eval()
    for i0 in range(0, N, BATCH_SIZE):
        xb = torch.from_numpy(
            np.asarray(X[i0:i0 + BATCH_SIZE]).transpose(0, 2, 1).copy()) \
            .float().to(device)
        _, lg, _ = model(xb)
        logits[i0:i0 + BATCH_SIZE] = lg.float().cpu().numpy()
    return logits, y[:N], lbl[:N]


# ────────────────────────────────────────────────────────────
# CALIBRATION
# ────────────────────────────────────────────────────────────
def expected_calibration_error(logits, y, n_bins=15):
    p = 1.0 / (1.0 + np.exp(-logits.ravel()))
    conf = np.maximum(p, 1 - p)
    correct = ((p > 0.5).astype(np.int64) == y.ravel())
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    ece, rows = 0.0, []
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i + 1] + (1e-9 if i == n_bins - 1 else 0))
        if m.sum() == 0:
            continue
        acc, cf, w = correct[m].mean(), conf[m].mean(), m.mean()
        ece += w * abs(acc - cf)
        rows.append((bins[i], bins[i + 1], m.sum(), cf, acc))
    return ece, rows


def fit_temperature(logits, y, device):
    lg = torch.from_numpy(logits.ravel()).float().to(device)
    yt = torch.from_numpy(y.ravel()).float().to(device)
    log_T = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=80)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(lg / torch.exp(log_T), yt)
        loss.backward()
        return loss
    opt.step(closure)
    return float(torch.exp(log_T).item())


# ────────────────────────────────────────────────────────────
# PROXY LDPC — regular Gallager construction + min-sum (GPU)
# ────────────────────────────────────────────────────────────
def build_regular_code(n, dv, dc, rng):
    """Gallager construction: dv stacked permuted band submatrices.
    Returns (m_total, check_idx, var_idx) with edges sorted by check,
    each check having exactly dc edges."""
    assert n % dc == 0
    m_band = n // dc
    check_idx, var_idx = [], []
    for s in range(dv):
        perm = rng.permutation(n)
        for pos, v in enumerate(perm):
            check_idx.append(s * m_band + pos // dc)
            var_idx.append(v)
    check_idx = np.asarray(check_idx)
    var_idx = np.asarray(var_idx)
    order = np.argsort(check_idx, kind="stable")
    return dv * m_band, check_idx[order], var_idx[order]


@torch.no_grad()
def minsum_decode(chan_llr, n, m, check_idx, var_idx, dc, device,
                  iters=LDPC_ITERS, alpha=MINSUM_ALPHA):
    """chan_llr: (B, n) torch tensor, +ve favors bit 0.
    Returns posterior LLRs (B, n)."""
    B = chan_llr.shape[0]
    E = len(var_idx)
    vi = torch.from_numpy(var_idx).long().to(device)
    V2C = chan_llr[:, vi].clone()                       # (B, E)
    C2V = torch.zeros_like(V2C)
    arange_dc = torch.arange(dc, device=device).view(1, 1, dc)

    for _ in range(iters):
        # ── check update (edges grouped by check: reshape (B, m, dc)) ──
        M = V2C.view(B, m, dc)
        sgn = torch.where(M >= 0, 1.0, -1.0)
        prod_sgn = sgn.prod(dim=2, keepdim=True) * sgn   # sign excl. self
        absM = M.abs()
        min1, idx1 = absM.min(dim=2, keepdim=True)
        tmp = absM.scatter(2, idx1, float("inf"))
        min2 = tmp.min(dim=2, keepdim=True)[0]
        mag = torch.where(arange_dc == idx1, min2, min1)  # min excl. self
        C2V = (alpha * prod_sgn * mag).view(B, E)
        # ── variable update ──
        acc = torch.zeros(B, n, device=device)
        acc.index_add_(1, vi, C2V)
        post = chan_llr + acc
        V2C = post[:, vi] - C2V
        # early stop: all-zero codeword recovered everywhere
        if (post >= 0).all():
            break
    return post


def decode_stratum(name, llr_signed, frame_ids, codes, device, rng,
                   interleave):
    """llr_signed: 1-D stream of signed channel LLRs (all-zero mapping),
    frame contiguity preserved. Returns rows for the report table."""
    rows = []
    stream = llr_signed
    if interleave:
        perm = rng.permutation(len(stream))
        stream = stream[perm]
    pre_ber = (stream < 0).mean()
    for tag, dv, dc, n in codes:
        m, ci, vi = build_regular_code(n, dv, dc, np.random.RandomState(SEED))
        n_blocks = len(stream) // n
        if n_blocks == 0:
            continue
        blocks = torch.from_numpy(
            stream[:n_blocks * n].reshape(n_blocks, n)).float().to(device)
        # decode in chunks to bound memory
        post_all = []
        CHUNK = 256
        for i0 in range(0, n_blocks, CHUNK):
            post_all.append(minsum_decode(blocks[i0:i0 + CHUNK], n, m, ci, vi,
                                          dc, device))
        post = torch.cat(post_all, dim=0)
        errs = (post < 0)
        post_ber = errs.float().mean().item()
        fer = errs.any(dim=1).float().mean().item()
        n_bad = int(errs.any(dim=1).sum().item())
        rows.append((name, tag, n_blocks, pre_ber, post_ber, fer, n_bad))
    return rows


# ────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.RandomState(SEED)
    print(f"Device: {device}")

    print("\n[1/5] Loading checkpoint ...")
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model = TelemetryPipelineV8(CFG).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded EMA weights (epoch {ckpt.get('epoch', '?')})")

    print("\n[2/5] Inference on VAL (for temperature fit) ...")
    lg_val, y_val, _ = collect_logits(model, "val", device)
    ece0, _ = expected_calibration_error(lg_val, y_val)
    T = fit_temperature(lg_val, y_val, device)
    ece1, rel = expected_calibration_error(lg_val / T, y_val)
    print(f"  Temperature T = {T:.3f}  "
          f"({'over' if T > 1 else 'under'}-confident raw logits)")
    print(f"  ECE before = {ece0:.5f}   ECE after = {ece1:.5f}")
    print("  Reliability (after scaling):  conf_bin -> empirical acc")
    for lo, hi, cnt, cf, acc in rel[-6:]:
        print(f"    [{lo:.2f},{hi:.2f})  n={cnt:>9,}  conf={cf:.4f}  acc={acc:.4f}")

    print("\n[3/5] Inference on TEST -> calibrated LLR export ...")
    lg_test, y_test, lbl = collect_logits(model, "test", device)
    # LLR convention: positive favors bit 0.  logit = log(p1/p0)  =>  llr = -logit/T
    llr = np.clip(-lg_test / T, -LLR_CLIP, LLR_CLIP).astype(np.float32)
    pred = (lg_test > 0).astype(np.int64)
    correct = (pred == y_test)
    pre_ber = 1.0 - correct.mean()
    print(f"  Test pre-FEC BER (sanity, should match diagnostic): {pre_ber:.6f}")
    np.savez(OUT_NPZ, llr=llr, y_bits=y_test.astype(np.int8),
             labels=lbl, temperature=np.float32(T))
    print(f"  Saved LLRs -> {OUT_NPZ}")

    print("\n[4/5] Building all-zero-codeword channel streams per stratum ...")
    # signed llr under all-zero mapping: +|llr| if correct else -|llr|
    signed = np.where(correct, np.abs(llr), -np.abs(llr)).astype(np.float32)
    lte = lbl[:, LBL_LTE] > 0.5
    nr = lbl[:, LBL_NR] > 0.5
    box = lte & (lbl[:, LBL_SJR_LTE] < 5.0) & (lbl[:, LBL_SJR_LTE] < 90.0)
    strata = {
        "ALL":         np.ones(len(lbl), dtype=bool),
        "failure_box": box,
        "lte_and_nr":  lte & nr,
        "outside_box": ~box,
    }
    for k, msk in strata.items():
        print(f"  {k:<12}: {msk.sum():>6,} frames "
              f"({msk.sum() * NUM_BITS:,} bits)")

    print("\n[5/5] LDPC decoding (proxy regular codes, normalized min-sum, "
          f"{LDPC_ITERS} iters) ...")
    all_rows = []
    for inter in (False, True):
        for k, msk in strata.items():
            stream = signed[msk].reshape(-1)
            all_rows += decode_stratum(
                k + (" +intlv" if inter else ""), stream,
                np.where(msk)[0], CODES, device, rng, interleave=inter)

    print("\n" + "=" * 86)
    print(" POST-FEC RESULTS  (all-zero-codeword extracted-channel method)")
    print("=" * 86)
    print(f"  {'stratum':<22} {'code':>5} {'#blocks':>8} "
          f"{'pre BER':>10} {'post BER':>10} {'FER':>10} {'#bad':>6}")
    print("  " + "-" * 80)
    for name, tag, nb, pb, qb, fer, n_bad in all_rows:
        print(f"  {name:<22} {tag:>5} {nb:>8,} {pb:>10.6f} "
              f"{qb:>10.2e} {fer:>10.2e} {n_bad:>6}")
    print("""
  NOTES
  - 'pre BER' is the raw model BER on that stratum; 'post BER'/'FER' are
    after LDPC.  FER is the deployment metric.
  - '+intlv' rows interleave bits across frames before blocking, as a
    deployed IRIG-106 chain would; non-interleaved rows are worst-case
    (a whole bad frame lands inside one codeword).
  - Proxy code caveat: regular Gallager codes, not the exact IRIG-106
    AR4JA family.  True standardized codes typically perform a few
    tenths of a dB BETTER, so FER ~ 0 here is a safe conclusion;
    nonzero FER here means re-check with the exact code before deciding.
  - If post-FEC FER in the failure box is ~0 at r1/2 and r2/3, raw-BER
    model iteration is officially done — move to sync + sim-to-real.
""")
    print("DONE")


if __name__ == "__main__":
    main()
