# ============================================================
# STEP 3 — CLASSICAL SYNC FRONT-END  (sync_frontend_v1.py)
#
# The robustness sweep set the spec: the model needs frames
# aligned to within ±1 sample (±3 after the v8.2 jitter
# fine-tune). This module provides that alignment classically:
#
#   * 32-bit IRIG-106 frame sync word 0xFE6B2840, PCM-FM
#     modulated with the exact v16 parameters
#     (1 Mbps, h=0.7 -> fd=350 kHz, BT=0.5, FS=10 MHz)
#   * Non-coherent SEGMENTED correlation: |corr| handles the
#     unknown carrier phase; splitting the template into 4
#     segments of 8 µs keeps the peak intact under CFO up to
#     ~±60 kHz (coarse CFO removal extends this further)
#   * Energy-normalized so a strong LTE/NR burst elsewhere in
#     the buffer cannot out-shout the sync peak
#   * Detects both normal and spectrally-inverted (conjugated)
#     reception
#
# v17 dataset change this implies (one line in your generator):
#     bits[0:32] = SYNC_BITS          # payload becomes 224 bits
# Frame size, sample count, array shapes all stay identical.
#
# Run this file directly for the SELF-TEST: acquisition
# accuracy over SNR x CFO x SJR (OFDM interferer) grid.
# ============================================================

import numpy as np

# ── v16 modulation parameters ──
FS   = 10e6
RB   = 1e6
SPB  = int(FS / RB)         # 10 samples/bit
H    = 0.7
FD   = H * RB / 2.0         # 350 kHz peak deviation
BT   = 0.5

SYNC_HEX  = 0xFE6B2840      # IRIG-106 32-bit frame sync pattern
SYNC_BITS = np.array([int(b) for b in format(SYNC_HEX, "032b")], dtype=np.int8)
SYNC_LEN  = len(SYNC_BITS) * SPB          # 320 samples
N_SEG     = 4                              # correlation segments
EDGE_TRIM = 8                              # samples trimmed each template end
                                           # (ISI from unknown neighbor bits)

# ────────────────────────────────────────────────────────────
# PCM-FM modulator (mirrors the v16 generator math)
# ────────────────────────────────────────────────────────────
def _gauss_kernel(bt=BT, spb=SPB):
    sigma = np.sqrt(np.log(2)) / (2 * np.pi * bt) * spb
    half = int(np.ceil(4 * sigma))
    n = np.arange(-half, half + 1)
    g = np.exp(-0.5 * (n / sigma) ** 2)
    return g / g.sum()


def pcmfm_iq(bits, phi0=0.0):
    """bits (K,) int -> complex IQ (K*SPB,), unit envelope."""
    nrz = 2.0 * np.asarray(bits, dtype=np.float64) - 1.0
    up = np.repeat(nrz, SPB)
    m = np.convolve(up, _gauss_kernel(), mode="same")
    # trapezoidal integration, dx = 1/FS
    dphi = 2 * np.pi * FD * (m[1:] + m[:-1]) / 2.0 / FS
    phi = phi0 + np.concatenate([[0.0], np.cumsum(dphi)])
    return np.exp(1j * phi)


# ────────────────────────────────────────────────────────────
# Sync correlator
# ────────────────────────────────────────────────────────────
def _build_templates():
    """Sync-word IQ split into N_SEG segments, edge-trimmed,
    each unit-normalized. Returns list of (offset, template)."""
    iq = pcmfm_iq(SYNC_BITS)
    iq = iq[EDGE_TRIM:SYNC_LEN - EDGE_TRIM]
    L = len(iq)
    seg = L // N_SEG
    out = []
    for k in range(N_SEG):
        t = iq[k * seg:(k + 1) * seg]
        out.append((EDGE_TRIM + k * seg, t / np.linalg.norm(t)))
    return out, seg


_TEMPLATES, _SEG_LEN = _build_templates()


def _moving_energy(x, w):
    p = np.concatenate([[0.0], np.cumsum(np.abs(x) ** 2)])
    return p[w:] - p[:-w]


def sync_metric(rx):
    """Energy-normalized segmented |correlation| metric.
    metric[i] is the evidence that a frame STARTS at sample i.
    Handles unknown phase (|.|), CFO (segmentation), and
    spectral inversion (max with conjugated template)."""
    n_lags = len(rx) - SYNC_LEN + 1
    if n_lags <= 0:
        raise ValueError("rx shorter than sync word")
    energy = np.sqrt(np.maximum(_moving_energy(rx, _SEG_LEN), 1e-12))

    def run(conjugate):
        acc = np.zeros(n_lags)
        for off, t in _TEMPLATES:
            tt = np.conj(t) if conjugate else t
            # corr[i] = sum_j rx[i+off+j] * conj(tt[j])
            c = np.correlate(rx, tt, mode="valid")      # len: len(rx)-seg+1
            c = np.abs(c[off:off + n_lags])
            acc += c / energy[off:off + n_lags]
        return acc

    return np.maximum(run(False), run(True)) / N_SEG


def find_frame_start(rx, cfo_hyps=None):
    """2-D acquisition (lag x CFO hypothesis), GPS-style.
    Correlating against the KNOWN sync template makes this robust to
    interference (unlike an FFT-centroid CFO estimate, which the
    interferer can hijack). 30 kHz spacing keeps the residual-CFO
    peak bias at 0 samples (verified: bias is 0 up to ~15 kHz
    residual, <=1 sample up to ~40 kHz).

    Returns (best_start_index, f_hat_hz, peak_value, peak_to_sidelobe).
    f_hat = winning hypothesis + fine estimate from inter-segment
    correlation phase; derotate the frame by -f_hat before the model.
    """
    if cfo_hyps is None:
        cfo_hyps = np.arange(-240e3, 240e3 + 1, 30e3)
    t = np.arange(len(rx)) / FS
    best = (-1.0, 0, 0.0)                       # (peak, idx, f_hyp)
    best_metric = None
    for f in cfo_hyps:
        rxd = rx * np.exp(-1j * 2 * np.pi * f * t)
        m = sync_metric(rxd)
        i = int(np.argmax(m))
        if m[i] > best[0]:
            best = (m[i], i, f)
            best_metric = m
    peak, i, f_hyp = best
    mask = np.ones(len(best_metric), dtype=bool)
    mask[max(0, i - SPB):i + SPB + 1] = False
    psl = peak / max(best_metric[mask].max(), 1e-12) if mask.any() else np.inf

    # fine CFO from inter-segment correlation phase at the winning lag
    rxd = rx * np.exp(-1j * 2 * np.pi * f_hyp * t)
    cs = []
    for off, tpl in _TEMPLATES:
        seg = rxd[i + off:i + off + len(tpl)]
        cs.append(np.vdot(tpl, seg))            # sum seg * conj(tpl)
    cs = np.array(cs)
    rot = np.sum(cs[1:] * np.conj(cs[:-1]))
    f_fine = np.angle(rot) * FS / (2 * np.pi * _SEG_LEN) if np.abs(rot) > 0 else 0.0
    return i, float(f_hyp + f_fine), float(peak), float(psl)


def apply_cfo(x, f_hz):
    t = np.arange(len(x)) / FS
    return x * np.exp(1j * 2 * np.pi * f_hz * t)


# ────────────────────────────────────────────────────────────
# SELF-TEST
# ────────────────────────────────────────────────────────────
def _ofdm_interferer(n, rng, n_sc=48):
    """Wideband OFDM-ish interferer (proxy for LTE/NR), unit power."""
    nfft = 512
    n_sym = int(np.ceil(n / nfft)) + 1
    x = []
    sc = rng.choice(np.arange(-nfft // 4, nfft // 4), size=n_sc, replace=False)
    for _ in range(n_sym):
        S = np.zeros(nfft, dtype=complex)
        S[sc % nfft] = (rng.choice([-1, 1], n_sc) +
                        1j * rng.choice([-1, 1], n_sc)) / np.sqrt(2)
        x.append(np.fft.ifft(S) * np.sqrt(nfft))
    x = np.concatenate(x)[:n]
    return x / np.sqrt(np.mean(np.abs(x) ** 2))


def self_test(n_trials=100, seed=0):
    rng = np.random.default_rng(seed)
    pad = 600
    print("=" * 76)
    print(" SYNC FRONT-END SELF-TEST")
    print(f" sync=0x{SYNC_HEX:08X}  template={SYNC_LEN}smp  "
          f"segments={N_SEG}x{_SEG_LEN}smp  trials/cell={n_trials}  CFO bank ±240kHz/30kHz")
    print("=" * 76)
    print(f"  {'SNR':>4} {'CFO kHz':>8} {'SJR':>5} | {'P(|e|<=1)':>10} "
          f"{'P(|e|<=3)':>10} {'med|e|':>7} {'mean PSL':>9}")
    print("  " + "-" * 64)
    results = {}
    for snr_db in (10, 5, 0):
        for cfo_khz in (0, 50, 100, 200):
            for sjr_db in (None, 0, -5):
                errs, psls = [], []
                for _ in range(n_trials):
                    payload = rng.integers(0, 2, 224)
                    frame = np.concatenate([SYNC_BITS, payload])
                    sig = pcmfm_iq(frame, phi0=rng.uniform(0, 2 * np.pi))
                    # padding = other random modulated bits (realistic stream)
                    pre = pcmfm_iq(rng.integers(0, 2, pad // SPB + 1))[:pad]
                    post = pcmfm_iq(rng.integers(0, 2, pad // SPB + 1))[:pad]
                    rx = np.concatenate([pre, sig, post])
                    true_start = pad
                    if cfo_khz:
                        rx = apply_cfo(rx, cfo_khz * 1e3)
                    if sjr_db is not None:
                        j = _ofdm_interferer(len(rx), rng)
                        rx = rx + j * 10 ** (-sjr_db / 20)
                    npow = 10 ** (-snr_db / 10)
                    rx = rx + (rng.standard_normal(len(rx)) +
                               1j * rng.standard_normal(len(rx))) * np.sqrt(npow / 2)
                    # receiver: 2-D acquisition (lag x CFO bank)
                    est, f_hat, _, psl = find_frame_start(rx)
                    errs.append(est - true_start)
                    psls.append(psl)
                errs = np.abs(np.array(errs))
                key = (snr_db, cfo_khz, sjr_db)
                results[key] = (np.mean(errs <= 1), np.mean(errs <= 3),
                                np.median(errs), np.mean(psls))
                sj = "  --" if sjr_db is None else f"{sjr_db:>4}"
                print(f"  {snr_db:>4} {cfo_khz:>8} {sj:>5} |"
                      f" {results[key][0]*100:>9.1f}%"
                      f" {results[key][1]*100:>9.1f}%"
                      f" {results[key][2]:>7.1f} {results[key][3]:>9.2f}")
    print("""
  READING THIS TABLE
  - P(|e|<=1): fraction of trials aligned within the v8.1 spec.
  - P(|e|<=3): within the post-fine-tune v8.2 spec.
  - PSL (peak-to-sidelobe) is the lock-confidence statistic; in
    deployment, declare sync only when PSL > ~1.5 and confirm on
    2 consecutive frames before passing IQ to the model.
""")
    return results


if __name__ == "__main__":
    self_test()
