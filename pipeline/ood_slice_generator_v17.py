# ============================================================
# OOD INTERFERENCE SLICE GENERATOR v17
#
# WHY THIS EXISTS — the integration test exposed it, the v16
# source confirmed it: v16's "LTE" is 2 subcarriers at 312.5 kHz
# spacing and its "NR" is 1 subcarrier at 625 kHz spacing — i.e.
# 2-3 narrowband modulated TONES. Real LTE/NR uses 15/30 kHz
# subcarrier spacing: inside the PCM-FM Carson band a real
# co-channel signal has ~45-90 CONTIGUOUS subcarriers. The model
# (SpectralGate especially) learned narrowband-tone excision and
# collapses on wideband interference at the same SJR.
#
# THE FIX — domain randomization. This script generates a
# training slice using the v16 impairment chain VERBATIM
# (phase noise -> Doppler -> interference -> LNA -> IM3 -> CFO ->
# multipath -> AGC -> AWGN -> IQ -> ADC -> SCO), but draws the
# interference from a randomized realistic family:
#
#   ofdm_wide   P=0.50  nfft 128..1024 (SCS 9.8-78 kHz), contiguous
#                       allocation 0.3-4 MHz wide, per-RB loading,
#                       4/16/64/256-QAM, CP, bursty duty
#   filt_noise  P=0.20  band-limited Gaussian noise, random band
#   multitone   P=0.15  2-8 modulated tones (superset of v16 style)
#   chirp_swept P=0.10  linear FM sweep across a random span
#   pulsed_wb   P=0.05  low-duty wideband noise bursts
#
# Output: pcmfm_ood_interf_v17/{train,val,test}
#   train (15k) -> ADD to training dirs for the v8.3 fine-tune
#   val/test (1.5k each) -> OOD evaluation sets (original v16
#   val/test remain the primary in-distribution benchmark)
#
# Labels stay 26-element v16-compatible:
#   [2] lte_present  -> interferer-A present (always 1 here)
#   [3] sjr_lte_db   -> interferer-A SJR
#   [4] nr_present   -> interferer-B present
#   [5] sjr_nr_db    -> interferer-B SJR
#   [21] lte_n_sc    -> interferer-A class id (0..4 per list above)
#   [22] nr_mini_slot-> interferer-B class id (or 0)
# ============================================================

import json, math, os, sys, time
from pathlib import Path
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import gaussian_filter1d

# ══════════════════════════════════════════════════════════════
# SYSTEM CONSTANTS — identical to v16
# ══════════════════════════════════════════════════════════════
BITRATE  = 1e6
FS       = 10e6
SPB      = int(FS / BITRATE)
NUM_BITS = 256
SIG_LEN  = NUM_BITS * SPB
H_INDEX  = 0.70
FREQ_DEV = H_INDEX * BITRATE / 2
BT       = 0.50
SIGMA_G  = (math.sqrt(math.log(2)) / (2 * math.pi * BT)) * SPB
AGC_TARGET = -10.0

N_TRAIN, N_VAL, N_TEST = 15_000, 1_500, 1_500
SEED    = 1717
OUT_DIR = "pcmfm_ood_interf_v17"

# interference SJR ranges (match the v16 hard region emphasis)
SJR_A   = (-5.0, 15.0)      # interferer A — always present
SJR_B   = (-5.0, 20.0)      # interferer B — present 50%
P_B     = 0.50
SJR_OOB = (15.0, 40.0)

# v16 impairment parameter ranges — verbatim
SNR_RNG  = (8.0, 30.0)
BETA_RNG = (5.0, 200.0)
P1DB_RNG = (-25.0, -5.0)
IIP3_RNG = (10.0, 25.0)
ADC_BITS_L = [6, 8, 10, 12]
ADC_BITS_P = [0.10, 0.50, 0.25, 0.15]
ADC_HEAD = (0.0, 6.0)
CFO_MAX, DOP_MAX, SCO_MAX = 100_000, 25_000, 200.0
IQ_AMP_MAX, IQ_PHI_MAX = 3.0, 6.0
MP_DLY_MAX, MP_ALP_MAX = 5, 0.5
P_IM3, P_MPATH, P_CFO, P_DOP, P_SCO, P_IQ = 0.60, 0.65, 0.90, 0.85, 0.80, 0.80

IF_CLASSES = ["ofdm_wide", "filt_noise", "multitone", "chirp_swept", "pulsed_wb"]
IF_PROBS   = [0.50, 0.20, 0.15, 0.10, 0.05]


# ══════════════════════════════════════════════════════════════
# HELPERS — verbatim from v16
# ══════════════════════════════════════════════════════════════
def norm(x):
    return x / (math.sqrt(np.mean(np.abs(x) ** 2)) + 1e-12)

def sjr_amp(sjr_db_v):
    return 10.0 ** (-sjr_db_v / 20.0)

def power_dbm(x):
    return 10.0 * math.log10(np.mean(np.abs(x) ** 2) * 1e3 + 1e-30)

def sjr_db(p_s, p_j):
    return 99.0 if p_j < 1e-20 else 10.0 * math.log10(p_s / p_j)


# ══════════════════════════════════════════════════════════════
# v16 CHAIN STAGES — verbatim
# ══════════════════════════════════════════════════════════════
def gen_pcmfm(rng):
    bits = rng.integers(0, 2, NUM_BITS).astype(np.int8)
    nrz = (2.0 * bits - 1.0).astype(np.float64)
    shaped = gaussian_filter1d(np.repeat(nrz, SPB), SIGMA_G)
    phase = 2 * math.pi * FREQ_DEV * cumulative_trapezoid(shaped, dx=1 / FS,
                                                          initial=0)
    return norm(np.exp(1j * phase).astype(np.complex64)), bits


def apply_phase_noise(sig, rng, beta):
    phi = np.cumsum(rng.normal(0, math.sqrt(2 * math.pi * beta / FS), SIG_LEN))
    return (sig * np.exp(1j * phi)).astype(np.complex64)


def apply_doppler(sig, fd_hz):
    t = np.arange(SIG_LEN, dtype=np.float64) / FS
    return (sig * np.exp(1j * 2 * math.pi * fd_hz * t)).astype(np.complex64)


def gen_oob(rng):
    noise = rng.normal(0, 1, SIG_LEN) + 1j * rng.normal(0, 1, SIG_LEN)
    freqs = np.fft.fftfreq(SIG_LEN, 1 / FS)
    dist = np.maximum(0.85 * (FS / 2) - np.abs(freqs), 0) / 1e6
    shape = 10 ** (-rng.uniform(0.2, 0.5) * dist / 10)
    return norm(np.fft.ifft(np.fft.fft(noise) *
                            np.sqrt(np.abs(shape))).astype(np.complex64))


def apply_lna(composite, p1db_dbm, p_rapp=2.0):
    v_p1db = math.sqrt(10 ** (p1db_dbm / 10) * 1e-3 * 50)
    v_sat = v_p1db * (2 * p_rapp / (2 * p_rapp - 1)) ** (1 / (2 * p_rapp))
    v_rms = math.sqrt(np.mean(np.abs(composite) ** 2) * 50e-3)
    if v_rms < 1e-15:
        return composite, 1.0
    scale = v_rms / (math.sqrt(np.mean(np.abs(composite) ** 2)) + 1e-15)
    sv = composite * scale
    amp = np.abs(sv)
    comp = 1 / (1 + (amp / v_sat) ** (2 * p_rapp)) ** (1 / (2 * p_rapp))
    return norm((sv * comp).astype(np.complex64)), float(comp.mean())


def apply_im3(sig, rng):
    iip3_dbm = float(rng.uniform(*IIP3_RNG))
    t = np.arange(SIG_LEN) / FS
    f1, f2 = 0.40 * (FS / 2), 0.43 * (FS / 2)
    iip3_w = 10 ** (iip3_dbm / 10) * 1e-3
    v_iip3 = math.sqrt(iip3_w * 50) * math.sqrt(2)
    A = float(np.clip(1 / (4 * v_iip3 ** 2), 0, 0.5))
    phi = rng.uniform(0, 2 * math.pi, 2)
    spur = (A * np.exp(1j * (2 * math.pi * (2 * f1 - f2) * t + phi[0])) +
            A * np.exp(1j * (2 * math.pi * (2 * f2 - f1) * t + phi[1])))
    return norm((sig + spur.astype(np.complex64))), iip3_dbm


def apply_cfo(sig, cfo_hz):
    t = np.arange(SIG_LEN, dtype=np.float64) / FS
    return (sig * np.exp(1j * 2 * math.pi * cfo_hz * t)).astype(np.complex64)


def apply_multipath(sig, rng):
    n = int(rng.integers(1, 4))
    delays = sorted(rng.integers(1, MP_DLY_MAX + 1, size=n).tolist())
    alphas = [float(rng.uniform(0.1, MP_ALP_MAX)) for _ in range(n)]
    out = sig.copy().astype(np.complex128)
    for d, a in zip(delays, alphas):
        out += a * np.concatenate([np.zeros(d, dtype=complex),
                                   sig.astype(complex)[:-d]])
    return norm(out.astype(np.complex64)), n, max(delays), max(alphas)


def apply_agc(sig):
    p_mw = np.mean(np.abs(sig) ** 2) * 1e3
    p_tgt_mw = 10 ** (AGC_TARGET / 10)
    gain_lin = p_tgt_mw / (p_mw + 1e-30)
    gain_db = 10 * math.log10(gain_lin + 1e-30)
    return (sig * math.sqrt(gain_lin)).astype(np.complex64), gain_db


def apply_awgn(sig, rng, snr_db_v, p_ref_w=None):
    p_ref = p_ref_w if p_ref_w is not None else float(np.mean(np.abs(sig) ** 2))
    n_pwr = p_ref / 10 ** (snr_db_v / 10)
    n = math.sqrt(n_pwr / 2) * (rng.normal(0, 1, SIG_LEN) +
                                1j * rng.normal(0, 1, SIG_LEN))
    return (sig + n).astype(np.complex64)


def apply_iq_imbalance(sig, amp_db, phi_deg):
    if amp_db < 0.01 and abs(phi_deg) < 0.1:
        return sig.copy().astype(np.complex64)
    eps = 10 ** (amp_db / 20) - 1
    phi = math.radians(phi_deg)
    I, Q = sig.real, sig.imag
    return ((1 + eps / 2) * I +
            1j * (math.sin(phi) * I +
                  math.cos(phi) * (1 - eps / 2) * Q)).astype(np.complex64)


def apply_adc(sig, n_bits, headroom_db):
    peak = float(np.abs(sig).max())
    if peak < 1e-12:
        return sig.copy().astype(np.complex64)
    vf = peak * 10 ** (headroom_db / 20)
    step = 2 * vf / 2 ** n_bits
    Iq = np.clip(np.round(sig.real / step) * step, -vf, vf)
    Qq = np.clip(np.round(sig.imag / step) * step, -vf, vf)
    return (Iq + 1j * Qq).astype(np.complex64)


def apply_sco(sig, sco_ppm):
    if abs(sco_ppm) < 0.5:
        return sig.copy().astype(np.complex64)
    delta = sco_ppm * 1e-6
    t = np.arange(SIG_LEN, dtype=np.float64)
    t2 = np.clip(t * (1 + delta), 0, SIG_LEN - 1.001)
    return (np.interp(t2, t, sig.real) +
            1j * np.interp(t2, t, sig.imag)).astype(np.complex64)


# ══════════════════════════════════════════════════════════════
# NEW — RANDOMIZED REALISTIC INTERFERENCE FAMILY
# ══════════════════════════════════════════════════════════════
def _qam_points(order):
    m = int(math.sqrt(order))
    lvl = np.arange(-(m - 1), m, 2, dtype=np.float64)
    pts = np.array([a + 1j * b for a in lvl for b in lvl])
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


def _burst_mask(rng, duty_lo=0.5, duty_hi=1.0, floor_db=-10.0):
    duty = float(rng.uniform(duty_lo, duty_hi))
    blen = max(1, int(duty * SIG_LEN))
    bst = int(rng.integers(0, max(1, SIG_LEN - blen)))
    mask = np.zeros(SIG_LEN, dtype=np.float64)
    mask[bst:bst + blen] = 1.0
    return mask + 10 ** (floor_db / 20) * (1 - mask)


def gen_ofdm_wide(rng):
    """Realistic CP-OFDM: dense contiguous allocation, narrow SCS.
    Spans the SCS bracket around real LTE/NR (9.8–78 kHz)."""
    nfft = int(rng.choice([128, 256, 512, 1024]))
    scs = FS / nfft
    cp = max(1, int(0.07 * nfft))
    sym = nfft + cp
    n_sym = SIG_LEN // sym + 2
    bw = float(rng.uniform(0.3e6, 4.0e6))
    n_sc = int(np.clip(bw / scs, 4, nfft // 2 - 4))
    fc = float(rng.uniform(-1, 1)) * max(0.0, (FS / 2 - bw / 2) * 0.55)
    c0 = int(round(fc / scs))
    sc_idx = np.arange(c0 - n_sc // 2, c0 - n_sc // 2 + n_sc)
    sc_idx = sc_idx[(sc_idx > -nfft // 2) & (sc_idx < nfft // 2) & (sc_idx != 0)]
    # per-RB (12-SC) loading: not every RB scheduled every symbol
    rb = np.array_split(sc_idx, max(1, len(sc_idx) // 12))
    pts = _qam_points(int(rng.choice([4, 16, 64, 256])))
    p_rb_on = float(rng.uniform(0.6, 1.0))
    stream = np.zeros(n_sym * sym, dtype=np.complex128)
    for s in range(n_sym):
        fd = np.zeros(nfft, dtype=np.complex128)
        for blk in rb:
            if rng.random() < p_rb_on and len(blk):
                fd[blk % nfft] = rng.choice(pts, size=len(blk))
        td = np.fft.ifft(fd) * math.sqrt(nfft)
        stream[s * sym:(s + 1) * sym] = np.concatenate([td[-cp:], td])
    out = stream[:SIG_LEN] * _burst_mask(rng)
    return norm(out.astype(np.complex64))


def gen_filt_noise(rng):
    """Band-limited Gaussian noise in a random band."""
    bw = float(rng.uniform(0.2e6, 4.0e6))
    fc = float(rng.uniform(-1, 1)) * max(0.0, (FS / 2 - bw / 2) * 0.6)
    n = rng.normal(0, 1, SIG_LEN) + 1j * rng.normal(0, 1, SIG_LEN)
    F = np.fft.fft(n)
    f = np.fft.fftfreq(SIG_LEN, 1 / FS)
    F[np.abs(f - fc) > bw / 2] = 0
    out = np.fft.ifft(F) * _burst_mask(rng, 0.6, 1.0)
    return norm(out.astype(np.complex64))


def gen_multitone(rng):
    """2-8 modulated tones (generalizes the v16 LTE/NR style)."""
    k = int(rng.integers(2, 9))
    t = np.arange(SIG_LEN) / FS
    pts = _qam_points(int(rng.choice([4, 16, 256])))
    out = np.zeros(SIG_LEN, dtype=np.complex128)
    for _ in range(k):
        f0 = float(rng.uniform(-2.0e6, 2.0e6))
        sym_len = int(rng.choice([22, 34, 39, 68]))   # v16-ish symbol rates
        n_sym = SIG_LEN // sym_len + 1
        syms = np.repeat(rng.choice(pts, size=n_sym), sym_len)[:SIG_LEN]
        out += syms * np.exp(1j * 2 * math.pi * f0 * t)
    out *= _burst_mask(rng, 0.6, 1.0)
    return norm(out.astype(np.complex64))


def gen_chirp_swept(rng):
    """Linear FM sweep across a random span (radar-like sweeper)."""
    f0 = float(rng.uniform(-3.0e6, 1.0e6))
    f1 = f0 + float(rng.uniform(0.5e6, 3.5e6))
    n_rep = int(rng.integers(1, 5))
    seg = SIG_LEN // n_rep
    t = np.arange(seg) / FS
    k = (f1 - f0) / (seg / FS)
    one = np.exp(1j * 2 * math.pi * (f0 * t + 0.5 * k * t ** 2))
    out = np.tile(one, n_rep + 1)[:SIG_LEN]
    return norm(out.astype(np.complex64))


def gen_pulsed_wb(rng):
    """Low-duty wideband noise bursts (impulsive)."""
    out = np.zeros(SIG_LEN, dtype=np.complex128)
    n_pulse = int(rng.integers(2, 7))
    for _ in range(n_pulse):
        w = int(rng.integers(40, 400))
        s = int(rng.integers(0, SIG_LEN - w))
        out[s:s + w] = (rng.normal(0, 1, w) + 1j * rng.normal(0, 1, w))
    return norm(out.astype(np.complex64))


GEN_FN = {"ofdm_wide": gen_ofdm_wide, "filt_noise": gen_filt_noise,
          "multitone": gen_multitone, "chirp_swept": gen_chirp_swept,
          "pulsed_wb": gen_pulsed_wb}


def gen_random_interferer(rng):
    cls = str(rng.choice(IF_CLASSES, p=IF_PROBS))
    return GEN_FN[cls](rng), IF_CLASSES.index(cls)


# ══════════════════════════════════════════════════════════════
# SAMPLE GENERATION — v16 chain, randomized interference
# ══════════════════════════════════════════════════════════════
def generate_sample(rng):
    tx, bits = gen_pcmfm(rng)
    p_sig = 1.0

    beta_hz = float(rng.uniform(*BETA_RNG))
    tx_pn = apply_phase_noise(tx, rng, beta_hz)
    x_clean = tx_pn.copy()

    dop_hz = float(rng.uniform(-DOP_MAX, DOP_MAX)) if rng.random() < P_DOP else 0.0
    composite = apply_doppler(tx_pn, dop_hz)
    base = composite.copy()

    # interferer A — always present (this slice exists to teach it)
    sjr_a = float(rng.uniform(*SJR_A))
    jA, clsA = gen_random_interferer(rng)
    termA = sjr_amp(sjr_a) * jA
    composite = composite + termA
    sjr_a_v = sjr_db(p_sig, float(np.mean(np.abs(termA) ** 2)))

    # interferer B — 50%
    b_present, clsB, sjr_b_v = False, 0, 99.0
    if rng.random() < P_B:
        b_present = True
        sjr_b = float(rng.uniform(*SJR_B))
        jB, clsB = gen_random_interferer(rng)
        termB = sjr_amp(sjr_b) * jB
        composite = composite + termB
        sjr_b_v = sjr_db(p_sig, float(np.mean(np.abs(termB) ** 2)))

    sjr_oob = float(rng.uniform(*SJR_OOB))
    composite = composite + sjr_amp(sjr_oob) * gen_oob(rng)

    p_jam = float(np.mean(np.abs(composite - base) ** 2))
    sjr_comp = sjr_db(p_sig, p_jam)
    comp_dbm = power_dbm(composite)

    p1db_dbm = float(rng.uniform(*P1DB_RNG))
    rx, lna_comp = apply_lna(composite, p1db_dbm)

    im3_present = False
    if rng.random() < P_IM3:
        im3_present = True
        rx, _ = apply_im3(rx, rng)

    cfo_hz = float(rng.uniform(-CFO_MAX, CFO_MAX)) if rng.random() < P_CFO else 0.0
    rx = apply_cfo(rx, cfo_hz)

    mp_n = mp_d = 0; mp_a = 0.0
    if rng.random() < P_MPATH:
        rx, mp_n, mp_d, mp_a = apply_multipath(rx, rng)

    rx, agc_gain_db = apply_agc(rx)
    snr = float(rng.uniform(*SNR_RNG))
    rx = apply_awgn(rx, rng, snr, float(np.mean(np.abs(rx) ** 2)))

    amp_db = float(rng.uniform(0, IQ_AMP_MAX)) if rng.random() < P_IQ else 0.0
    phi_deg = float(rng.uniform(0, IQ_PHI_MAX)) if rng.random() < P_IQ else 0.0
    rx = apply_iq_imbalance(rx, amp_db, phi_deg)

    adc_n = int(rng.choice(ADC_BITS_L, p=ADC_BITS_P))
    head_db = float(rng.uniform(*ADC_HEAD))
    rx = apply_adc(rx, adc_n, head_db)

    sco_ppm = float(rng.uniform(0, SCO_MAX)) if rng.random() < P_SCO else 0.0
    x_noisy = apply_sco(rx, (1.0 if rng.random() < 0.5 else -1.0) * sco_ppm)

    labels = np.array([
        snr, sjr_comp,
        1.0, sjr_a_v,                       # interferer A in the LTE slots
        float(b_present), sjr_b_v,          # interferer B in the NR slots
        sjr_oob, p1db_dbm, float(im3_present),
        beta_hz, agc_gain_db, cfo_hz, dop_hz, sco_ppm,
        amp_db, phi_deg, float(adc_n), head_db,
        float(mp_n), float(mp_d), mp_a,
        float(clsA), float(clsB), 0.0,
        comp_dbm, lna_comp,
    ], dtype=np.float32)

    return {"x_clean": x_clean.astype(np.complex64),
            "x_noisy": x_noisy.astype(np.complex64),
            "bits": bits, "labels": labels}


# ══════════════════════════════════════════════════════════════
# BUILDER
# ══════════════════════════════════════════════════════════════
def to_iq(z):
    return np.stack([z.real, z.imag], axis=-1).astype(np.float32)


def smoke_test():
    print("  Smoke test ...")
    rng = np.random.default_rng(0)
    seen = set()
    for i in range(20):
        s = generate_sample(rng)
        lb = s["labels"]
        assert s["x_noisy"].shape == (SIG_LEN,)
        assert s["bits"].shape == (NUM_BITS,)
        assert lb.shape == (26,)
        assert not np.any(np.isnan(s["x_noisy"]))
        assert not np.any(np.isnan(lb))
        seen.add(int(lb[21]))
    print(f"  PASSED — interferer-A classes seen: "
          f"{sorted(IF_CLASSES[c] for c in seen)}")
    return True


def build(out_dir=OUT_DIR, seed=SEED):
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    splits = {"train": N_TRAIN, "val": N_VAL, "test": N_TEST}
    arrays = {}
    for sp, n in splits.items():
        d = out / sp
        d.mkdir(parents=True, exist_ok=True)
        arrays[sp] = {
            "X_clean": np.lib.format.open_memmap(str(d / "X_clean.npy"),
                mode="w+", dtype=np.float32, shape=(n, SIG_LEN, 2)),
            "X_noisy": np.lib.format.open_memmap(str(d / "X_noisy.npy"),
                mode="w+", dtype=np.float32, shape=(n, SIG_LEN, 2)),
            "y_bits": np.lib.format.open_memmap(str(d / "y_bits.npy"),
                mode="w+", dtype=np.int8, shape=(n, NUM_BITS)),
            "labels": np.lib.format.open_memmap(str(d / "labels.npy"),
                mode="w+", dtype=np.float32, shape=(n, 26)),
        }
    for sp, n in splits.items():
        t0 = time.time()
        for i in range(n):
            s = generate_sample(rng)
            arrays[sp]["X_clean"][i] = to_iq(s["x_clean"])
            arrays[sp]["X_noisy"][i] = to_iq(s["x_noisy"])
            arrays[sp]["y_bits"][i] = s["bits"]
            arrays[sp]["labels"][i] = s["labels"]
            if (i + 1) % 1000 == 0 or i + 1 == n:
                el = time.time() - t0
                print(f"\r  {sp:6s} {i+1:>6,}/{n:,}  {el:.0f}s "
                      f"eta {el/(i+1)*(n-i-1):.0f}s ", end="", flush=True)
        print()
    meta = {"version": "ood_interf_v17", "seed": seed,
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "interferer_classes": IF_CLASSES, "class_probs": IF_PROBS,
            "sjr_a": SJR_A, "sjr_b": SJR_B, "p_b": P_B,
            "note": ("v16 impairment chain verbatim; interference replaced "
                     "by randomized realistic family. Labels v16-compatible; "
                     "lbl[21]/lbl[22] hold interferer class ids."),
            "splits": splits}
    with open(out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved -> {out.resolve()}")


if __name__ == "__main__":
    print("=" * 65)
    print("  OOD INTERFERENCE SLICE v17 — domain-randomized interference")
    print("=" * 65)
    if not smoke_test():
        sys.exit(1)
    build()
