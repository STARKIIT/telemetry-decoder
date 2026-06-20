# ============================================================
# END-TO-END CODED TELEMETRY DEMO — v8.5 + LDPC  (demo_v8_5_fec.py)
# ============================================================
# The full stack, live, as one pipeline:
#
#   text -> ASCII bits -> LDPC ENCODE (rate 1/2) -> scramble ->
#   split across N sync-bearing frames -> PCM-FM -> continuous
#   stream w/ unknown offset/phase/CFO + interference + LNA +
#   AWGN  ->  [per frame] sync acquire -> derotate -> v8.5 ->
#   calibrated LLRs  ->  concatenate -> min-sum LDPC DECODE ->
#   descramble -> recovered text.
#
# This is the first artifact where sync + model + calibrated
# LLRs + FEC all run together. The payoff: on hard interference
# the per-frame panels show red pre-FEC bit errors, and the
# decoded message still comes out EXACTLY right.
#
# REQUIRES (same folder): integration_test_v1.py,
# sync_frontend_v1.py, ood_slice_generator_v17.py,
# llr_ldpc_harness_v8_1.py
# ============================================================

import os
import sys
import math
import importlib.util
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHECKPOINT = "/content/drive/MyDrive/pcmfm_dataset_v16_5G_May/telemetry_v8_5.pth"
PAD        = 600
CODE_N, CODE_DV, CODE_DC = 2046, 3, 6     # rate-1/2 proxy code (k=1025)
CODE_SEED  = 7
TEMPERATURE = 0.55                         # global-ish T; overridable
LLR_CLIP   = 24.0
DECODE_ITERS = 50
ALPHA      = 0.75


def _load(name):
    for cand in (name, f"/content/{name}", f"/content/drive/MyDrive/{name}"):
        if os.path.exists(cand):
            spec = importlib.util.spec_from_file_location(name[:-3], cand)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name[:-3]] = mod
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(name)


it = _load("integration_test_v1.py")
g  = _load("ood_slice_generator_v17.py")
hh = _load("llr_ldpc_harness_v8_1.py")
sf = it.load_sync_module()
NUM_BITS, SIG_LEN, FS = it.NUM_BITS, it.SIG_LEN, sf.FS
SYNC_LEN_BITS = len(sf.SYNC_BITS)
PAYLOAD_BITS  = NUM_BITS - SYNC_LEN_BITS          # 224 coded bits per frame


# ────────────────────────────────────────────────────────────
# LDPC encoder (systematic G derived from the proxy H over GF(2))
# ────────────────────────────────────────────────────────────
def _build_H(n, dv, dc, seed):
    m, ci, vi = hh.build_regular_code(n, dv, dc, np.random.RandomState(seed))
    H = np.zeros((m, n), dtype=np.uint8)
    H[ci, vi] ^= 1
    return H, m, ci, vi


def _systematic_G(H):
    H = H.copy().astype(np.uint8); m, n = H.shape
    piv_cols = []; r = 0
    for c in range(n):
        if r >= m:
            break
        rows = np.where(H[r:, c] == 1)[0]
        if len(rows) == 0:
            continue
        pr = r + rows[0]
        H[[r, pr]] = H[[pr, r]]
        for rr in range(m):
            if rr != r and H[rr, c] == 1:
                H[rr] ^= H[r]
        piv_cols.append(c); r += 1
    rank = r
    piv = set(piv_cols)
    info_cols = [c for c in range(n) if c not in piv]
    Hi = H[:rank][:, info_cols]
    k = len(info_cols)
    G = np.zeros((k, n), dtype=np.uint8)
    for i, ic in enumerate(info_cols):
        G[i, ic] = 1
    for i in range(k):
        for j, pc in enumerate(piv_cols):
            G[i, pc] = Hi[j, i]
    return G, np.array(info_cols)


_H, _M, _CI, _VI = _build_H(CODE_N, CODE_DV, CODE_DC, CODE_SEED)
_G, _INFO = _systematic_G(_H)
CODE_K = len(_INFO)


def ldpc_encode(info_bits):
    u = np.asarray(info_bits, dtype=np.uint8)[:CODE_K]
    if len(u) < CODE_K:
        u = np.concatenate([u, np.zeros(CODE_K - len(u), dtype=np.uint8)])
    return (u @ _G) % 2


def ldpc_decode(llr_codeword, device):
    chan = torch.from_numpy(llr_codeword[None, :CODE_N]).float().to(device)
    post = hh.minsum_decode(chan, CODE_N, _M, _CI, _VI, CODE_DC, device,
                            iters=DECODE_ITERS, alpha=ALPHA)
    hard = (post[0] < 0).long().cpu().numpy().astype(np.uint8)
    return hard[_INFO]                                  # systematic info bits


# ────────────────────────────────────────────────────────────
# ASCII + scrambler
# ────────────────────────────────────────────────────────────
def string_to_bits(text, nbits):
    max_chars = nbits // 8
    if len(text) > max_chars:
        print(f"  [WARN] truncated to {max_chars} chars")
        text = text[:max_chars]
    text = text + "\0" * (max_chars - len(text))
    out = []
    for c in text:
        b = ord(c) & 0xFF
        out += [(b >> i) & 1 for i in range(7, -1, -1)]
    return np.array(out, dtype=np.uint8)


def bits_to_string(bits):
    out = []
    for i in range(0, len(bits) - 7, 8):
        v = 0
        for j in range(8):
            v = (v << 1) | int(bits[i + j])
        out.append(chr(v))
    return "".join(out).rstrip("\x00")


def scramble(bits, seed=0xACE1):
    lfsr = seed & 0xFFFF
    out = np.zeros_like(bits)
    for i in range(len(bits)):
        sb = ((lfsr >> 14) ^ (lfsr >> 13)) & 1
        out[i] = bits[i] ^ sb
        lfsr = ((lfsr << 1) | sb) & 0xFFFF
    return out


descramble = scramble


def make_interference(mode, n, rng):
    if mode == "none":
        return None
    if mode == "v16_tones":
        return it._lte_v16(n, rng)
    if mode == "wideband":
        return it._ofdm(n, rng)
    if mode == "tones_cont":
        fft_size = 32; cp = max(1, int(0.0694 * fft_size)); sym = fft_size + cp
        n_sym = n // sym + 2
        off = rng.integers(-2, 3)
        sc = np.clip([fft_size // 2 + off - 1, fft_size // 2 + off + 1],
                     1, fft_size - 1)
        lvl = np.array([-3, -1, 1, 3]) / math.sqrt(10)
        pts = np.array([a + 1j * b for a in lvl for b in lvl])
        st = np.zeros(n_sym * sym, dtype=np.complex128)
        for s in range(n_sym):
            fd = np.zeros(fft_size, dtype=np.complex128)
            fd[sc] = rng.choice(pts, size=len(sc))
            td = np.fft.ifft(np.fft.ifftshift(fd))
            st[s * sym:(s + 1) * sym] = np.concatenate([td[-cp:], td])
        st = st[:n]
        return st / (np.sqrt(np.mean(np.abs(st) ** 2)) + 1e-15)
    raise ValueError(mode)


# ────────────────────────────────────────────────────────────
# transmit ONE frame of payload bits through the channel, return
# the model's calibrated LLRs + diagnostics
# ────────────────────────────────────────────────────────────
def tx_rx_frame(payload_bits, mode, sjr_db, snr_db, rng, model, device, T):
    tx_bits = np.concatenate([sf.SYNC_BITS, payload_bits]).astype(np.int64)
    sig = sf.pcmfm_iq(tx_bits, phi0=rng.uniform(0, 2 * np.pi))
    pre = sf.pcmfm_iq(rng.integers(0, 2, PAD // sf.SPB + 1))[:PAD]
    post = sf.pcmfm_iq(rng.integers(0, 2, PAD // sf.SPB + 1))[:PAD]
    stream = np.concatenate([pre, sig, post])
    j = make_interference(mode, len(stream), rng)
    if j is not None:
        stream = stream + j * 10 ** (-sjr_db / 20)
    stream, _ = g.apply_lna(stream.astype(np.complex64),
                            float(rng.uniform(*g.P1DB_RNG)))
    f_true = float(rng.uniform(-80, 80)) * 1e3
    stream = sf.apply_cfo(stream, f_true)
    p_ref = float(np.mean(np.abs(stream) ** 2))
    npow = p_ref / 10 ** (snr_db / 10)
    stream = stream + (rng.standard_normal(len(stream)) +
                       1j * rng.standard_normal(len(stream))) * np.sqrt(npow / 2)

    est, f_hat, _, psl = sf.find_frame_start(stream)
    x_rx = it.extract_frame(sf, stream, est, f_hat)
    with torch.no_grad():
        _, logits, _ = model(torch.from_numpy(x_rx).unsqueeze(0).to(device))
    lg = logits.squeeze(0).float().cpu().numpy()
    pred = (lg > 0).astype(np.uint8)
    # LLR (+ favours bit 0): llr = -logit / T, clipped
    llr_payload = np.clip(-lg[SYNC_LEN_BITS:] / T, -LLR_CLIP, LLR_CLIP)
    return {"llr": llr_payload.astype(np.float32),
            "pred_payload": pred[SYNC_LEN_BITS:],
            "tx_payload": payload_bits.astype(np.uint8),
            "sync_err": int(abs(est - PAD)), "psl": psl}


# ────────────────────────────────────────────────────────────
# full coded message demo
# ────────────────────────────────────────────────────────────
def run_coded_demo(message, interference="tones_cont", sjr_db=0.0,
                   snr_db=15.0, seed=42, model=None, device=None,
                   T=TEMPERATURE, fig_path="demo_fec.png"):
    rng = np.random.default_rng(seed)
    if model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(CHECKPOINT, map_location=device)
        model = it.TelemetryPipelineV8(it.CFG).to(device).eval()
        model.load_state_dict(ckpt["model_state"])

    msg_char_cap = CODE_K // 8
    print("=" * 68)
    print(f"  MESSAGE: '{message}'  ({len(message)} chars, capacity {msg_char_cap})")
    print(f"  channel: {interference}"
          + (f" @ SJR {sjr_db:+.0f} dB" if interference != "none" else "")
          + f", SNR {snr_db:.0f} dB | rate-1/2 LDPC, n={CODE_N} k={CODE_K}")
    print("=" * 68)

    # encode: text -> info bits -> LDPC codeword -> scramble
    info = string_to_bits(message, CODE_K)
    codeword = ldpc_encode(info)
    tx = scramble(codeword)

    # split codeword across frames of PAYLOAD_BITS each
    n_frames = int(np.ceil(len(tx) / PAYLOAD_BITS))
    padded = np.concatenate([tx, np.zeros(n_frames * PAYLOAD_BITS - len(tx),
                                          dtype=np.uint8)])
    frames = padded.reshape(n_frames, PAYLOAD_BITS)
    print(f"  codeword spans {n_frames} sync-bearing frames "
          f"({n_frames * NUM_BITS} transmitted bits)\n")

    # transmit each frame, collect LLRs + per-frame raw stats
    llr_stream = np.zeros(n_frames * PAYLOAD_BITS, dtype=np.float32)
    raw_errs = []; sync_errs = []; psls = []
    panels = []
    for fi in range(n_frames):
        r = tx_rx_frame(frames[fi], interference, sjr_db, snr_db, rng,
                        model, device, T)
        llr_stream[fi * PAYLOAD_BITS:(fi + 1) * PAYLOAD_BITS] = r["llr"]
        e = int((r["pred_payload"] != r["tx_payload"]).sum())
        raw_errs.append(e); sync_errs.append(r["sync_err"]); psls.append(r["psl"])
        if fi < 4:
            panels.append((r["tx_payload"], r["pred_payload"], e))
        print(f"    frame {fi+1}/{n_frames}: sync err {r['sync_err']:+d} samp | "
              f"PSL {r['psl']:.2f} | raw payload errors {e}/{PAYLOAD_BITS}")

    # de-scramble LLRs are tricky: scrambler flips bit value -> flips LLR sign.
    # Reconstruct the scramble sequence and apply sign flips, then trim to n.
    scr_seq = scramble(np.zeros(len(llr_stream), dtype=np.uint8))   # = key stream
    llr_descr = np.where(scr_seq[:len(llr_stream)] == 1,
                         -llr_stream, llr_stream)[:CODE_N]

    # decode
    dec_info = ldpc_decode(llr_descr, device)
    recovered = bits_to_string(dec_info[:len(message) * 8])

    total_raw = sum(raw_errs)
    pre_ber = total_raw / (n_frames * PAYLOAD_BITS)
    ok = recovered[:len(message)] == message
    print(f"\n  aggregate raw BER across frames: {pre_ber:.4f} "
          f"({total_raw} bit errors over {n_frames} frames)")
    print(f"  RECOVERED: '{recovered[:len(message)]}'")
    print("  " + ("✓ MESSAGE RECOVERED PERFECTLY (FEC corrected all "
                  f"{total_raw} raw errors)" if ok else
                  "✗ message not fully recovered — raw BER exceeded FEC budget"))

    _viz_coded(message, recovered[:len(message)], panels, raw_errs, psls,
               interference, sjr_db, pre_ber, ok, total_raw, fig_path)
    return ok, model, device


def _viz_coded(orig, recv, panels, raw_errs, psls, mode, sjr_db,
               pre_ber, ok, total_raw, fig_path):
    safe = lambda s: "".join(c if 32 <= ord(c) < 127 else "·" for c in s)
    n_panel = len(panels)
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, max(n_panel, 2))

    # row 0-1: per-frame bit panels (first up-to-4 frames)
    for i, (tx, pr, e) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        nb = 96
        ax.step(range(nb), tx[:nb], where="post", lw=1.4, label="sent")
        ax.step(range(nb), pr[:nb], where="post", lw=1.0, ls="--",
                label="decoded")
        bad = np.where(tx[:nb] != pr[:nb])[0]
        if len(bad):
            ax.scatter(bad, tx[bad], color="red", s=40, zorder=5)
        ax.set_title(f"Frame {i+1}: {e} raw errors", fontsize=10)
        ax.set_ylim(-.2, 1.2); ax.set_xticks([]); ax.set_yticks([0, 1])
        if i == 0:
            ax.legend(fontsize=7, loc="center right")

    # row 1: per-frame raw error bar chart
    axb = fig.add_subplot(gs[1, :])
    axb.bar(range(1, len(raw_errs) + 1), raw_errs, color="#c0504d")
    axb.set_title("Raw bit errors per frame (BEFORE FEC)")
    axb.set_xlabel("Frame"); axb.set_ylabel("bit errors")
    axb.grid(alpha=.3, axis="y")

    # row 2: the verdict panel
    axv = fig.add_subplot(gs[2, :]); axv.axis("off")
    color = "#cdeb8b" if ok else "#f4a6a6"
    txt = (f"CHANNEL:  {mode} @ SJR {sjr_db:+.0f} dB\n"
           f"PIPELINE: text → LDPC encode → scramble → "
           f"{len(raw_errs)}× (PCM-FM → interference → sync → v8.5 → LLR) "
           f"→ decode → text\n\n"
           f"SENT:       '{safe(orig)}'\n"
           f"RECOVERED:  '{safe(recv)}'\n\n"
           f"Raw aggregate BER (pre-FEC): {pre_ber:.4f}   "
           f"({total_raw} bit errors)\n"
           f"After LDPC: {'PERFECT — all errors corrected ✓' if ok else 'errors remain ✗'}")
    axv.text(0.02, 0.5, txt, transform=axv.transAxes, fontsize=12,
             family="monospace", va="center",
             bbox=dict(boxstyle="round,pad=0.6", facecolor=color,
                       edgecolor="black"))
    fig.suptitle("End-to-end coded telemetry recovery "
                 "(sync + v8.5 model + LDPC)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure -> {fig_path}")


# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = device = None
    examples = [
        ("ALTITUDE 35000 FT  SPEED 480 KTS  HEADING 270",
         "wideband", 0, 15, "demo_fec_wideband.png"),
        ("MAYDAY MAYDAY ENGINE 2 FLAMEOUT FUEL 12 PERCENT",
         "tones_cont", 0, 15, "demo_fec_tones_cont.png"),
    ]
    res = []
    for msg, mode, sjr, snr, fp in examples:
        print("\n" + "▼" * 68)
        ok, model, device = run_coded_demo(msg, interference=mode, sjr_db=sjr,
                                           snr_db=snr, seed=42, model=model,
                                           device=device, fig_path=fp)
        res.append((msg, mode, ok))
    print("\n" + "=" * 68)
    print("  SUMMARY")
    for msg, mode, ok in res:
        print(f"  {'✓' if ok else '✗'}  [{mode}]  '{msg}'")
    print("=" * 68)
