# ============================================================
# SIGNAL-JOURNEY DEMO — v8.5 + LDPC  (demo_journey.py)
# ============================================================
# Type a message and SEE its whole journey:
#   (1) the clean PCM-FM signal you transmit
#   (2) the interference/noise that hits it
#   (3) the corrupted signal the receiver actually gets
#   (4) the signal the model reconstructs
#   (5) the recovered text — error-free after FEC
#
# Shown in BOTH time domain (what the waveform looks like) and
# frequency domain (where the interference sits vs the signal).
#
# Two entry points:
#   demo("YOUR TEXT", interference="wideband", sjr_db=0)   # one call
#   demo_interactive()                                     # type at a prompt
#
# REQUIRES (same folder): integration_test_v1.py,
# sync_frontend_v1.py, ood_slice_generator_v17.py,
# llr_ldpc_harness_v8_1.py, demo_v8_5_fec.py
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
import matplotlib.gridspec as gridspec


def _load(name):
    for cand in (name, f"/content/{name}", f"/content/drive/MyDrive/{name}"):
        if os.path.exists(cand):
            spec = importlib.util.spec_from_file_location(name[:-3], cand)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name[:-3]] = mod
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(name)


# reuse the coded-demo machinery (encoder, scrambler, ASCII, channel, model)
fec = _load("demo_v8_5_fec.py")
it, g, sf, hh = fec.it, fec.g, fec.sf, fec.hh
FS, SIG_LEN, PAD = fec.FS, fec.SIG_LEN, fec.PAD
SYNC_LEN_BITS, PAYLOAD_BITS = fec.SYNC_LEN_BITS, fec.PAYLOAD_BITS

_MODEL = {"m": None, "dev": None}


def _get_model():
    if _MODEL["m"] is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(fec.CHECKPOINT, map_location=dev)
        m = it.TelemetryPipelineV8(it.CFG).to(dev).eval()
        m.load_state_dict(ckpt["model_state"])
        _MODEL["m"], _MODEL["dev"] = m, dev
    return _MODEL["m"], _MODEL["dev"]


# ────────────────────────────────────────────────────────────
# transmit one frame, KEEPING every intermediate signal for the plot
# ────────────────────────────────────────────────────────────
def _tx_rx_capture(payload_bits, mode, sjr_db, snr_db, rng, model, device, T):
    tx_bits = np.concatenate([sf.SYNC_BITS, payload_bits]).astype(np.int64)
    sig = sf.pcmfm_iq(tx_bits, phi0=rng.uniform(0, 2 * np.pi))   # CLEAN frame
    pre = sf.pcmfm_iq(rng.integers(0, 2, PAD // sf.SPB + 1))[:PAD]
    post = sf.pcmfm_iq(rng.integers(0, 2, PAD // sf.SPB + 1))[:PAD]
    stream = np.concatenate([pre, sig, post])

    interf = fec.make_interference(mode, len(stream), rng)        # INTERFERENCE
    interf_frame = (interf[PAD:PAD + SIG_LEN].copy()
                    if interf is not None else np.zeros(SIG_LEN, np.complex64))
    if interf is not None:
        stream = stream + interf * 10 ** (-sjr_db / 20)
    stream, _ = g.apply_lna(stream.astype(np.complex64),
                            float(rng.uniform(*g.P1DB_RNG)))
    f_true = float(rng.uniform(-80, 80)) * 1e3
    stream = sf.apply_cfo(stream, f_true)
    p_ref = float(np.mean(np.abs(stream) ** 2))
    npow = p_ref / 10 ** (snr_db / 10)
    stream = stream + (rng.standard_normal(len(stream)) +
                       1j * rng.standard_normal(len(stream))) * np.sqrt(npow / 2)

    est, f_hat, _, psl = sf.find_frame_start(stream)
    x_rx = it.extract_frame(sf, stream, est, f_hat)              # RECEIVED frame
    with torch.no_grad():
        x_recon_t, logits, _ = model(
            torch.from_numpy(x_rx).unsqueeze(0).to(device))
    x_recon = x_recon_t.squeeze(0).float().cpu().numpy()         # RECONSTRUCTED
    lg = logits.squeeze(0).float().cpu().numpy()
    pred = (lg > 0).astype(np.uint8)
    llr_payload = np.clip(-lg[SYNC_LEN_BITS:] / T, -fec.LLR_CLIP, fec.LLR_CLIP)

    cap = {"clean": sig[:SIG_LEN].astype(np.complex64),
           "interf": interf_frame.astype(np.complex64),
           "received": (x_rx[0] + 1j * x_rx[1]).astype(np.complex64),
           "recon": (x_recon[0] + 1j * x_recon[1]).astype(np.complex64),
           "psl": psl, "sync_err": int(abs(est - PAD))}
    return llr_payload.astype(np.float32), pred[SYNC_LEN_BITS:], cap


# ────────────────────────────────────────────────────────────
# main demo
# ────────────────────────────────────────────────────────────
def demo(message, interference="wideband", sjr_db=0.0, snr_db=15.0,
         seed=42, T=fec.TEMPERATURE, fig_path="signal_journey.png"):
    rng = np.random.default_rng(seed)
    model, device = _get_model()

    print("=" * 66)
    print(f"  MESSAGE: '{message}'")
    print(f"  channel: {interference}"
          + (f" @ SJR {sjr_db:+.0f} dB" if interference != "none" else "")
          + f", SNR {snr_db:.0f} dB")
    print("=" * 66)

    # encode + scramble + split into frames
    info = fec.string_to_bits(message, fec.CODE_K)
    codeword = fec.ldpc_encode(info)
    tx = fec.scramble(codeword)
    n_frames = int(np.ceil(len(tx) / PAYLOAD_BITS))
    padded = np.concatenate([tx, np.zeros(n_frames * PAYLOAD_BITS - len(tx),
                                          dtype=np.uint8)])
    frames = padded.reshape(n_frames, PAYLOAD_BITS)

    # transmit all frames; keep the FIRST frame's signals for the picture
    llr_stream = np.zeros(n_frames * PAYLOAD_BITS, dtype=np.float32)
    raw_errs = []
    cap0 = None
    for fi in range(n_frames):
        llr, pred, cap = _tx_rx_capture(frames[fi], interference, sjr_db,
                                        snr_db, rng, model, device, T)
        llr_stream[fi * PAYLOAD_BITS:(fi + 1) * PAYLOAD_BITS] = llr
        raw_errs.append(int((pred != frames[fi]).sum()))
        if fi == 0:
            cap0 = cap

    # descramble LLR signs + decode
    scr = fec.scramble(np.zeros(len(llr_stream), dtype=np.uint8))
    llr_descr = np.where(scr[:len(llr_stream)] == 1,
                         -llr_stream, llr_stream)[:fec.CODE_N]
    dec_info = fec.ldpc_decode(llr_descr, device)
    recovered = fec.bits_to_string(dec_info[:len(message) * 8])[:len(message)]

    total_raw = sum(raw_errs)
    pre_ber = total_raw / (n_frames * PAYLOAD_BITS)
    ok = recovered == message
    print(f"  spans {n_frames} frames | raw BER {pre_ber:.4f} "
          f"({total_raw} errors) | sync err {cap0['sync_err']:+d} samp")
    print(f"\n  RECOVERED: '{recovered}'")
    print("  " + ("✓ PERFECT after FEC" if ok else "✗ exceeded FEC budget"))

    _viz_journey(message, recovered, cap0, raw_errs, pre_ber, ok, total_raw,
                 interference, sjr_db, snr_db, fig_path)
    return ok


# ────────────────────────────────────────────────────────────
# the signal-journey figure
# ────────────────────────────────────────────────────────────
def _psd(x):
    X = np.fft.fftshift(np.fft.fft(x))
    f = np.fft.fftshift(np.fft.fftfreq(len(x), 1 / FS)) / 1e6
    return f, 20 * np.log10(np.abs(X) + 1e-9)


def _viz_journey(orig, recv, cap, raw_errs, pre_ber, ok, total_raw,
                 mode, sjr_db, snr_db, fig_path):
    safe = lambda s: "".join(c if 32 <= ord(c) < 127 else "·" for c in s)
    n_show = 400
    t_us = np.arange(n_show) / FS * 1e6

    fig = plt.figure(figsize=(16, 13))
    gs = gridspec.GridSpec(5, 2, height_ratios=[1, 1, 1, 1, 0.9],
                           hspace=0.5, wspace=0.2)

    stages = [
        ("1. Transmitted PCM-FM (clean)", cap["clean"], "#1f77b4"),
        ("2. Interference + noise hitting the signal", cap["interf"], "#d62728"),
        ("3. Received signal (corrupted)", cap["received"], "#7f4fa0"),
        ("4. Model-reconstructed signal", cap["recon"], "#2ca02c"),
    ]
    # left column: time domain
    for i, (title, x, col) in enumerate(stages):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(t_us, x.real[:n_show], color=col, lw=0.9)
        ax.plot(t_us, x.imag[:n_show], color=col, lw=0.9, alpha=0.45)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylabel("amp"); ax.grid(alpha=.25)
        if i == 3:
            ax.set_xlabel("Time (µs)")
    # right column: frequency domain (same four stages)
    for i, (title, x, col) in enumerate(stages):
        ax = fig.add_subplot(gs[i, 1])
        if np.any(x):
            f, P = _psd(x)
            ax.plot(f, P, color=col, lw=0.7)
        ax.set_title(title.replace("(clean)", "(spectrum)"), fontsize=10,
                     loc="left")
        ax.set_ylabel("dB"); ax.set_xlim(-5, 5); ax.grid(alpha=.25)
        if i == 3:
            ax.set_xlabel("Frequency (MHz)")

    # bottom: the text result panel spanning both columns (dedicated row)
    axr = fig.add_subplot(gs[4, :]); axr.axis("off")
    color = "#cdeb8b" if ok else "#f4a6a6"
    msg = (f"CHANNEL:   {mode} @ SJR {sjr_db:+.0f} dB,  SNR {snr_db:.0f} dB\n\n"
           f"YOU SENT:    \"{safe(orig)}\"\n"
           f"RECOVERED:   \"{safe(recv)}\"\n\n"
           f"Raw errors before FEC: {total_raw}  (BER {pre_ber:.4f})      "
           f"After LDPC: {'ALL CORRECTED — perfect recovery ✓' if ok else 'errors remain ✗'}")
    axr.text(0.01, 0.5, msg, transform=axr.transAxes, fontsize=13,
             family="monospace", va="center",
             bbox=dict(boxstyle="round,pad=0.7", facecolor=color,
                       edgecolor="black", lw=1.5))

    fig.suptitle("Signal journey:  text → PCM-FM → interference → "
                 "receiver (sync + v8.5 model) → LDPC → text",
                 fontsize=14, y=0.995)
    fig.savefig(fig_path, dpi=125, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure -> {fig_path}")


def demo_interactive(interference="wideband", sjr_db=0.0, snr_db=15.0):
    msg = input("Enter telemetry message to transmit: ").strip()
    if not msg:
        msg = "ALTITUDE 35000 SPEED 480 HEADING 270"
        print(f"  (empty — using default: '{msg}')")
    demo(msg, interference=interference, sjr_db=sjr_db, snr_db=snr_db)


if __name__ == "__main__":
    for msg, mode, sjr in [
        ("ALTITUDE 35000 SPEED 480 HEADING 270", "wideband", 0),
        ("MAYDAY ENGINE 2 FLAMEOUT FUEL 12 PCT", "tones_cont", 0),
    ]:
        print("\n" + "▼" * 66)
        demo(msg, interference=mode, sjr_db=sjr,
             fig_path=f"journey_{mode}.png")
