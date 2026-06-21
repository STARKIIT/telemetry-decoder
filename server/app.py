# ============================================================
# server/app.py — Telemetry Decoder API
# ============================================================
import os
import sys
import importlib.util
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ────────────────────────────────────────────────────────────
# Locate and import the pipeline modules.
# PIPELINE_DIR can override; otherwise assume ../pipeline relative to this file,
# falling back to the current working directory.
# ────────────────────────────────────────────────────────────
def _pipeline_dir():
    env = os.environ.get("PIPELINE_DIR")
    if env and os.path.isdir(env):
        return os.path.abspath(env)
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.abspath(os.path.join(here, "..", "pipeline"))
    if os.path.isdir(cand):
        return cand
    return os.getcwd()


PIPELINE = _pipeline_dir()
if PIPELINE not in sys.path:
    sys.path.insert(0, PIPELINE)

# Make cwd the pipeline dir so candidate path lookups in _load() succeed.
os.chdir(PIPELINE)


def _import(name):
    path = os.path.join(PIPELINE, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Pipeline file '{name}' not found in {PIPELINE}. "
            f"Set PIPELINE_DIR to the folder containing the .py files and "
            f"telemetry_v8_5.pth.")
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name[:-3]] = mod
    spec.loader.exec_module(mod)
    return mod


# Importing demo_journey transitively loads the whole pipeline
dj = _import("demo_journey.py")
fec = dj.fec                       # demo_v8_5_fec, already loaded by demo_journey
FS = dj.FS
PAYLOAD_BITS = fec.PAYLOAD_BITS
CODE_N, CODE_K = fec.CODE_N, fec.CODE_K

DEFAULT_T = float(os.environ.get("LLR_TEMPERATURE", fec.TEMPERATURE))

INTERFERENCE_MODES = {"none", "v16_tones", "wideband", "tones_cont"}
STAGE_NAMES = {
    "clean": "Transmitted PCM-FM (clean)",
    "interf": "Interference + noise",
    "received": "Received signal (corrupted)",
    "recon": "Recovered signal (regenerated from decoded bits)",
}
N_TIME_SAMPLES = 400        # samples of waveform returned per stage
N_PSD_POINTS = 256          # downsampled spectrum points per stage
N_BIT_FRAMES = 4            # frames whose bit grids are returned
N_BIT_SHOW = 96             # bits per frame in the grid

# ────────────────────────────────────────────────────────────
# App + model warmup
# ────────────────────────────────────────────────────────────
app = FastAPI(title="PCM-FM Telemetry Decoder Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # dev; tighten for any real deployment
    allow_methods=["*"], allow_headers=["*"],
)

_STATE = {"model": None, "device": None}


@app.on_event("startup")
def _warmup():
    model, device = dj._get_model()
    _STATE["model"], _STATE["device"] = model, device
    # one dummy run so the first real request isn't slow
    try:
        rng = np.random.default_rng(0)
        payload = np.zeros(PAYLOAD_BITS, dtype=np.uint8)
        dj._tx_rx_capture(payload, "none", 0.0, 20.0, rng, model, device,
                          DEFAULT_T)
        print("[warmup] completed successfully.")
    except Exception as e:        # warmup is best-effort
        print(f"[warmup] skipped: {e}")
    print(f"[startup] model ready on {device}; pipeline at {PIPELINE}")


# ────────────────────────────────────────────────────────────
# Request / response models
# ────────────────────────────────────────────────────────────
class RunRequest(BaseModel):
    message: str = Field(..., max_length=128)
    interference: str = "wideband"
    sjr_db: float = 0.0
    snr_db: float = 15.0
    seed: int | None = None


# ────────────────────────────────────────────────────────────
# Helpers: signal -> transportable JSON
# ────────────────────────────────────────────────────────────
def _time_payload(x):
    n = min(N_TIME_SAMPLES, len(x))
    seg = x[:n]
    t_us = (np.arange(n) / FS * 1e6)
    return {
        "i": np.real(seg).astype(float).round(6).tolist(),
        "q": np.imag(seg).astype(float).round(6).tolist(),
        "t_us": t_us.round(4).tolist(),
    }


def _spectrum_payload(x):
    if not np.any(x):
        # all-zero stage (e.g. interference=none) -> flat floor
        freqs = (np.fft.fftshift(np.fft.fftfreq(len(x), 1 / FS)) / 1e6)
        idx = np.linspace(0, len(freqs) - 1, N_PSD_POINTS).astype(int)
        return {"freq_mhz": freqs[idx].round(4).tolist(),
                "psd_db": [-90.0] * len(idx)}
    X = np.fft.fftshift(np.fft.fft(x))
    psd = 20 * np.log10(np.abs(X) + 1e-9)
    freqs = np.fft.fftshift(np.fft.fftfreq(len(x), 1 / FS)) / 1e6
    idx = np.linspace(0, len(freqs) - 1, N_PSD_POINTS).astype(int)
    return {"freq_mhz": freqs[idx].round(4).tolist(),
            "psd_db": psd[idx].round(3).tolist()}


def _stage(stage_id, key, sig):
    return {"id": stage_id, "name": STAGE_NAMES[key],
            "time": _time_payload(sig), "spectrum": _spectrum_payload(sig)}


def _bits_str(arr, n=N_BIT_SHOW):
    return "".join(str(int(b)) for b in arr[:n])


# ────────────────────────────────────────────────────────────
# The run endpoint
# ────────────────────────────────────────────────────────────
@app.post("/api/run")
def run_simulation(req: RunRequest):
    # ---- validation ----
    msg = req.message
    if not msg:
        return {"ok": False, "error": "Message is empty."}
    if req.interference not in INTERFERENCE_MODES:
        raise HTTPException(422, f"interference must be one of "
                                 f"{sorted(INTERFERENCE_MODES)}")
    sjr = float(np.clip(req.sjr_db, -5.0, 20.0))
    snr = float(np.clip(req.snr_db, 5.0, 30.0))
    seed = req.seed if req.seed is not None else int(np.random.randint(1, 1 << 31))
    rng = np.random.default_rng(seed)
    model, device = _STATE["model"], _STATE["device"]
    if model is None:
        model, device = dj._get_model()
        _STATE["model"], _STATE["device"] = model, device

    try:
        # ---- encode: text -> info -> LDPC codeword -> scramble ----
        info = fec.string_to_bits(msg, CODE_K)
        codeword = fec.ldpc_encode(info)
        tx = fec.scramble(codeword)

        n_frames = int(np.ceil(len(tx) / PAYLOAD_BITS))
        padded = np.concatenate(
            [tx, np.zeros(n_frames * PAYLOAD_BITS - len(tx), dtype=np.uint8)])
        frames = padded.reshape(n_frames, PAYLOAD_BITS)

        # ---- transmit each frame; keep stage signals from the first ----
        llr_stream = np.zeros(n_frames * PAYLOAD_BITS, dtype=np.float32)
        raw_errs, psls, sync_errs = [], [], []
        first_cap = None
        frame_bits = []
        pred_concat = np.zeros(n_frames * PAYLOAD_BITS, dtype=np.uint8)

        for fi in range(n_frames):
            llr, pred, cap = dj._tx_rx_capture(
                frames[fi], req.interference, sjr, snr, rng, model, device,
                DEFAULT_T)
            llr_stream[fi * PAYLOAD_BITS:(fi + 1) * PAYLOAD_BITS] = llr
            pred_concat[fi * PAYLOAD_BITS:(fi + 1) * PAYLOAD_BITS] = pred
            e = int((pred != frames[fi]).sum())
            raw_errs.append(e); psls.append(cap["psl"])
            sync_errs.append(cap["sync_err"])
            if fi == 0:
                first_cap = cap
            if fi < N_BIT_FRAMES:
                frame_bits.append({
                    "index": fi + 1, "raw_errors": e,
                    "sync_err": cap["sync_err"], "psl": round(cap["psl"], 3),
                    "sent_bits": _bits_str(frames[fi]),
                    "pred_bits": _bits_str(pred),
                })

        # ---- pre-FEC text: descramble the raw model bits and decode ASCII ----
        pre_descr = fec.descramble(pred_concat)[:CODE_K]
        recovered_pre = fec.bits_to_string(pre_descr[:len(msg) * 8])[:len(msg)]

        # ---- post-FEC: descramble LLR signs, LDPC decode ----
        scr = fec.scramble(np.zeros(len(llr_stream), dtype=np.uint8))
        llr_descr = np.where(scr[:len(llr_stream)] == 1,
                             -llr_stream, llr_stream)[:CODE_N]
        dec_info = fec.ldpc_decode(llr_descr, device)
        recovered_post = fec.bits_to_string(dec_info[:len(msg) * 8])[:len(msg)]

        total_raw = int(sum(raw_errs))
        ok = recovered_post == msg

        stages = [
            _stage(1, "clean", first_cap["clean"]),
            _stage(2, "interf", first_cap["interf"]),
            _stage(3, "received", first_cap["received"]),
            _stage(4, "recon", first_cap["recon"]),
        ]

        return {
            "ok": ok,
            "message_sent": msg,
            "recovered_pre_fec": recovered_pre,
            "recovered_post_fec": recovered_post,
            "stats": {
                "n_frames": n_frames,
                "raw_ber": round(total_raw / (n_frames * PAYLOAD_BITS), 6),
                "total_raw_errors": total_raw,
                "errors_corrected_by_fec": total_raw if ok else None,
                "sync_err_samples": int(sync_errs[0]),
                "psl": round(float(psls[0]), 3),
                "interference": req.interference,
                "sjr_db": sjr, "snr_db": snr, "seed": seed,
                "simulated": True,
            },
            "stages": stages,
            "frames": frame_bits,
        }

    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


@app.get("/api/health")
def health():
    dev = str(_STATE["device"]) if _STATE["device"] is not None else "unloaded"
    return {"status": "ok", "device": dev, "pipeline_dir": PIPELINE,
            "code": {"n": CODE_N, "k": CODE_K, "payload_bits": PAYLOAD_BITS},
            "temperature": DEFAULT_T}

# Mount static files at root (placed after API routes to avoid shadowing)
from fastapi.staticfiles import StaticFiles
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend"))
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="static")
