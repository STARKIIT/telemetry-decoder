# ARCHITECTURE.md — PCM-FM Telemetry Decoder: End-to-End Web Demo

> **Purpose of this document.** This is a build specification for an AI coding
> agent (Antigravity). It describes a full-stack web application that wraps an
> existing, validated Python signal-processing + deep-learning pipeline and
> exposes it through an interactive browser UI. A user types a telemetry
> message; the app transmits it through a simulated RF channel with
> interference, recovers it with a trained model + LDPC error correction, and
> visualizes the signal at every stage — before and after the model, and
> before and after FEC.
>
> **Critical constraint for the agent:** the Python pipeline already exists and
> is correct. DO NOT reimplement the DSP, the model, the sync front-end, or the
> LDPC decoder. Wrap the existing modules. Treat them as a black-box library
> and call their functions. The work is: (1) a thin API server around them,
> (2) a React front-end that calls the API and renders the returned signal
> arrays and results.

---

## 1. What the application does (user story)

1. User opens the web app and sees a single-page interface.
2. User types a short telemetry message (e.g., `ALTITUDE 35000 SPEED 480`)
   and chooses channel conditions: interference type, signal-to-jammer ratio
   (SJR), signal-to-noise ratio (SNR).
3. User clicks **Transmit & Recover**.
4. The app shows, in sequence, the signal's journey:
   - **Stage 1 — Transmitted PCM-FM** (clean signal): time-domain waveform +
     frequency spectrum.
   - **Stage 2 — Interference + noise**: the corrupting signal alone, time +
     spectrum (so the user sees what is attacking the signal and *where* in
     frequency it sits).
   - **Stage 3 — Received signal** (corrupted): time + spectrum, with the
     signal now buried in interference.
   - **Stage 4 — Model-reconstructed signal**: the deep-learning model's
     cleaned output, time + spectrum (the spectrum should visibly resemble
     Stage 1 again).
5. The app then shows the **decode result** in two parts:
   - **Before FEC**: the raw bits the model predicted, per frame, with bit
     errors highlighted, and the (garbled or partial) text this raw output
     decodes to.
   - **After FEC**: the LDPC-corrected bits and the final recovered text,
     with a clear ✓/✗ verdict and the count of errors the FEC layer fixed.
6. A summary panel states: message sent, message recovered, raw BER,
   post-FEC result, sync-lock quality.

The emotional payload of the demo: the user watches their own message get
swamped by interference, then perfectly reconstructed — and sees FEC visibly
rescue the hard cases where the model alone leaves a few errors.

---

## 2. System architecture

```
┌──────────────────────────────────────────────────────────────┐
│  BROWSER (React + Vite + TypeScript)                          │
│  - message input, channel controls (interference/SJR/SNR)     │
│  - calls POST /api/run                                        │
│  - renders signal arrays as time-domain + spectrum charts     │
│  - renders per-frame bit grids (pre-FEC) and final text       │
│  - "before FEC" vs "after FEC" toggle/columns                 │
└───────────────────────────┬──────────────────────────────────┘
                            │ JSON over HTTP (localhost)
┌───────────────────────────▼──────────────────────────────────┐
│  API SERVER (FastAPI, Python 3.11)                            │
│  - single endpoint POST /api/run                              │
│  - imports the EXISTING pipeline modules (do not rewrite)     │
│  - runs: encode → channel → sync → model → LLR → LDPC decode  │
│  - downsamples signal arrays for transport, returns JSON      │
└───────────────────────────┬──────────────────────────────────┘
                            │ direct Python imports
┌───────────────────────────▼──────────────────────────────────┐
│  EXISTING PIPELINE (provided .py files — DO NOT MODIFY logic) │
│  sync_frontend_v1.py        PCM-FM mod + sync acquisition     │
│  integration_test_v1.py     model class, interferers, extract │
│  ood_slice_generator_v17.py v16 RF chain stages (LNA etc.)    │
│  llr_ldpc_harness_v8_1.py   LDPC build + min-sum decoder      │
│  demo_v8_5_fec.py           LDPC encode (G-matrix), scramble, │
│                             ASCII, frame split, orchestration │
│  telemetry_v8_5.pth         trained model checkpoint          │
└──────────────────────────────────────────────────────────────┘
```

**Why this split.** The signal processing and ML must run in Python (PyTorch,
NumPy, SciPy). The browser cannot run them. So the Python stays server-side and
authoritative; the front-end is a pure visualization client. This also means
the demo's correctness is guaranteed by the already-validated Python — the
agent cannot accidentally break the science by touching the UI.

---

## 3. The existing pipeline (reference — what to call, not rebuild)

All files sit in one directory (the `pipeline/` folder of the repo). They are
self-contained and import each other by filename via an internal `_load()`
helper, so they must remain co-located. Key entry points the API will use:

**From `demo_v8_5_fec.py`** (the orchestration layer — prefer reusing it):
- `string_to_bits(text, nbits)` / `bits_to_string(bits)` — ASCII ↔ bits.
- `scramble(bits)` / `descramble(bits)` — LFSR whitening (self-inverse).
- `ldpc_encode(info_bits)` → length-2046 codeword (rate-1/2, k=1025).
- `ldpc_decode(llr_codeword, device)` → recovered info bits.
- `CODE_N=2046`, `CODE_K=1025`, `PAYLOAD_BITS=224`, `SYNC_LEN_BITS=32`.
- `make_interference(mode, n, rng)` — modes: `"none"`, `"v16_tones"`,
  `"wideband"`, `"tones_cont"`.
- `CHECKPOINT` — path to `telemetry_v8_5.pth`. `TEMPERATURE` — LLR scale (~0.55).

**From `demo_journey.py`** (already captures per-stage signals — REUSE THIS):
- `_tx_rx_capture(payload_bits, mode, sjr_db, snr_db, rng, model, device, T)`
  returns `(llr_payload, pred_payload, cap)` where `cap` is a dict with the
  four stage signals as complex arrays:
  `cap["clean"]`, `cap["interf"]`, `cap["received"]`, `cap["recon"]`
  (each length 2560 complex64), plus `cap["psl"]`, `cap["sync_err"]`.
  **This function is the heart of the demo — the API calls it per frame.**
- `_get_model()` → `(model, device)`, loads the checkpoint once and caches it.

**From `sync_frontend_v1.py`**: `FS=10e6` (sample rate), `SIG_LEN=2560`,
`SYNC_BITS` (the 32-bit sync word), `pcmfm_iq()`, `find_frame_start()`.

The agent should **reuse `demo_journey.demo()` and `_tx_rx_capture` as the
computational core** and only refactor enough to return data instead of
saving a matplotlib figure.

---

## 4. API server specification

**Framework:** FastAPI + uvicorn. Single file `server/app.py`. CORS enabled
for the front-end origin (localhost dev).

### Endpoint: `POST /api/run`

**Request body (JSON):**
```json
{
  "message": "ALTITUDE 35000 SPEED 480",
  "interference": "wideband",      // none | v16_tones | wideband | tones_cont
  "sjr_db": 0,                     // -5 .. 20
  "snr_db": 15,                    // 5 .. 30
  "seed": 42                       // optional; random if omitted
}
```

**Server logic (reuse the existing pipeline):**
1. Validate inputs (message length ≤ 128 chars; clamp sjr_db/snr_db to ranges;
   interference in the allowed set).
2. `model, device = demo_journey._get_model()` (cached).
3. Encode: `info = string_to_bits(message, CODE_K)`,
   `codeword = ldpc_encode(info)`, `tx = scramble(codeword)`.
4. Split `tx` into `n_frames` frames of `PAYLOAD_BITS`, zero-padded.
5. For each frame, call `_tx_rx_capture(...)`. Collect:
   - the LLR stream (for decoding),
   - per-frame raw bit errors (pred vs sent),
   - per-frame `psl` and `sync_err`,
   - **the four stage signals from the FIRST frame only** (one representative
     frame is enough for the visualization — do not return signals for every
     frame; that would be megabytes).
6. Descramble LLR signs, `ldpc_decode(...)`, `bits_to_string(...)` →
   `recovered`.
7. Compute: `raw_ber`, `total_raw_errors`, `ok = recovered == message`,
   and a **pre-FEC text** = `bits_to_string(descramble(concatenated raw frame
   predictions))[:len(message)]` so the UI can show the garbled "before FEC"
   text next to the clean "after FEC" text.

**Signal transport — IMPORTANT.** Each stage signal is 2560 complex samples.
Return BOTH representations the UI needs, downsampled for transport:
- **Time domain:** real and imaginary parts of the **first 400 samples** of
  each stage signal (enough to see the waveform), as plain float arrays.
- **Frequency domain:** compute the PSD on the **full 2560 samples** server-side
  (`20*log10(|fftshift(fft(x))| + 1e-9)`), then downsample to ~256 points with
  the frequency axis in MHz (`fftshift(fftfreq(2560, 1/FS))/1e6`). Return as
  `{freq_mhz: [...], psd_db: [...]}`. Doing the FFT server-side keeps the
  front-end trivial and guarantees it matches the validated science.

**Response body (JSON):**
```json
{
  "ok": true,
  "message_sent": "ALTITUDE 35000 SPEED 480",
  "recovered_pre_fec": "AL?I_UDE 350#0 S?EED 4??",
  "recovered_post_fec": "ALTITUDE 35000 SPEED 480",
  "stats": {
    "n_frames": 10,
    "raw_ber": 0.0021,
    "total_raw_errors": 5,
    "errors_corrected_by_fec": 5,
    "sync_err_samples": 1,
    "psl": 1.52,
    "interference": "wideband", "sjr_db": 0, "snr_db": 15
  },
  "stages": [
    {
      "id": 1, "name": "Transmitted PCM-FM (clean)",
      "time": { "i": [...400 floats...], "q": [...400 floats...], "t_us": [...] },
      "spectrum": { "freq_mhz": [...256...], "psd_db": [...256...] }
    },
    { "id": 2, "name": "Interference + noise", ... },
    { "id": 3, "name": "Received signal (corrupted)", ... },
    { "id": 4, "name": "Model-reconstructed signal", ... }
  ],
  "frames": [
    { "index": 1, "raw_errors": 0, "sync_err": 0, "psl": 1.6,
      "sent_bits": "0101...", "pred_bits": "0101..." }
    // include per-frame sent/pred bits for the FIRST 4 frames only,
    // truncated to first 96 bits each, for the bit-grid visualization
  ]
}
```

**Error handling:** on any exception, return HTTP 200 with
`{ "ok": false, "error": "<message>" }` so the UI can show a friendly message.
Validation failures return 422 with a clear detail string.

**Performance note:** on CPU a full run (≈10 frames) takes a few seconds; on
GPU under a second. The endpoint is synchronous; show a loading state in the
UI. The model loads once at server start (warm it with a dummy run on startup).

---

## 5. Front-end specification

**Stack:** React + TypeScript + Vite. Charting: **Recharts** (or uPlot if
Recharts is too slow for 400-point line charts — both are acceptable; prefer
Recharts for simplicity). No backend state; the front-end is stateless between
runs.

### Layout (single page, top to bottom)

**A. Control bar (sticky top).**
- Text input for the message (placeholder: "Type a telemetry message…",
  maxlength 128, with a live character counter).
- Dropdown: interference type — `None`, `Wideband 5G/LTE (realistic)`,
  `Narrowband tones`, `Continuous tones (hardest)` → mapping to
  `none / wideband / v16_tones / tones_cont`.
- Two sliders: **SJR** (−5 to 20 dB) and **SNR** (5 to 30 dB), with current
  value shown. Add a one-line helper: "Lower SJR = stronger interference."
- Primary button: **Transmit & Recover**. Shows a spinner while awaiting the
  API.
- A few preset buttons that fill example messages + settings, e.g.
  "Easy (clean)", "Realistic (wideband, SJR 0)", "Hard (continuous tones)".

**B. Signal journey (the centerpiece).**
Four stage cards stacked vertically, each card showing **two charts side by
side**: a time-domain line chart (I solid, Q faint) on the left and a spectrum
line chart (PSD in dB vs MHz) on the right. Stage cards in order:
1. Transmitted PCM-FM (clean) — blue.
2. Interference + noise — red. (Caption: "what's attacking the signal.")
3. Received signal (corrupted) — purple. (Caption: "your signal is now buried
   in interference.")
4. Model-reconstructed signal — green. (Caption: "the model pulls the signal
   back out — compare this spectrum to Stage 1.")

Use a small connecting arrow / "↓" between cards to convey flow. Animate the
cards in sequentially on result arrival (a ~150 ms stagger) so the user
perceives a journey, not a static dump.

**C. Decode result (two columns: BEFORE FEC vs AFTER FEC).**
- **Left column — "Model output (before FEC)":** show the first up-to-4 frames
  as small bit grids — each frame a row of 96 cells, sent bits as the baseline,
  predicted bits overlaid, mismatched bits highlighted red. Below the grids,
  show `recovered_pre_fec` in a monospace box (it will look garbled/partial on
  hard channels — that's the point). Show the raw BER and total raw error
  count.
- **Right column — "After LDPC (FEC)":** show `recovered_post_fec` in a large
  monospace box, with a big ✓ (green) or ✗ (red) verdict, and the line
  "FEC corrected N errors." If `ok`, style the box green; if not, amber with
  the note "raw errors exceeded the FEC budget for this extreme channel."

**D. Summary strip (bottom).**
Compact stats: frames, sync lock (`psl` with a LOCK/WEAK badge, threshold 1.15),
sync error in samples, channel settings echoed back.

### UX details
- The "before vs after FEC" contrast is the climax — make it visually obvious
  that the left side is messy and the right side is clean. On easy channels
  both may be clean; that's fine.
- Keep a results-history list optional (nice-to-have): each run as a collapsed
  row the user can re-expand. Not required for v1.
- Mobile: stack the side-by-side charts vertically below ~700 px width.
- Provide a "Download this figure" button that renders the current signal
  journey to a PNG (client-side, e.g. via `html-to-image`) — useful for the
  user's portfolio/screenshots.

---

## 6. Repository structure to generate

```
telemetry-demo/
├── ARCHITECTURE.md                 (this file)
├── README.md                       (generate: how to run, screenshots)
├── pipeline/                       (the EXISTING .py files — copied in as-is)
│   ├── sync_frontend_v1.py
│   ├── integration_test_v1.py
│   ├── ood_slice_generator_v17.py
│   ├── llr_ldpc_harness_v8_1.py
│   ├── demo_v8_5_fec.py
│   ├── demo_journey.py
│   └── telemetry_v8_5.pth          (model checkpoint; large binary)
├── server/
│   ├── app.py                      (FastAPI; imports from ../pipeline)
│   ├── requirements.txt            (fastapi, uvicorn, torch, numpy, scipy)
│   └── README.md
├── web/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── main.tsx
│       ├── App.tsx                 (layout + state + fetch)
│       ├── api.ts                  (typed client for POST /api/run)
│       ├── types.ts                (RunRequest, RunResponse, Stage, Frame)
│       └── components/
│           ├── ControlBar.tsx
│           ├── SignalStageCard.tsx (time + spectrum charts for one stage)
│           ├── DecodeResult.tsx    (before/after FEC columns)
│           └── SummaryStrip.tsx
├── docker-compose.yml              (optional: server + web together)
└── .gitignore
```

**The pipeline `_load()` helper** searches the current directory and a couple
of `/content/...` paths. For the server, set the working directory to
`pipeline/` (or adjust `CHECKPOINT` and the loader search paths via an env var)
so the modules find each other and the checkpoint. Document this in
`server/README.md`. Do not edit the pipeline files' logic; if a path needs
changing, do it via environment variable or a tiny wrapper, not by rewriting.

---

## 7. Constraints, correctness, and non-goals

**Must:**
- Keep all DSP/ML/FEC computation in the Python pipeline, called as-is.
- Compute FFTs/PSDs server-side; the front-end only plots arrays it receives.
- Make the before-FEC vs after-FEC contrast the visual climax.
- Handle the four interference modes and the SJR/SNR ranges given.
- Load the model once and reuse it across requests.

**Must not:**
- Reimplement PCM-FM modulation, the channel, the model, sync, LLRs, or LDPC
  in JavaScript or in new Python. There is exactly one source of truth.
- Send all 2560 samples of every stage for every frame (transport bloat).
  Send first-frame stage signals only, downsampled as specified.
- Claim real-world performance in the UI copy. This is a **simulation demo**;
  label it "simulated RF channel" somewhere visible.

**Non-goals (explicitly out of scope for v1):**
- User accounts, persistence, databases.
- Real SDR hardware input (that's a future project phase).
- Training or fine-tuning from the UI.
- The model's auxiliary outputs (SNR/SJR estimates) — ignore them for now.

**Accuracy / honesty note for UI copy:** describe results as obtained on a
simulated channel. The headline capability is: recovering a typed message from
a signal corrupted by co-channel 5G/LTE-class interference at negative SJR,
with FEC closing the residual errors. Post-FEC recovery is exact within the
characterized envelope; the `tones_cont` mode at SJR 0 is the documented hard
corner where the model alone leaves errors that FEC then corrects — a great
thing to demonstrate, not hide.

---

## 8. Build order for the agent

1. **Scaffold** the repo structure (Section 6). Copy the provided pipeline
   files into `pipeline/` untouched.
2. **Server first.** Write `server/app.py` with the `/api/run` endpoint
   (Section 4). Refactor only by *wrapping* `demo_journey._tx_rx_capture` /
   `demo_v8_5_fec` functions to return JSON-able data instead of figures.
   Add a startup model warm-up. Verify with `curl` that a request returns the
   documented JSON, including four stage objects and the pre/post-FEC texts.
3. **Front-end skeleton.** Vite + React + TS, the typed API client, and the
   control bar wired to `POST /api/run` with a loading state.
4. **Signal stage cards** (time + spectrum charts) rendering the returned
   arrays. Get the four-stage journey looking right with one example.
5. **Decode result** before/after-FEC columns and the bit grids.
6. **Summary strip**, presets, sequential card animation, PNG download.
7. **README** with run instructions and screenshots.

Deliver a working `docker-compose up` (or two terminals: `uvicorn` + `vite`)
that lets a user open `localhost`, type a message, and watch it survive the
channel.