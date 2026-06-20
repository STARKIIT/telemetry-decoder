# Server — Telemetry Decoder Demo API

FastAPI wrapper around the existing Python pipeline. It does **not**
reimplement any signal processing, model, or FEC logic — it imports
`demo_journey.py` / `demo_v8_5_fec.py` from the `pipeline/` folder and calls
their functions.

## Setup

```bash
pip install -r requirements.txt
```

Place the pipeline files and the model checkpoint together, e.g.:

```
pipeline/
  sync_frontend_v1.py
  integration_test_v1.py
  ood_slice_generator_v17.py
  llr_ldpc_harness_v8_1.py
  demo_v8_5_fec.py
  demo_journey.py
  telemetry_v8_5.pth
```

## Run

Point the server at that folder via `PIPELINE_DIR` (recommended):

```bash
PIPELINE_DIR=/abs/path/to/pipeline \
  uvicorn app:app --host 0.0.0.0 --port 8000
```

The model loads once at startup (with a warm-up run). First request after
startup is fast.

## Config (env vars)

- `PIPELINE_DIR` — folder with the pipeline `.py` files and `telemetry_v8_5.pth`.
  If unset, the server looks for `../pipeline` then the current directory.
- `LLR_TEMPERATURE` — LLR calibration temperature (default 0.55). Set to the
  value fitted for your checkpoint if different.

## Endpoints

### `GET /api/health`
Returns `{status, device, pipeline_dir, code:{n,k,payload_bits}, temperature}`.

### `POST /api/run`
Request:
```json
{ "message": "ALTITUDE 35000 SPEED 480",
  "interference": "wideband",   // none | v16_tones | wideband | tones_cont
  "sjr_db": 0, "snr_db": 15, "seed": 42 }
```
Returns the four-stage signal journey (time + spectrum), per-frame bit grids,
and both `recovered_pre_fec` and `recovered_post_fec` with a stats block.
See ARCHITECTURE.md §4 for the full response schema.

## Notes
- All results are from a **simulated** RF channel; the response includes
  `stats.simulated = true`. UI copy should say so.
- CORS is open (`*`) for local development. Restrict origins before any real
  deployment.
- The pipeline modules resolve each other by filename relative to the working
  directory; the server `chdir`s into `PIPELINE_DIR` at import so this works.
  Do not edit the pipeline files to "fix" paths — use `PIPELINE_DIR`.
