# About This Project — copy blocks

Ready-to-use copy in plain language. Pick the version that fits the space.
All of it is written to be understood by a non-engineer while staying honest
for a technical reader.

---

## A. One-liner (hero subtitle / repo tagline)

> A deep-learning radio receiver that recovers aircraft telemetry data even
> when 5G/LTE interference is stronger than the signal itself — something
> traditional receivers can't do.

---

## B. Short blurb (landing-page "About" card, ~60 words)

> Aircraft and missiles beam back telemetry — altitude, speed, position — over
> radio. Those bands now overlap with 5G and LTE, and a nearby tower can drown
> the signal out entirely. This project is a neural-network receiver that
> learns the *structure* of cellular interference and separates it from the
> telemetry, recovering the original message error-free. Demonstrated in
> simulation.

---

## C. Medium blurb (README intro, ~150 words)

> ### The problem
> Aircraft and missiles send telemetry — altitude, speed, position, sensor
> data — back to the ground as radio signals. The frequency bands they use now
> overlap with 5G and LTE networks. When a cell tower transmits over the same
> band, the interference can be *stronger than the telemetry signal itself*,
> and the data is lost.
>
> ### Why traditional receivers fail
> Classical receivers were built to pull a signal out of random background
> static. They assume interference is weak and noise-like. But 5G/LTE isn't
> random noise — it's a strong, structured signal sitting right on top of
> yours. A traditional receiver can't tell the two apart, so it fails.
>
> ### What this does
> This is a deep-learning receiver that *learns* what cellular interference
> looks like and separates it from the telemetry signal — like picking out one
> voice in a loud room instead of just turning the volume down. A neural
> network reconstructs the clean signal, and error-correction coding fixes
> whatever remains. The result: the original message is recovered exactly, even
> when the interference outpowers the signal. (Results shown are on a simulated
> RF channel; real-hardware testing is the next phase.)

---

## D. "How the demo works" (explainer for the app, walks through the 4 stages)

> **Watch a message survive the channel.**
> Type a telemetry message and choose how hostile the radio environment is.
> You'll see it travel through four stages:
>
> 1. **The original signal** — your message, encoded as a clean radio waveform.
> 2. **The interference** — the 5G/LTE signal that's about to corrupt it. Notice
>    in the frequency view how it sits right on top of the telemetry band.
> 3. **The received signal** — what the receiver actually picks up: your signal
>    now buried under interference.
> 4. **The recovered signal** — what the neural network pulls back out. Compare
>    its frequency shape to stage 1.
>
> Finally, the decoder shows two results side by side: the **raw model output**
> (often garbled — the model alone doesn't get every bit) and the output **after
> error correction** (the original message, recovered exactly). The contrast is
> the point: the neural network does the heavy lifting, and error-correction
> coding cleans up the rest.

---

## E. "Why it's novel" (one paragraph, for a portfolio/about section)

> There's no published method that recovers telemetry bits when co-channel
> cellular interference is *stronger* than the signal. This receiver does — and
> it's validated end to end: a classical sync front-end finds the frame, a
> neural network excises the interference, and an error-correction layer closes
> the remaining errors, producing zero post-correction errors across the tested
> range. A key finding along the way was diagnosing that the model had quietly
> learned a shortcut from an artifact in the training data, and fixing it by
> making the simulated interference more realistic — the kind of debugging that
> separates a model that scores well on paper from one you can actually deploy.

---

## Honesty checklist (keep these true on the live site)

- Say **"simulated"** somewhere visible. Results are on a synthetic RF channel;
  real-hardware validation is a future phase. (The API already returns
  `stats.simulated = true`.)
- Avoid **"real-time"** unprompted — measured throughput is ~0.75× real-time on
  an L4 GPU. "Near-real-time" or no timing claim is safer.
- Don't claim **"state of the art"** — the honest, stronger framing is "first
  demonstrated method for this problem" (no published benchmark exists).
- The pre-FEC vs post-FEC contrast is **real**, not staged: the raw model leaves
  errors, and the LDPC decoder corrects them. Label the raw side honestly
  ("before error correction") rather than hiding it.

---

## Suggested stage labels (plain headline + technical subheading)

Replace jargon-y titles with a two-tier label: a plain headline everyone
understands, and a small technical subheading for credibility.

| Stage | Plain headline | Technical subheading |
|---|---|---|
| 1 | The Original Signal | Transmitted PCM-FM waveform (clean) |
| 2 | The Interference | Co-channel 5G/LTE jammer |
| 3 | The Signal We Receive | Received signal, buried in interference |
| 4 | The Recovered Signal | Neural-network (Conformer) reconstruction |

Result panel labels:
- "Garbled Pre-FEC" → **Before error correction** (raw model output)
- "Decoded Post-FEC" → **After error correction** (final result)
- "PSL lock metric: 1.39" → **Sync lock: strong (1.39)**
- Show **"corrected N errors"** explicitly next to the raw error count.
