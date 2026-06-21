# Limitations & Open Challenges

This project is a working, end-to-end-validated telemetry receiver — but it is
a research prototype, not a fielded system. This page is an honest account of
what it does *not* yet do, the bottlenecks that remain, and what the next phase
of work needs to solve. (Stating limits plainly is part of the engineering;
a result you can't bound is a result you can't trust.)

---

## 1. It is validated in simulation, not on real hardware

Every result — 93% perfect-frame recovery, zero post-correction errors — is
measured on a **simulated** RF channel built from a physical model of the
transmitter, interference, and receiver impairments. It has **not** yet been
tested on signals captured from real radio hardware.

Why this matters: real hardware introduces effects the simulation only
approximates — actual antenna behavior, real 5G/LTE traffic patterns and
scheduling, true thermal and oscillator noise. Real interference is, in effect,
a third distribution the model hasn't seen.

**Open challenge:** the next phase is sim-to-real — evaluating on replayed real
cellular recordings, then on live software-defined-radio captures, with a
held-out real test set the model never trains on. The two biggest simulation
gaps (interference realism and a training-data shortcut) were already found and
fixed, so this phase begins on solid ground — but real-world numbers are still
to be earned.

---

## 2. Throughput is below real-time on current test hardware

The receiver runs at about **0.75× real-time** on an NVIDIA L4 GPU — it
processes data slightly slower than it arrives. A deployed range receiver
needs to keep up with a continuous stream (one 256-bit frame every 256
microseconds, ~3,900 frames/second).

The bottleneck is partly the model's size (6 million parameters) and partly one
specific component — the spectral-filtering front-end uses an FFT operation that
today's GPU compilers won't optimize, so it runs un-accelerated.

**Three known paths to close this gap (next phase):**
- **Better deployment hardware** (A100/H100-class) plus proper GPU optimization
  (TensorRT), with the FFT component split out and handled separately. This
  alone likely clears real-time.
- **A hybrid receiver** — use a cheap classical decoder for clean signals and
  only engage the neural network when interference is detected, cutting the
  per-frame workload dramatically.
- **A distilled, smaller model** — a lighter network trained to imitate the
  full one, trading a little accuracy (which error-correction absorbs) for
  ~3× speed.

The right choice depends on the final deployment hardware, which is why this is
deferred rather than guessed at now.

---

## 3. One interference type still operates in a degraded mode

The model handles realistic wideband 5G/LTE interference very well. One
specific case is harder: **continuous narrowband tone jamming** sitting right on
the carrier at equal power (think a continuous-wave jammer or a strong spur).
In that case the raw model leaves more errors (~2.5% bit error rate) than it
does elsewhere.

This is recoverable — error-correction coding cleans it up, and the message
still comes through — but the safety margin is smaller, and at even stronger
jamming it would eventually exceed what the coding can fix.

**Why it's hard (and partly fundamental):** at that point the interference
blankets the entire signal band with no spectral gap to exploit, so there is
genuinely less information to separate. **Open challenge:** if real-world
testing shows this threat class matters, the planned fix is a larger model that
looks at neighboring frames for context, recovering information that a
single-frame view loses.

---

## 4. It depends on a clean input "contract" the front-end must guarantee

The model assumes its input is well-conditioned: aligned to within a few
samples, at a specific power level (set by the receiver's automatic gain
control), with carrier-frequency offset already mostly removed. A classical
sync front-end provides all of this and was validated to meet the requirement.

But this is a real operating constraint, not a free lunch: if a deployment
violates the input contract — wrong gain, poor sync — accuracy degrades. These
requirements are now documented, but they are assumptions a fielded system must
keep satisfying.

---

## 5. The error-correction layer uses a stand-in code

Post-correction results use a proxy LDPC code that closely matches the real
telemetry standard (IRIG-106), but isn't the exact standardized code. The real
codes typically perform slightly *better*, so the conclusions hold — but
swapping in the exact standardized code is a validation step still to be done.

---

## Honest summary

| Capability | Status |
|---|---|
| Recovers telemetry under realistic 5G/LTE interference | ✅ demonstrated (simulation) |
| Zero errors after correction, across tested range | ✅ demonstrated (simulation) |
| Validated end-to-end (sync → model → error correction) | ✅ |
| Robust to timing / gain / frequency offset | ✅ within documented limits |
| Runs at real-time speed | ⚠️ ~0.75× on L4; clear paths to close |
| Continuous narrowband jamming at equal power | ⚠️ degraded but correctable |
| Tested on real radio hardware | ❌ next phase |
| Exact standardized error-correction code | ❌ proxy used; swap pending |

**Bottom line:** a complete, validated receiver in simulation, with a clear and
honest list of what real-world deployment still requires — real-hardware
testing, real-time throughput, and a couple of validation steps. The hard
science (recovering bits when interference outpowers the signal) is done; the
remaining work is engineering and real-world verification.
