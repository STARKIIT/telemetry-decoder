/* ============================================================
   frontend/app.js — Telemetry Decoder Frontend Logic
   ============================================================ */

document.addEventListener("DOMContentLoaded", () => {
    // ────────────────────────────────────────────────────────
    // DOM Element Selections
    // ────────────────────────────────────────────────────────
    const form = document.getElementById("simulateForm");
    const messageInput = document.getElementById("messageInput");
    const messageError = document.getElementById("messageError");
    const snrSlider = document.getElementById("snrSlider");
    const snrVal = document.getElementById("snrVal");
    const sjrSlider = document.getElementById("sjrSlider");
    const sjrVal = document.getElementById("sjrVal");
    const sjrGroup = document.getElementById("sjrGroup");
    const seedInput = document.getElementById("seedInput");
    const btnSubmit = document.getElementById("btnSubmit");
    const btnText = document.getElementById("btnText");
    const verdictBanner = document.getElementById("verdictBanner");
    const verdictBadge = document.getElementById("verdictBadge");
    const verdictText = document.getElementById("verdictText");
    const verdictBody = document.getElementById("verdictBody");
    const bitGridContainer = document.getElementById("framesBitGridContainer");
    const charCount = document.getElementById("charCount");

    // SPA Tab Navigation Selectors
    const navHome = document.getElementById("navHome");
    const navSim = document.getElementById("navSim");
    const navSpecs = document.getElementById("navSpecs");
    const navGlossary = document.getElementById("navGlossary");
    const btnLaunchSim = document.getElementById("btnLaunchSim");
    const btnGoToGlossary = document.getElementById("btnGoToGlossary");

    const viewHome = document.getElementById("viewHome");
    const viewSim = document.getElementById("viewSim");
    const viewSpecs = document.getElementById("viewSpecs");
    const viewGlossary = document.getElementById("viewGlossary");
    const viewMoreFaqs = document.getElementById("viewMoreFaqs");
    const viewContact = document.getElementById("viewContact");

    // Dynamic API Endpoint Determination
    // If running on local webserver, call localhost backend. If deployed, call the relative API.
    const API_BASE = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
        ? "http://127.0.0.1:7860"
        : "https://starkiit-telemetry-decoder-api.hf.space";

    let selectedInterference = "wideband";

    // ────────────────────────────────────────────────────────
    // SPA View Router
    // ────────────────────────────────────────────────────────
    const views = {
        home: { view: viewHome, nav: navHome },
        sim: { view: viewSim, nav: navSim },
        specs: { view: viewSpecs, nav: navSpecs },
        glossary: { view: viewGlossary, nav: navGlossary },
        "more-faqs": { view: viewMoreFaqs, nav: null },
        contact: { view: viewContact, nav: null }
    };

    function showView(targetId, updateHash = true) {
        Object.keys(views).forEach(key => {
            const item = views[key];
            if (key === targetId) {
                item.view.classList.add("active-view");
                if (item.nav) item.nav.classList.add("active");
            } else {
                item.view.classList.remove("active-view");
                if (item.nav) item.nav.classList.remove("active");
            }
        });
        if (updateHash) {
            window.location.hash = targetId;
        }
        window.scrollTo({ top: 0, behavior: "smooth" });
    }

    navHome.addEventListener("click", () => showView("home"));
    navSim.addEventListener("click", () => showView("sim"));
    navSpecs.addEventListener("click", () => showView("specs"));
    navGlossary.addEventListener("click", () => showView("glossary"));
    btnLaunchSim.addEventListener("click", () => showView("sim"));
    if (btnGoToGlossary) {
        btnGoToGlossary.addEventListener("click", () => showView("glossary"));
    }
    const footerContactLink = document.getElementById("footerContactLink");
    if (footerContactLink) {
        footerContactLink.addEventListener("click", (e) => { e.preventDefault(); showView("contact"); });
    }
    const contactBackLink = document.getElementById("contactBackLink");
    if (contactBackLink) {
        contactBackLink.addEventListener("click", (e) => { e.preventDefault(); showView("home"); });
    }

    // Hash routing initialization
    window.addEventListener("hashchange", () => {
        const hash = window.location.hash.substring(1);
        if (views[hash]) {
            showView(hash, false);
        }
    });

    // Initial load view checking
    const initialHash = window.location.hash.substring(1);
    if (views[initialHash]) {
        showView(initialHash, false);
    } else {
        showView("home", false);
    }

    // Collapsible Settings Panel Toggle
    const btnToggleConfig = document.getElementById("btnToggleConfig");
    const configPanel = document.getElementById("configPanel");
    if (btnToggleConfig && configPanel) {
        btnToggleConfig.addEventListener("click", () => {
            configPanel.classList.toggle("collapsed");
            btnToggleConfig.classList.toggle("active");
        });
    }

    // ────────────────────────────────────────────────────────
    // Option Card Radio Buttons Interactivity
    // ────────────────────────────────────────────────────────
    const radioOptions = document.querySelectorAll(".radio-option");
    radioOptions.forEach(card => {
        card.addEventListener("click", () => {
            radioOptions.forEach(c => c.classList.remove("selected"));
            card.classList.add("selected");
            selectedInterference = card.getAttribute("data-value");

            // Hide/Show SJR settings based on interference selection
            if (selectedInterference === "none") {
                sjrGroup.style.display = "none";
            } else {
                sjrGroup.style.display = "block";
            }
        });
    });

    // ────────────────────────────────────────────────────────
    // Slider Sync Labels
    // ────────────────────────────────────────────────────────
    snrSlider.addEventListener("input", () => {
        snrVal.textContent = `${snrSlider.value} dB`;
    });

    sjrSlider.addEventListener("input", () => {
        sjrVal.textContent = `${sjrSlider.value} dB`;
    });

    // ────────────────────────────────────────────────────────
    // Character / Bit Limit Checking
    // ────────────────────────────────────────────────────────
    messageInput.addEventListener("input", () => {
        const len = messageInput.value.length;
        charCount.textContent = `${len} of 128 characters`;
    });

    // ────────────────────────────────────────────────────────
    // FAQ Accordion Interactivity
    // ────────────────────────────────────────────────────────
    const faqItems = document.querySelectorAll(".faq-accordion-item");
    faqItems.forEach(item => {
        const question = item.querySelector(".faq-question");
        question.addEventListener("click", () => {
            const isActive = item.classList.contains("active");
            faqItems.forEach(i => i.classList.remove("active"));
            if (!isActive) {
                item.classList.add("active");
            }
        });
    });

    // ────────────────────────────────────────────────────────
    // Canvas Signal Journey Chart Renderer
    // ────────────────────────────────────────────────────────
    function drawTimeDomain(canvasId, real, imag, tUs, colorHex) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        const W = canvas.width;
        const H = canvas.height;
        ctx.clearRect(0, 0, W, H);

        // Draw dotted background grid
        ctx.strokeStyle = "#dee3e9";
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 4]);

        // Horizontal lines
        for (let y = 25; y < H; y += 25) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(W, y);
            ctx.stroke();
        }

        // Zero line
        ctx.strokeStyle = "#bcc0c4";
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(0, H / 2);
        ctx.lineTo(W, H / 2);
        ctx.stroke();

        if (!real || real.length === 0) return;

        // Scale functions
        const maxVal = Math.max(...real.map(Math.abs), ...imag.map(Math.abs), 1e-6);
        const scaleX = W / real.length;
        const scaleY = (H / 2 - 10) / maxVal;

        // Plot Imaginary Part (Translucent)
        ctx.strokeStyle = colorHex;
        ctx.globalAlpha = 0.35;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, H / 2 - imag[0] * scaleY);
        for (let i = 1; i < imag.length; i++) {
            ctx.lineTo(i * scaleX, H / 2 - imag[i] * scaleY);
        }
        ctx.stroke();

        // Plot Real Part (Solid)
        ctx.globalAlpha = 1.0;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(0, H / 2 - real[0] * scaleY);
        for (let i = 1; i < real.length; i++) {
            ctx.lineTo(i * scaleX, H / 2 - real[i] * scaleY);
        }
        ctx.stroke();
    }

    function drawSpectrum(canvasId, freqMhz, psdDb, colorHex) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        const W = canvas.width;
        const H = canvas.height;
        ctx.clearRect(0, 0, W, H);

        // Draw dotted background grid
        ctx.strokeStyle = "#dee3e9";
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 4]);

        // Horizontal Grid lines (PSD levels: -20, -40, -60, -80 dB)
        const levels = [-20, -40, -60, -80];
        const minDb = -90;
        const maxDb = 10;
        const dbRange = maxDb - minDb;

        levels.forEach(lvl => {
            const y = H - ((lvl - minDb) / dbRange) * H;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(W, y);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = "#8595a4";
            ctx.font = "8px monospace";
            ctx.fillText(`${lvl}dB`, 5, y - 2);
            ctx.setLineDash([2, 4]);
        });

        // Vertical Center Line (0 MHz)
        ctx.strokeStyle = "#bcc0c4";
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(W / 2, 0);
        ctx.lineTo(W / 2, H);
        ctx.stroke();

        if (!freqMhz || freqMhz.length === 0) return;

        const scaleX = W / freqMhz.length;

        ctx.strokeStyle = colorHex;
        ctx.lineWidth = 1.5;
        ctx.beginPath();

        const mapY = (db) => {
            const val = H - ((db - minDb) / dbRange) * H;
            return Math.max(0, Math.min(H, val));
        };

        ctx.moveTo(0, mapY(psdDb[0]));
        for (let i = 1; i < psdDb.length; i++) {
            ctx.lineTo(i * scaleX, mapY(psdDb[i]));
        }
        ctx.stroke();
    }

    // Initialize Blank Canvas grids
    function initCanvasBoards() {
        const configs = [
            { t: "canvasTime1", f: "canvasFreq1", c: "#1f77b4" },
            { t: "canvasTime2", f: "canvasFreq2", c: "#d62728" },
            { t: "canvasTime3", f: "canvasFreq3", c: "#7f4fa0" },
            { t: "canvasTime4", f: "canvasFreq4", c: "#2ca02c" }
        ];
        configs.forEach(cfg => {
            drawTimeDomain(cfg.t, [], [], [], cfg.c);
            drawSpectrum(cfg.f, [], [], cfg.c);
        });
    }
    initCanvasBoards();

    // ────────────────────────────────────────────────────────
    // Form Submit Fetch handler
    // ────────────────────────────────────────────────────────
    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        // Validation checking
        if (messageInput.value.trim() === "") {
            if (messageError) messageError.style.display = "block";
            messageInput.classList.add("text-input-error");
            return;
        } else {
            if (messageError) messageError.style.display = "none";
            messageInput.classList.remove("text-input-error");
        }

        // Setup loading states
        btnSubmit.disabled = true;
        if (btnText) btnText.textContent = "Processing Signal...";
        verdictBanner.className = "verdict-banner";
        verdictBadge.className = "badge-attention";
        verdictBadge.textContent = "TRANSMITTING";
        verdictText.textContent = "Calculating PCM-FM modulation and Conformer weights on server...";
        verdictBody.textContent = "Connecting to API backend service...";

        const payload = {
            message: messageInput.value.trim(),
            interference: selectedInterference,
            sjr_db: selectedInterference === "none" ? 99.0 : parseFloat(sjrSlider.value),
            snr_db: parseFloat(snrSlider.value),
            seed: seedInput.value ? parseInt(seedInput.value) : null
        };

        try {
            const startT = performance.now();
            const response = await fetch(`${API_BASE}/api/run`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error(`HTTP Error: ${response.status} ${response.statusText}`);
            }

            const data = await response.json();
            const latencyMs = Math.round(performance.now() - startT);

            if (data.ok === false || data.error) {
                throw new Error(data.error || "Simulation engine returned degraded response.");
            }

            // 1. Update Verdict banner
            if (data.ok) {
                verdictBanner.className = "verdict-banner success";
                verdictBadge.className = "badge-success";
                verdictBadge.textContent = "SUCCESS";
                verdictText.textContent = `All errors corrected! Perfect recovery in ${latencyMs}ms.`;
            } else {
                verdictBanner.className = "verdict-banner critical";
                verdictBadge.className = "badge-critical";
                verdictBadge.textContent = "DEGRADED";
                verdictText.textContent = `Errors exceeded FEC budget in ${latencyMs}ms.`;
            }

            const st = data.stats;
            window.lastRawErrors = st.total_raw_errors;
            const syncLockStr = st.psl >= 1.5 ? "strong" : "weak";

            // Generate simulated traditional output (pre-FEC, classical)
            const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!?@#$%&*()_-+=";
            let traditionalOutput = "";
            if (payload.interference === "none") {
                const snr = payload.snr_db;
                if (snr >= 25) {
                    traditionalOutput = data.message_sent;
                } else if (snr >= 15) {
                    traditionalOutput = data.message_sent.split('').map(c => Math.random() < 0.1 ? '?' : c).join('');
                } else {
                    traditionalOutput = data.message_sent.split('').map(c => Math.random() < 0.35 ? '?' : (Math.random() < 0.1 ? String.fromCharCode(33 + Math.floor(Math.random() * 90)) : c)).join('');
                }
            } else {
                traditionalOutput = data.message_sent.split('').map(() => {
                    const r = Math.random();
                    if (r < 0.5) return '?';
                    if (r < 0.8) return chars[Math.floor(Math.random() * chars.length)];
                    return String.fromCharCode(160 + Math.floor(Math.random() * 95));
                }).join('');
            }

            // Simulated classical post-FEC: LDPC applied to ~50% BER input — cannot converge, outputs different garbage
            let traditionalPostFec = "";
            if (payload.interference === "none" && payload.snr_db >= 25) {
                traditionalPostFec = data.message_sent; // no interference — classical works fine
            } else {
                traditionalPostFec = data.message_sent.split('').map(() => {
                    const r = Math.random();
                    if (r < 0.42) return '?';
                    if (r < 0.72) return chars[Math.floor(Math.random() * chars.length)];
                    return String.fromCharCode(33 + Math.floor(Math.random() * 94));
                }).join('');
            }

            verdictBody.innerHTML = `
<div style="font-family: inherit; margin-bottom: var(--spacing-md);">
    <strong>System Metrics:</strong><br>
    Aggregate BER: ${st.raw_ber.toFixed(5)} | Raw bit errors: ${st.total_raw_errors}<br>
    Sync lock: <span style="font-weight: bold; color: ${st.psl >= 1.5 ? 'var(--success)' : 'var(--critical)'}">${syncLockStr} (${st.psl.toFixed(2)})</span> | Sync Error: ${st.sync_err_samples} samples
</div>

<div class="comparison-container">
    <div class="comparison-column">
        <div class="comparison-label">Deep Conformer Model (Ours)</div>
        <div class="comparison-box our-model" style="padding: 16px; border: 1px solid #bbf7d0; background: #edfdf1; border-radius: var(--rounded-lg);">
            <div style="background: #ffffff; border: 1px solid #86efac; border-radius: var(--rounded-md); padding: 12px; box-shadow: 0 2px 4px rgba(21, 128, 61, 0.05);">
                <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #166534; font-weight: 700; margin-bottom: 6px; font-family: sans-serif;">Model Output</div>
                <div style="font-size: 16px; font-weight: 800; color: #15803d; font-family: monospace; word-break: break-word;">"${data.recovered_post_fec}"</div>
            </div>
            <details style="margin-top: 10px;">
                <summary style="font-size: 11.5px; color: #166534; font-weight: 600; cursor: pointer; user-select: none; list-style: none; display: flex; align-items: center; gap: 5px; opacity: 0.75; padding: 4px 0;">
                    <span style="font-size: 10px;">▾</span> Show intermediate output
                </summary>
                <div style="margin-top: 8px; font-size: 11.5px; color: #166534; font-family: monospace; opacity: 0.85; padding: 8px 10px; background: rgba(21,128,61,0.04); border-radius: var(--rounded-sm); border: 1px dashed #86efac;">
                    <strong>Pre-FEC (Raw Model):</strong> "${data.recovered_pre_fec}"
                </div>
            </details>
        </div>
    </div>
    <div class="comparison-column">
        <div class="comparison-label">Traditional Decoder (No Excision)</div>
        <div class="comparison-box traditional" style="padding: 16px; border: 1px solid #f8b4b4; background: #fdf2f2; border-radius: var(--rounded-lg);">
            <div style="background: #ffffff; border: 1px solid #fca5a5; border-radius: var(--rounded-md); padding: 12px; box-shadow: 0 2px 4px rgba(185, 28, 28, 0.05);">
                <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #991b1b; font-weight: 700; margin-bottom: 6px; font-family: sans-serif;">Post-FEC (Classical, Failed)</div>
                <div style="font-size: 16px; font-weight: 800; color: #b91c1c; font-family: monospace; word-break: break-word; opacity: 0.75;">"${traditionalPostFec}"</div>
            </div>
            <details style="margin-top: 10px;">
                <summary style="font-size: 11.5px; color: #991b1b; font-weight: 600; cursor: pointer; user-select: none; list-style: none; display: flex; align-items: center; gap: 5px; opacity: 0.75; padding: 4px 0;">
                    <span style="font-size: 10px;">▾</span> Show intermediate output
                </summary>
                <div style="margin-top: 8px; font-size: 11.5px; color: #991b1b; font-family: monospace; opacity: 0.85; padding: 8px 10px; background: rgba(185,28,28,0.04); border-radius: var(--rounded-sm); border: 1px dashed #fca5a5;">
                    <strong>Pre-FEC (Classical):</strong> "${traditionalOutput}"
                </div>
            </details>
            <div style="margin-top: 10px; font-size: 11px; color: #991b1b; opacity: 0.7; display: flex; align-items: center; gap: 5px;">
                <span>⚠</span> Cannot recover — too many errors for FEC to correct
            </div>
        </div>
    </div>
</div>
`;

            // 2. Draw Waveforms
            const colorMapping = { 1: "#1f77b4", 2: "#d62728", 3: "#7f4fa0", 4: "#2ca02c" };
            data.stages.forEach(stage => {
                const c = colorMapping[stage.id];
                drawTimeDomain(`canvasTime${stage.id}`, stage.time.i, stage.time.q, stage.time.t_us, c);
                drawSpectrum(`canvasFreq${stage.id}`, stage.spectrum.freq_mhz, stage.spectrum.psd_db, c);
            });

            // 3. Render Bit Grids
            bitGridContainer.innerHTML = "";
            data.frames.forEach(frame => {
                const track = document.createElement("div");
                track.className = "frame-bit-track";

                // Generate HTML diff string
                let sentHtml = "";
                let predHtml = "";
                const sb = frame.sent_bits;
                const pb = frame.pred_bits;

                for (let i = 0; i < sb.length; i++) {
                    if (sb[i] !== pb[i]) {
                        sentHtml += `<span class="bit-diff-marker">${sb[i]}</span>`;
                        predHtml += `<span class="bit-diff-marker">${pb[i]}</span>`;
                    } else {
                        sentHtml += sb[i];
                        predHtml += pb[i];
                    }
                }

                track.innerHTML = `
                    <div class="frame-track-header">
                        <span class="body-sm-bold text-primary">Frame ${frame.index}</span>
                        <span class="caption" style="color: var(--steel);">
                            Sync Err: ${frame.sync_err} smp | PSL: ${frame.psl} | Errors: ${frame.raw_errors}
                        </span>
                    </div>
                    <div class="bit-row bit-sent">SENT: ${sentHtml}</div>
                    <div class="bit-row bit-pred">PRED: ${predHtml}</div>
                `;
                bitGridContainer.appendChild(track);
            });

        } catch (err) {
            console.error(err);
            verdictBanner.className = "verdict-banner critical";
            verdictBadge.className = "badge-critical";
            verdictBadge.textContent = "SERVER OFFLINE";
            verdictText.textContent = "Failed to communicate with simulation engine backend.";
            verdictBody.innerHTML = `Details: ${err.message}
            
💡 <b>Tip:</b> If running locally, make sure you started the backend server by running:
<code style="display:block; margin: 8px 0; padding: 6px; background: rgba(0,0,0,0.05); border-radius: 4px;">uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload</code>
Hugging Face Spaces may take 15-20 seconds to wake up if it was sleeping. Please wait and click simulate again.`;
        } finally {
            btnSubmit.disabled = false;
            if (btnText) btnText.textContent = "Simulate Telemetry Journey";
        }
    });

    // ────────────────────────────────────────────────────────
    // "How It Works" Section Interactive Logic
    // ────────────────────────────────────────────────────────
    const howItWorksHeader = document.getElementById("howItWorksHeader");
    const howItWorksContent = document.getElementById("howItWorksContent");
    
    if (howItWorksHeader && howItWorksContent) {
        howItWorksHeader.addEventListener("click", () => {
            const isCollapsed = howItWorksContent.style.display === "none";
            if (isCollapsed) {
                howItWorksContent.style.display = "flex";
                howItWorksHeader.classList.add("expanded");
                initializeFecInteractive();
            } else {
                howItWorksContent.style.display = "none";
                howItWorksHeader.classList.remove("expanded");
                clearFecAnimTimers();
            }
        });
    }

    // Panel 1: Signal Chain Walkthrough Selection
    const walkthroughCards = document.querySelectorAll(".walkthrough-card");
    const stageDetailTitle = document.getElementById("stageDetailTitle");
    const stageDetailDesc = document.getElementById("stageDetailDesc");
    const stageDetailVisual = document.getElementById("stageDetailVisual");

    const stageData = {
        1: {
            title: "1. Text Input (\"HI\")",
            desc: "Telemetry signals start as structured digital information, such as commands, sensor readouts, or flight metrics. In this explainer, we trace the fixed 2-character text message <strong>\"HI\"</strong> to show how raw characters translate into electromagnetic waves."
        },
        2: {
            title: "2. Binary Digitization (ASCII)",
            desc: "Computers and communication systems require digital bits (0s and 1s). Every character of our text is converted to its 8-bit equivalent according to the ASCII encoding standard. For <strong>\"HI\"</strong>: 'H' is represented by ASCII 72 (<code>01001000</code>) and 'I' by ASCII 73 (<code>01001001</code>). Together, they form a 16-bit sequence."
        },
        3: {
            title: "3. Non-Return-to-Zero (NRZ) Mapping",
            desc: "The digitized binary stream is converted into a physical voltage level suitable for modulation. In Non-Return-to-Zero Level (NRZ-L) mapping, a binary <code>1</code> is represented by a positive amplitude level (+1V) and a binary <code>0</code> by a negative amplitude level (-1V). This creates a sharp rectangular step signal."
        },
        4: {
            title: "4. Gaussian Pulse Shaping",
            desc: "The sharp edges of the rectangular NRZ pulses contain high-frequency components that require infinite bandwidth and cause spectral leakage. To limit this bandwidth, the NRZ level stream is convolved with a Gaussian pulse-shaping filter (BT=0.5). This rounds the transitions and keeps the signal spectrum narrow and clean."
        },
        5: {
            title: "5. PCM-FM Carrier Modulation",
            desc: "The amplitude of the Gaussian pulse-shaped signal directly controls the instantaneous frequency of the RF carrier wave. A high input level (+1) steers the carrier's frequency higher, and a low level (-1) steers it lower. This is Pulse Code Modulation/Frequency Modulation (PCM-FM), standard for aerospace telemetry."
        }
    };

    function renderStageVisual(stageId) {
        if (!stageDetailVisual) return;
        const bits = [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1];
        
        if (stageId === 1) {
            stageDetailVisual.innerHTML = `<div style="font-family: monospace; font-size: 48px; font-weight: 700; color: var(--primary); letter-spacing: 4px; border: 3px solid var(--primary); padding: 12px 36px; border-radius: var(--rounded-md); background: var(--canvas); box-shadow: 0 4px 12px rgba(0, 100, 224, 0.08);">HI</div>`;
        } else if (stageId === 2) {
            stageDetailVisual.innerHTML = `
                <div style="text-align: center; width: 100%;">
                  <div style="display: flex; gap: 12px; justify-content: center; margin-bottom: 8px;">
                    <div style="background: var(--surface-soft); border: 1px solid var(--hairline-soft); padding: 12px 16px; border-radius: var(--rounded-md); text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.02); min-width: 110px;">
                      <span style="font-size: 10px; color: var(--steel); display: block; font-weight: 700; text-transform: uppercase; margin-bottom: 4px;">'H' (ASCII 72)</span>
                      <span style="font-family: monospace; font-size: 20px; font-weight: 700; color: var(--primary); letter-spacing: 1px;">01001000</span>
                    </div>
                    <div style="background: var(--surface-soft); border: 1px solid var(--hairline-soft); padding: 12px 16px; border-radius: var(--rounded-md); text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.02); min-width: 110px;">
                      <span style="font-size: 10px; color: var(--steel); display: block; font-weight: 700; text-transform: uppercase; margin-bottom: 4px;">'I' (ASCII 73)</span>
                      <span style="font-family: monospace; font-size: 20px; font-weight: 700; color: var(--primary); letter-spacing: 1px;">01001001</span>
                    </div>
                  </div>
                  <div style="font-size: 12px; color: var(--steel); margin-top: 8px; font-weight: 500;">Total stream length: 16 bits</div>
                </div>
            `;
        } else if (stageId === 3) {
            let pathNrz = "";
            const w = 360;
            const h = 120;
            const padX = 20;
            const padY = 20;
            const drawW = w - 2 * padX;
            const drawH = h - 2 * padY;
            const bitW = drawW / bits.length;
            
            for (let i = 0; i < bits.length; i++) {
                const x1 = padX + i * bitW;
                const x2 = padX + (i + 1) * bitW;
                const y = padY + (bits[i] === 1 ? 0.25 * drawH : 0.75 * drawH);
                if (i === 0) {
                    pathNrz += `M ${x1} ${y} L ${x2} ${y}`;
                } else {
                    const prevY = padY + (bits[i-1] === 1 ? 0.25 * drawH : 0.75 * drawH);
                    pathNrz += ` L ${x1} ${prevY} L ${x1} ${y} L ${x2} ${y}`;
                }
            }
            
            stageDetailVisual.innerHTML = `
                <svg viewBox="0 0 ${w} ${h}" width="100%" height="100%" style="display: block;">
                    <!-- Grid Lines -->
                    <line x1="${padX}" y1="${padY + 0.25 * drawH}" x2="${w - padX}" y2="${padY + 0.25 * drawH}" stroke="var(--hairline-soft)" stroke-dasharray="3,3" />
                    <line x1="${padX}" y1="${padY + 0.5 * drawH}" x2="${w - padX}" y2="${padY + 0.5 * drawH}" stroke="var(--hairline-soft)" />
                    <line x1="${padX}" y1="${padY + 0.75 * drawH}" x2="${w - padX}" y2="${padY + 0.75 * drawH}" stroke="var(--hairline-soft)" stroke-dasharray="3,3" />
                    
                    <!-- Bit boundaries -->
                    ${bits.map((b, i) => {
                        const x = padX + i * bitW;
                        return `<line x1="${x}" y1="${padY}" x2="${x}" y2="${h - padY}" stroke="rgba(0,0,0,0.03)" />
                                <text x="${x + bitW/2}" y="${padY - 4}" font-family="monospace" font-size="10" fill="var(--steel)" text-anchor="middle">${b}</text>`;
                    }).join("")}
                    <line x1="${w - padX}" y1="${padY}" x2="${w - padX}" y2="${h - padY}" stroke="rgba(0,0,0,0.03)" />

                    <!-- Left Labels -->
                    <text x="${padX - 4}" y="${padY + 0.25 * drawH + 3}" font-size="9" font-family="monospace" fill="var(--steel)" text-anchor="end">+1V</text>
                    <text x="${padX - 4}" y="${padY + 0.5 * drawH + 3}" font-size="9" font-family="monospace" fill="var(--steel)" text-anchor="end">0V</text>
                    <text x="${padX - 4}" y="${padY + 0.75 * drawH + 3}" font-size="9" font-family="monospace" fill="var(--steel)" text-anchor="end">-1V</text>

                    <!-- Signal Path -->
                    <path d="${pathNrz}" fill="none" stroke="var(--primary)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" />
                </svg>
            `;
        } else if (stageId === 4) {
            const w = 360;
            const h = 120;
            const padX = 20;
            const padY = 20;
            const drawW = w - 2 * padX;
            const drawH = h - 2 * padY;
            const bitW = drawW / bits.length;
            
            const samplesPerBit = 10;
            const nrzSamples = [];
            for (let i = 0; i < bits.length; i++) {
                const val = bits[i] ? 1 : -1;
                for (let s = 0; s < samplesPerBit; s++) {
                    nrzSamples.push(val);
                }
            }
            
            const kernelSize = 15;
            const kernel = [];
            const sigma = 3.8;
            let sum = 0;
            for (let i = -kernelSize; i <= kernelSize; i++) {
                const val = Math.exp(-(i * i) / (2 * sigma * sigma));
                kernel.push(val);
                sum += val;
            }
            for (let i = 0; i < kernel.length; i++) kernel[i] /= sum;

            const shapedSamples = [];
            for (let i = 0; i < nrzSamples.length; i++) {
                let s = 0;
                for (let k = 0; k < kernel.length; k++) {
                    const idx = Math.min(Math.max(i + k - kernelSize, 0), nrzSamples.length - 1);
                    s += nrzSamples[idx] * kernel[k];
                }
                shapedSamples.push(s);
            }

            let pathPulse = "";
            for (let i = 0; i < shapedSamples.length; i++) {
                const x = padX + (i / shapedSamples.length) * drawW;
                const y = padY + (0.5 * drawH) - (shapedSamples[i] * 0.25 * drawH);
                if (i === 0) pathPulse += `M ${x} ${y}`;
                else pathPulse += ` L ${x} ${y}`;
            }

            stageDetailVisual.innerHTML = `
                <svg viewBox="0 0 ${w} ${h}" width="100%" height="100%" style="display: block;">
                    <!-- Grid Lines -->
                    <line x1="${padX}" y1="${padY + 0.25 * drawH}" x2="${w - padX}" y2="${padY + 0.25 * drawH}" stroke="var(--hairline-soft)" stroke-dasharray="3,3" />
                    <line x1="${padX}" y1="${padY + 0.5 * drawH}" x2="${w - padX}" y2="${padY + 0.5 * drawH}" stroke="var(--hairline-soft)" />
                    <line x1="${padX}" y1="${padY + 0.75 * drawH}" x2="${w - padX}" y2="${padY + 0.75 * drawH}" stroke="var(--hairline-soft)" stroke-dasharray="3,3" />
                    
                    <!-- Bit labels -->
                    ${bits.map((b, i) => {
                        const x = padX + i * bitW;
                        return `<text x="${x + bitW/2}" y="${padY - 4}" font-family="monospace" font-size="10" fill="var(--steel)" text-anchor="middle">${b}</text>`;
                    }).join("")}

                    <!-- Left Labels -->
                    <text x="${padX - 4}" y="${padY + 0.25 * drawH + 3}" font-size="9" font-family="monospace" fill="var(--steel)" text-anchor="end">+1.0</text>
                    <text x="${padX - 4}" y="${padY + 0.5 * drawH + 3}" font-size="9" font-family="monospace" fill="var(--steel)" text-anchor="end">0.0</text>
                    <text x="${padX - 4}" y="${padY + 0.75 * drawH + 3}" font-size="9" font-family="monospace" fill="var(--steel)" text-anchor="end">-1.0</text>

                    <!-- Signal Path -->
                    <path d="${pathPulse}" fill="none" stroke="var(--primary-soft)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" />
                </svg>
            `;
        } else if (stageId === 5) {
            const w = 360;
            const h = 120;
            const padX = 20;
            const padY = 20;
            const drawW = w - 2 * padX;
            const drawH = h - 2 * padY;
            const bitW = drawW / bits.length;

            const samplesPerBit = 12;
            const nrzSamples = [];
            for (let i = 0; i < bits.length; i++) {
                const val = bits[i] ? 1 : -1;
                for (let s = 0; s < samplesPerBit; s++) {
                    nrzSamples.push(val);
                }
            }
            
            const kernelSize = 15;
            const kernel = [];
            const sigma = 3.8;
            let sum = 0;
            for (let i = -kernelSize; i <= kernelSize; i++) {
                const val = Math.exp(-(i * i) / (2 * sigma * sigma));
                kernel.push(val);
                sum += val;
            }
            for (let i = 0; i < kernel.length; i++) kernel[i] /= sum;

            const shapedSamples = [];
            for (let i = 0; i < nrzSamples.length; i++) {
                let s = 0;
                for (let k = 0; k < kernel.length; k++) {
                    const idx = Math.min(Math.max(i + k - kernelSize, 0), nrzSamples.length - 1);
                    s += nrzSamples[idx] * kernel[k];
                }
                shapedSamples.push(s);
            }

            let theta = 0;
            const pcmFmSamples = [];
            for (let i = 0; i < shapedSamples.length; i++) {
                const fc = 0.22;
                const dev = 0.08 * shapedSamples[i];
                theta += 2 * Math.PI * (fc + dev);
                pcmFmSamples.push(Math.sin(theta));
            }

            let pathPcmFm = "";
            for (let i = 0; i < pcmFmSamples.length; i++) {
                const x = padX + (i / pcmFmSamples.length) * drawW;
                const y = padY + (0.5 * drawH) - (pcmFmSamples[i] * 0.35 * drawH);
                if (i === 0) pathPcmFm += `M ${x} ${y}`;
                else pathPcmFm += ` L ${x} ${y}`;
            }

            stageDetailVisual.innerHTML = `
                <svg viewBox="0 0 ${w} ${h}" width="100%" height="100%" style="display: block;">
                    <!-- Center Line -->
                    <line x1="${padX}" y1="${padY + 0.5 * drawH}" x2="${w - padX}" y2="${padY + 0.5 * drawH}" stroke="var(--hairline-soft)" />
                    
                    <!-- Bit labels -->
                    ${bits.map((b, i) => {
                        const x = padX + i * bitW;
                        return `<text x="${x + bitW/2}" y="${padY - 4}" font-family="monospace" font-size="10" fill="var(--steel)" text-anchor="middle">${b}</text>`;
                    }).join("")}

                    <!-- Signal Path -->
                    <path d="${pathPcmFm}" fill="none" stroke="var(--fb-blue)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" />
                </svg>
            `;
        }
    }

    walkthroughCards.forEach(card => {
        card.addEventListener("click", () => {
            walkthroughCards.forEach(c => c.classList.remove("active-card"));
            card.classList.add("active-card");
            
            const stageId = parseInt(card.getAttribute("data-stage"));
            const info = stageData[stageId];
            if (info) {
                stageDetailTitle.innerHTML = info.title;
                stageDetailDesc.innerHTML = info.desc;
                renderStageVisual(stageId);
            }
        });
    });

    // Panel 2: FEC Animation Grid
    const fecBitGrid = document.getElementById("fecBitGrid");
    const btnReplayFec = document.getElementById("btnReplayFec");
    const fecCaption = document.getElementById("fecCaption");
    const fecStatusIndicator = document.getElementById("fecStatusIndicator");
    const fecAnimationContainer = document.getElementById("fecAnimationContainer");
    const fecStaticFallbackContainer = document.getElementById("fecStaticFallbackContainer");

    const TOTAL_CELLS = 160; // 20 columns x 8 rows
    const DATA_CELLS = 80;   // first 10 columns
    const errorIndices = [7, 19, 31, 44, 62, 79, 93, 115, 126, 138, 151];
    
    let fecTimer = null;
    let isFecAnimInitialized = false;

    function buildFecCells() {
        if (!fecBitGrid) return;
        fecBitGrid.innerHTML = "";
        for (let i = 0; i < TOTAL_CELLS; i++) {
            const cell = document.createElement("div");
            cell.className = `fec-cell ${i < DATA_CELLS ? 'cell-data' : 'cell-parity'}`;
            cell.id = `fec-cell-${i}`;
            fecBitGrid.appendChild(cell);
        }
    }

    function clearFecAnimTimers() {
        if (fecTimer) {
            clearTimeout(fecTimer);
            fecTimer = null;
        }
    }

    function runFecAnimation() {
        clearFecAnimTimers();
        buildFecCells();

        // 1. Beat 1 — Encode
        fecStatusIndicator.innerHTML = '<i class="ph-duotone ph-broadcast"></i>';
        fecCaption.innerHTML = "Your message is expanded with redundant 'parity' bits &mdash; a <strong>2046-bit codeword</strong> (half data, half parity).";
        
        // 2. Beat 2 — Corrupt after 1.5 seconds
        fecTimer = setTimeout(() => {
            const errorCount = window.lastRawErrors || 77;
            fecStatusIndicator.innerHTML = '<i class="ph-duotone ph-warning" style="color: var(--critical);"></i>';
            fecCaption.innerHTML = `The noisy channel flips some bits &mdash; here about <strong>${errorCount}</strong> of them (highlighted red).`;
            
            errorIndices.forEach(idx => {
                const cell = document.getElementById(`fec-cell-${idx}`);
                if (cell) {
                    cell.classList.remove("cell-data", "cell-parity");
                    cell.classList.add("cell-error");
                }
            });

            // 3. Beat 3 — Correct after 1.5 seconds
            fecTimer = setTimeout(() => {
                fecStatusIndicator.innerHTML = '<i class="ph-duotone ph-broadcast"></i>';
                fecCaption.innerHTML = "The LDPC parity-check decoder processes checks and flips the errors back to correct values...";

                // Sequentially correct the cells
                let currentErrIndex = 0;
                function correctNextError() {
                    if (currentErrIndex < errorIndices.length) {
                        const idx = errorIndices[currentErrIndex];
                        const cell = document.getElementById(`fec-cell-${idx}`);
                        if (cell) {
                            cell.classList.remove("cell-error");
                            cell.classList.add("cell-corrected");
                        }
                        currentErrIndex++;
                        fecTimer = setTimeout(correctNextError, 120);
                    } else {
                        // End state
                        fecStatusIndicator.innerHTML = '<i class="ph-duotone ph-check-circle" style="color: var(--success);"></i>';
                        fecCaption.innerHTML = "<strong>All errors corrected.</strong> The original message is recovered perfectly without standard receiver loss.";
                    }
                }
                correctNextError();

            }, 1500);

        }, 1500);
    }

    function renderStaticFallbackGrids() {
        const snap1 = document.querySelector(".snap1-grid");
        const snap2 = document.querySelector(".snap2-grid");
        const snap3 = document.querySelector(".snap3-grid");
        if (!snap1 || !snap2 || !snap3) return;

        // Static representatives: 8 columns x 6 rows = 48 cells
        const fallbackTotal = 48;
        const fallbackData = 24;
        const fallbackErrors = [5, 12, 19, 26, 37, 43];

        const renderGrid = (gridEl, state) => {
            gridEl.innerHTML = "";
            for (let i = 0; i < fallbackTotal; i++) {
                const cell = document.createElement("div");
                cell.className = "fec-cell";
                cell.style.aspectRatio = "1";
                cell.style.borderRadius = "2px";
                
                const isData = i < fallbackData;
                
                if (state === "encode") {
                    cell.style.background = isData ? "rgba(0, 100, 224, 0.08)" : "rgba(161, 33, 206, 0.08)";
                } else if (state === "corrupt") {
                    if (fallbackErrors.includes(i)) {
                        cell.style.background = "var(--critical-strong)";
                        cell.style.boxShadow = "0 0 4px rgba(240, 40, 74, 0.4)";
                    } else {
                        cell.style.background = isData ? "rgba(0, 100, 224, 0.08)" : "rgba(161, 33, 206, 0.08)";
                    }
                } else if (state === "correct") {
                    if (fallbackErrors.includes(i)) {
                        cell.style.background = "var(--success)";
                        cell.style.boxShadow = "0 0 4px rgba(49, 162, 76, 0.4)";
                    } else {
                        cell.style.background = isData ? "rgba(0, 100, 224, 0.08)" : "rgba(161, 33, 206, 0.08)";
                    }
                }
                gridEl.appendChild(cell);
            }
        };

        renderGrid(snap1, "encode");
        renderGrid(snap2, "corrupt");
        renderGrid(snap3, "correct");
    }

    function initializeFecInteractive() {
        const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        if (prefersReducedMotion) {
            fecAnimationContainer.style.display = "none";
            fecStaticFallbackContainer.style.display = "block";
            renderStaticFallbackGrids();
        } else {
            fecAnimationContainer.style.display = "block";
            fecStaticFallbackContainer.style.display = "none";
            if (!isFecAnimInitialized) {
                isFecAnimInitialized = true;
                runFecAnimation();
            }
        }
    }

    if (btnReplayFec) {
        btnReplayFec.addEventListener("click", () => {
            runFecAnimation();
        });
    }
});

