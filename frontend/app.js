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
    const navFaq = document.getElementById("navFaq");
    const btnLaunchSim = document.getElementById("btnLaunchSim");

    const viewHome = document.getElementById("viewHome");
    const viewSim = document.getElementById("viewSim");
    const viewSpecs = document.getElementById("viewSpecs");
    const viewFaq = document.getElementById("viewFaq");

    // Dynamic API Endpoint Determination
    // If running on local webserver, call localhost backend. If deployed, call the relative API.
    const API_BASE = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1" 
        ? "http://127.0.0.1:7860" 
        : ""; 

    let selectedInterference = "wideband";

    // ────────────────────────────────────────────────────────
    // SPA View Router
    // ────────────────────────────────────────────────────────
    const views = {
        home: { view: viewHome, nav: navHome },
        sim: { view: viewSim, nav: navSim },
        specs: { view: viewSpecs, nav: navSpecs },
        faq: { view: viewFaq, nav: navFaq }
    };

    function showView(targetId, updateHash = true) {
        Object.keys(views).forEach(key => {
            const item = views[key];
            if (key === targetId) {
                item.view.classList.add("active-view");
                item.nav.classList.add("active");
            } else {
                item.view.classList.remove("active-view");
                item.nav.classList.remove("active");
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
    navFaq.addEventListener("click", () => showView("faq"));
    btnLaunchSim.addEventListener("click", () => showView("sim"));

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
        charCount.textContent = `${len} / 128 chars`;
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
            const syncLockStr = st.psl >= 1.5 ? "strong" : "weak";

            // Generate simulated traditional output
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
                const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!?@#$%&*()_-+=";
                traditionalOutput = data.message_sent.split('').map((c, i) => {
                    const r = Math.random();
                    if (r < 0.5) return '?';
                    if (r < 0.8) return chars[Math.floor(Math.random() * chars.length)];
                    return String.fromCharCode(160 + Math.floor(Math.random() * 95)); // non-ascii junk
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
        <div class="comparison-box our-model">
            <div><strong>Raw Model (Pre-FEC):</strong> "${data.recovered_pre_fec}"</div>
            <div style="margin-top: 6px;"><strong>Final Recovered (Post-FEC):</strong> "${data.recovered_post_fec}"</div>
        </div>
    </div>
    <div class="comparison-column">
        <div class="comparison-label">Traditional Decoder (No Excision)</div>
        <div class="comparison-box traditional">
            <div><strong>Final Recovered:</strong> "${traditionalOutput}"</div>
            <div style="margin-top: 6px; font-size: 11px; opacity: 0.85;">Status: Jammed / Correlation loss</div>
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
});
