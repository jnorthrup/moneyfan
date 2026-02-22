/* 
    console/app.js 
    Brainless JS polling to update the DOM
*/

const API_POLL_INTERVAL = 1000;

// Chart.js Configuration
let equityChartInstance = null;

try {
    if (typeof Chart !== 'undefined') {
        Chart.defaults.color = 'rgba(255, 255, 255, 0.5)';
        Chart.defaults.font.family = 'Inter, sans-serif';
    }
} catch (e) {
    console.warn("Chart.js not initialized", e);
}

function initChart() {
    try {
        const canvas = document.getElementById('equityChart');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const gradient = ctx.createLinearGradient(0, 0, 0, 400);
        gradient.addColorStop(0, 'rgba(88, 166, 255, 0.5)');
        gradient.addColorStop(1, 'rgba(88, 166, 255, 0.0)');

        equityChartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Capital Equity',
                    data: [],
                    borderColor: '#58a6ff',
                    backgroundColor: gradient,
                    borderWidth: 2,
                    tension: 0.1,
                    fill: true,
                    pointRadius: 0,
                    pointHoverRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 0 // Disable animation for live fast redraw
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    } catch (e) {
        console.warn("Failed to init chart", e);
    }
}
let lastSessionStartTime = null;
let typingInterval = null;

// --- State Polling ---Loop
async function pollState() {
    try {
        const res = await fetch('/api/state');
        if (res.ok) {
            const data = await res.json();
            updateUI(data);
            updateCodecsUI(data);
            updateMissionClock(data.session_start_time);
            pulseHeartbeat();
        }
    } catch (err) {
        console.error("Failed to fetch state API");
        setOffline();
    }
}

async function pollCache() {
    try {
        const res = await fetch('/api/cache');
        if (res.ok) {
            const data = await res.json();
            updateCacheUI(data);
        }
    } catch (err) {
        console.error("Failed to fetch cache API");
    }
}

async function pollDrawthru() {
    try {
        const res = await fetch('/api/drawthru');
        if (res.ok) {
            const data = await res.json();
            updateDrawthruUI(data);
        }
    } catch (err) {
        console.error("Failed to fetch drawthru API");
    }
}

function formatDollar(val) {
    if (val === undefined || val === null) return "$0.00";
    return "$" + val.toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

function formatPct(val) {
    if (val === undefined || val === null) return "0.0%";
    return (val * 100).toFixed(1) + "%";
}

function updateUI(data) {
    if (data.status === 'booting') {
        document.getElementById('statusText').innerText = "System Booting...";
        return;
    }

    if (data.status === 'running') {
        document.getElementById('statusText').innerText = "Training Engine Live";
        document.getElementById('statusIndicator').className = "status-indicator live";
    } else if (data.status === 'stopped') {
        const totalEpisodes = data.history ? data.history.length : 0;
        if (totalEpisodes > 0) {
            document.getElementById('statusText').innerText = `Training Complete (${totalEpisodes} eps)`;
            document.getElementById('statusIndicator').className = "status-indicator complete";
        } else {
            document.getElementById('statusText').innerText = "Mission Complete (Stopped)";
            document.getElementById('statusIndicator').className = "status-indicator";
        }
    }

    // Top Level Metrics
    if (data.latest_metrics) {
        document.getElementById('globalEpoch').innerText = data.latest_metrics.total_trained;
        document.getElementById('globalCapital').innerText = formatDollar(data.latest_metrics.current_capital);

        const pnl = data.latest_metrics.total_realized_pnl;
        const pnlEl = document.getElementById('globalPnl');
        pnlEl.innerText = (pnl >= 0 ? "+" : "") + formatDollar(pnl);
        pnlEl.className = "metric-value highlight" + (pnl < 0 ? " negative" : "");
    }

    // Determine the active equity curve
    if (data.history && data.history.length > 0) {
        // Last finished episode metrics
        const lastEp = data.history[data.history.length - 1];
        document.getElementById('lastWinRate').innerText = formatPct(lastEp.hit_rate);

        const epPnlEl = document.getElementById('lastPnl');
        epPnlEl.innerText = (lastEp.realized_pnl >= 0 ? "+" : "") + formatDollar(lastEp.realized_pnl);
        epPnlEl.style.color = lastEp.realized_pnl >= 0 ? "var(--accent-green)" : "var(--accent-red)";

        document.getElementById('lastTrades').innerText = lastEp.total_trades || 0;
        document.getElementById('lastLoss').innerText = (lastEp.predictor_loss || 0).toFixed(4);
        document.getElementById('lastShocks').innerText = lastEp.outlier_extents || 0;
        document.getElementById('lastExpert').innerText = lastEp.winning_agent || "HRM Manager";

        // Draw graph
        if (lastEp.equity_curve && lastEp.equity_curve.length > 0) {
            if (equityChartInstance) {
                try {
                    equityChartInstance.data.labels = lastEp.equity_curve.map((_, i) => i);
                    equityChartInstance.data.datasets[0].data = lastEp.equity_curve;
                    equityChartInstance.update();
                } catch (e) {
                    console.warn("Chart update failed", e);
                }
            }
        }

        // Leaderboard Construction
        if (lastEp.codec_scores) {
            const listEl = document.getElementById('leaderboardList');
            listEl.innerHTML = "";
            let sortedCodecs = Object.entries(lastEp.codec_scores).sort((a, b) => b[1] - a[1]);

            // Show top 16 active
            sortedCodecs.slice(0, 16).forEach(([name, score]) => {
                const li = document.createElement('li');
                li.className = "leaderboard-item";
                li.innerHTML = `
                    <span class="expert-name">${name.replace('___class__', '')}</span>
                    <span class="expert-score">${score.toFixed(2)}</span>
                `;
                listEl.appendChild(li);
            });
        }
    }
}

function updateCacheUI(data) {
    // Always render transfers if they exist
    if (data.transfers) {
        renderTransfers(data.transfers);
    }

    if (data.cache_status === 'offline') return;

    const badge = document.getElementById('cacheStatus');
    badge.innerText = "Online";
    badge.className = "badge online";

    // Progress bar calculations
    const fillPct = (data.current_size / data.max_size) * 100;
    document.getElementById('cacheFill').style.width = fillPct + "%";
    document.getElementById('cachePct').innerText = `${data.current_size} / ${data.max_size} keys`;

    // Row total
    document.getElementById('cacheRows').innerText = (data.memory_rows || 0).toLocaleString() + " frames";

    // Recently Accessed List (Hot Cache) 
    // access_order pushes newly requested to the end, so reverse it
    const listEl = document.getElementById('cacheList');
    listEl.innerHTML = "";
    if (data.access_order && data.access_order.length > 0) {
        const topRequested = data.access_order.slice(-10).reverse();
        topRequested.forEach(key => {
            const li = document.createElement('li');
            li.innerText = key.replace('candles:', '');
            listEl.appendChild(li);
        });
    } else {
        listEl.innerHTML = '<li class="empty-state">No cache activity</li>';
    }
}

function updateDrawthruUI(data) {
    const statusEl = document.getElementById('drawthruStatus');
    const rowsEl = document.getElementById('drawthruRows');
    const symbolsEl = document.getElementById('drawthruSymbols');
    const latestEl = document.getElementById('drawthruLatest');
    const tableEl = document.getElementById('drawthruTable');
    const topList = document.getElementById('drawthruTopSymbols');
    if (!statusEl || !rowsEl || !symbolsEl || !latestEl || !tableEl || !topList) return;

    const status = data && data.status ? data.status : 'offline';
    statusEl.innerText = status.toUpperCase();
    statusEl.className = 'badge';
    if (status === 'ok') {
        statusEl.classList.add('online');
    }

    if (status !== 'ok') {
        rowsEl.innerText = '0';
        symbolsEl.innerText = '0';
        latestEl.innerText = '--';
        tableEl.innerText = '--';
        topList.innerHTML = `<li class="empty-state">${data.error || data.db_path || 'DuckDB offline'}</li>`;
        return;
    }

    rowsEl.innerText = Number(data.row_count || 0).toLocaleString();
    symbolsEl.innerText = String(data.symbol_count || 0);
    latestEl.innerText = (data.max_ts || '--').replace('T', ' ').slice(0, 19);
    tableEl.innerText = data.table || '--';

    const top = Array.isArray(data.top_symbols) ? data.top_symbols : [];
    topList.innerHTML = '';
    if (!top.length) {
        topList.innerHTML = '<li class="empty-state">No imported symbols yet</li>';
        return;
    }
    top.forEach(row => {
        const li = document.createElement('li');
        const lastTs = (row.last_ts || '--').replace('T', ' ').slice(0, 16);
        li.innerText = `${row.symbol} • ${Number(row.row_count || 0).toLocaleString()} rows • ${lastTs}`;
        topList.appendChild(li);
    });
}

function renderTransfers(transfers) {
    const grid = document.getElementById('transferGrid');
    if (!grid) return;

    if (!transfers || transfers.length === 0) {
        grid.innerHTML = '<div class="empty-state">No active transfers</div>';
        return;
    }

    grid.innerHTML = '';

    transfers.forEach(xfer => {
        const item = document.createElement('div');
        // 'file' or 'api'
        const typeClass = xfer.type === 'file' ? 'file' : 'api';
        const icon = xfer.type === 'file' ? '📄' : '🌐';

        item.className = `transfer-item ${typeClass}`;

        // Progress parsing
        let progressHtml = '';
        let metaHtml = '';

        if (xfer.progress_pct !== undefined && xfer.progress_pct >= 0) {
            progressHtml = `<div class="transfer-progress-fill" style="width: ${Math.min(100, xfer.progress_pct)}%;"></div>`;
            metaHtml = `${Math.round(xfer.progress_pct)}%`;
        } else {
            progressHtml = `<div class="transfer-progress-indeterminate"></div>`;
            metaHtml = `Active`;
        }

        if (xfer.status_text) {
            metaHtml = xfer.status_text;
        }

        item.innerHTML = `
            <div class="transfer-header">
                <div class="transfer-title">
                    <span class="transfer-icon">${icon}</span>
                    <span>${xfer.name || 'Unnamed Transfer'}</span>
                </div>
                <div class="transfer-meta">${metaHtml}</div>
            </div>
            <div class="transfer-progress-container">
                ${progressHtml}
            </div>
        `;

        grid.appendChild(item);
    });
}

function setOffline() {
    document.getElementById('statusText').innerText = "Disconnected from Daemon";
    document.getElementById('statusIndicator').className = "status-indicator";
    document.getElementById('cacheStatus').className = "badge";
    document.getElementById('cacheStatus').innerText = "Offline";
}

// VQA Logic
async function submitVQA() {
    const inputEl = document.getElementById('vqaInput');
    const msg = inputEl.value.trim();
    if (!msg) return;

    appendVQAMessage('user', msg);
    inputEl.value = "";

    // Show thinking indicator
    const thinkingId = appendVQAMessage('system', "PROCESSING_QUERY...", true);

    try {
        const res = await fetch('/api/vqa', {
            method: 'POST',
            body: JSON.stringify({ question: msg, context: 'cockpit_view' }),
            headers: { 'Content-Type': 'application/json' }
        });
        if (res.ok) {
            const data = await res.json();
            removeMessage(thinkingId);
            simulateTyping('system', data.answer);
        }
    } catch (err) {
        removeMessage(thinkingId);
        appendVQAMessage('system', "Comms error: Pilot unavailable.");
    }
}

function simulateTyping(role, text) {
    const historyEl = document.getElementById('vqaHistory');
    const div = document.createElement('div');
    div.className = `vqa-msg ${role}`;
    div.innerText = "PILOT_DECRYPTING...";
    historyEl.appendChild(div);
    historyEl.scrollTop = historyEl.scrollHeight;

    let i = 0;
    const interval = setInterval(() => {
        div.innerText = "Pilot: " + text.slice(0, i) + "█";
        i++;
        historyEl.scrollTop = historyEl.scrollHeight;
        if (i > text.length) {
            clearInterval(interval);
            div.innerText = "Pilot: " + text;
        }
    }, 20);
}

function removeMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function appendVQAMessage(role, text, isThinking = false) {
    const historyEl = document.getElementById('vqaHistory');
    const div = document.createElement('div');
    const id = "msg-" + Date.now();
    div.id = id;
    div.className = `vqa-msg ${role} ${isThinking ? 'thinking' : ''}`;
    div.innerText = (role === 'system' ? "Pilot: " : "") + text;
    historyEl.appendChild(div);
    historyEl.scrollTop = historyEl.scrollHeight;
    return id;
}

function updateMissionClock(startTime) {
    if (!startTime) return;
    const start = new Date(startTime);
    const now = new Date();
    const diff = Math.floor((now - start) / 1000);

    const h = Math.floor(diff / 3600).toString().padStart(2, '0');
    const m = Math.floor((diff % 3600) / 60).toString().padStart(2, '0');
    const s = (diff % 60).toString().padStart(2, '0');

    document.getElementById('missionClock').innerText = `MISSION: ${h}:${m}:${s}`;
}

function pulseHeartbeat() {
    const pulse = document.getElementById('kernelHeartbeat');
    if (!pulse) return;
    pulse.style.transform = 'scale(1.4)';
    setTimeout(() => {
        pulse.style.transform = 'scale(1)';
    }, 200);
}

document.getElementById('vqaSubmit').addEventListener('click', submitVQA);
document.getElementById('vqaInput').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') submitVQA();
});


// Tab Switching Logic
document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', (e) => {
        e.preventDefault();

        // Remove active class from all tabs
        document.querySelectorAll('.nav-item').forEach(nav => nav.classList.remove('active'));

        // Add active class to clicked tab
        e.currentTarget.classList.add('active');

        // Hide all viewports
        document.querySelectorAll('.viewport').forEach(vp => vp.classList.remove('active'));

        // Show target viewport
        const targetId = e.currentTarget.getAttribute('data-target');
        if (targetId) {
            const targetEl = document.getElementById(targetId);
            if (targetEl) targetEl.classList.add('active');
        }
    });
});

// Init
initChart();
setInterval(() => {
    pollState();
    pollCache();
    pollDrawthru();
}, API_POLL_INTERVAL);

function updateCodecsUI(data) {
    if (!data.latest_metrics && !data.history) return;

    const radar = document.getElementById('convictionRadar');
    if (!radar) return;

    // Use the latest scores from the history if available
    let scores = {};
    if (data.history && data.history.length > 0) {
        scores = data.history[data.history.length - 1].codec_scores || {};
    } else if (data.latest_expert_weights) {
        scores = data.latest_expert_weights;
    }

    if (Object.keys(scores).length === 0) {
        radar.innerHTML = '<div class="empty-state">No neural data available</div>';
        return;
    }

    renderRadar(radar, scores);
    updateGauges(data, scores);
}

function renderRadar(container, scores) {
    container.innerHTML = '';
    const sortedKeys = Object.keys(scores).sort();

    // Find max score for normalization
    const maxScore = Math.max(...Object.values(scores), 1.0);

    sortedKeys.forEach(name => {
        const val = scores[name];
        const pct = (val / maxScore) * 100;

        const row = document.createElement('div');
        row.className = 'radar-row';
        row.innerHTML = `
            <div class="radar-label">${name}</div>
            <div class="radar-bar-container">
                <div class="radar-bar-fill" style="width: ${pct}%"></div>
            </div>
            <div class="radar-value">${val.toFixed(2)}</div>
        `;
        container.appendChild(row);
    });
}

function updateGauges(data, scores) {
    // 1. Regime Entropy (How much experts disagree)
    const vals = Object.values(scores);
    if (vals.length > 0) {
        const sum = vals.reduce((a, b) => a + Math.abs(b), 0);
        const entropy = sum / vals.length; // Simplified metric
        const entropyPct = Math.min(100, (entropy / 2) * 100);

        const valueEl = document.getElementById('entropyValue');
        if (valueEl) valueEl.innerText = entropy.toFixed(2);
        setGaugeFill('entropyFill', entropyPct);

        const labelEl = document.getElementById('entropyLabel');
        if (labelEl) {
            if (entropy > 1.2) {
                labelEl.innerText = "Chaos / Volatility";
                labelEl.style.color = "var(--accent-red)";
            } else if (entropy > 0.6) {
                labelEl.innerText = "Transitional Regime";
                labelEl.style.color = "var(--accent-purple)";
            } else {
                labelEl.innerText = "Consensus Stable";
                labelEl.style.color = "var(--accent-green)";
            }
        }
    }

    // 2. Neural Confidence (Hit Rate from last metric)
    const hitRate = (data.history && data.history.length > 0)
        ? data.history[data.history.length - 1].hit_rate * 100
        : 0;

    const confValueEl = document.getElementById('confidenceValue');
    if (confValueEl) confValueEl.innerText = Math.round(hitRate) + "%";
    setGaugeFill('confidenceFill', hitRate);

    const confLabelEl = document.getElementById('confidenceLabel');
    if (confLabelEl) {
        confLabelEl.innerText = hitRate > 55 ? "Strong Predictive Lead" : "Calibration Required";
        confLabelEl.style.color = hitRate > 55 ? "var(--accent-green)" : "var(--text-secondary)";
    }
}

function setGaugeFill(id, percent) {
    const el = document.getElementById(id);
    if (!el) return;
    const radius = 45;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (percent / 100) * circumference;
    el.style.strokeDasharray = `${circumference} ${circumference}`;
    el.style.strokeDashoffset = offset;
}
