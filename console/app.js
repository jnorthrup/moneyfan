/* 
    console/app.js 
    Brainless JS polling to update the DOM
*/

const API_POLL_INTERVAL = 1000;

// Chart.js Configuration
let equityChartInstance = null;
let drawthruPreviewChartInstance = null;

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

function initDrawthruPreviewChart() {
    try {
        const canvas = document.getElementById('drawthruPreviewChart');
        if (!canvas || typeof Chart === 'undefined') return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const gradient = ctx.createLinearGradient(0, 0, 0, 140);
        gradient.addColorStop(0, 'rgba(63, 185, 80, 0.35)');
        gradient.addColorStop(1, 'rgba(63, 185, 80, 0.0)');

        drawthruPreviewChartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Close',
                    data: [],
                    borderColor: '#3fb950',
                    backgroundColor: gradient,
                    borderWidth: 1.5,
                    tension: 0.15,
                    fill: true,
                    pointRadius: 0,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 0 },
                scales: {
                    x: { display: false, grid: { display: false } },
                    y: { grid: { color: 'rgba(255,255,255,0.05)' } },
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            title: (items) => (items?.[0]?.label || '').replace('T', ' '),
                            label: (item) => `Close: ${Number(item.raw).toFixed(4)}`,
                        }
                    }
                }
            }
        });
    } catch (e) {
        console.warn("Failed to init drawthru preview chart", e);
    }
}
let lastStateData = null;
let lastDrawthruData = null;
let settingsInitializedFromState = false;
const SANE_RUNTIME_CONTROL_DEFAULTS = {
    notional: 100.0,
    pair_width: 30,
    bar_sequences_per_episode: 100,
    epochs: 1,
    min_bar_window: 64,
    max_bar_window: 256,
    cache_size: 1000,
    candles_per_extent: 1000,
    shock_z_threshold: 2.0,
    bar_shock_z_threshold: 3.0,
    max_adaptive_replays: 3,
    use_mechanical_veto: false,
    replay_coalescing: false,
    replay_coalescing_chunk_size: 8,
    optimizer_name: "adamw",
    learning_rate: 1e-4,
    weight_decay: 1e-2,
};

// --- State Polling ---Loop
async function pollState() {
    try {
        const res = await fetch('/api/state');
        if (res.ok) {
            const data = await res.json();
            lastStateData = data;
            updateUI(data);
            updateCodecsUI(data);
            syncRealtimeControlsFromState(data);
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
            lastDrawthruData = data;
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

    updateTrainingTelemetry(data);
    renderSampleLog(data.samples || []);
    updateControlRuntimeSnapshot(data);

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
    const previewSymbolEl = document.getElementById('drawthruPreviewSymbol');
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
        if (previewSymbolEl) previewSymbolEl.innerText = '--';
        if (drawthruPreviewChartInstance) {
            drawthruPreviewChartInstance.data.labels = [];
            drawthruPreviewChartInstance.data.datasets[0].data = [];
            drawthruPreviewChartInstance.update();
        }
        topList.innerHTML = `<li class="empty-state">${data.error || data.db_path || 'DuckDB offline'}</li>`;
        updateTelemetryDrawthru(data);
        return;
    }

    rowsEl.innerText = Number(data.row_count || 0).toLocaleString();
    symbolsEl.innerText = String(data.symbol_count || 0);
    latestEl.innerText = (data.max_ts || '--').replace('T', ' ').slice(0, 19);
    tableEl.innerText = data.table || '--';
    if (previewSymbolEl) previewSymbolEl.innerText = data.preview_symbol || '--';

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

    if (drawthruPreviewChartInstance && Array.isArray(data.preview_bars) && data.preview_bars.length > 0) {
        const bars = data.preview_bars;
        drawthruPreviewChartInstance.data.labels = bars.map(b => (b.timestamp || '').slice(11, 19));
        drawthruPreviewChartInstance.data.datasets[0].data = bars.map(b => Number(b.close || 0));
        drawthruPreviewChartInstance.update();
    }

    updateTelemetryDrawthru(data);
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
    setControlStatus("Disconnected from daemon.", "error");
}

function setControlStatus(message, kind = "neutral") {
    const el = document.getElementById('controlStatus');
    if (!el) return;
    el.textContent = message;
    el.className = `control-status ${kind}`;
}

function formatPctSigned(val) {
    if (val === undefined || val === null || Number.isNaN(Number(val))) return "--";
    const n = Number(val) * 100;
    const sign = n > 0 ? "+" : "";
    return `${sign}${n.toFixed(2)}%`;
}

function fmtFixed(val, digits = 4, fallback = "--") {
    const n = Number(val);
    return Number.isFinite(n) ? n.toFixed(digits) : fallback;
}

function escapeHtml(str) {
    return String(str ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

function findLatestCurveResult(history) {
    if (!Array.isArray(history)) return null;
    for (let i = history.length - 1; i >= 0; i--) {
        const row = history[i];
        if (Array.isArray(row?.equity_curve) && row.equity_curve.length > 0) {
            return row;
        }
    }
    return null;
}

function computeDrawdownStats(equityCurve) {
    if (!Array.isArray(equityCurve) || equityCurve.length === 0) {
        return { currentDrawdown: 0, maxDrawdown: 0 };
    }
    let peak = Number(equityCurve[0]) || 0;
    let last = peak;
    let maxDd = 0;
    equityCurve.forEach((pt) => {
        const v = Number(pt);
        if (!Number.isFinite(v)) return;
        if (v > peak) peak = v;
        last = v;
        if (peak > 0) {
            const dd = (v - peak) / peak;
            if (dd < maxDd) maxDd = dd;
        }
    });
    const currentDd = peak > 0 ? (last - peak) / peak : 0;
    return { currentDrawdown: currentDd, maxDrawdown: maxDd };
}

function updateTrainingTelemetry(data) {
    const logRowsEl = document.getElementById('telemetryAgentLogRows');
    const lossKpiEl = document.getElementById('telemetryLossKpi');
    const currentDdEl = document.getElementById('telemetryCurrentDd');
    const maxDdEl = document.getElementById('telemetryMaxDd');
    const vetoEl = document.getElementById('telemetryVetoKpi');
    const replayEl = document.getElementById('telemetryReplayKpi');
    const optEl = document.getElementById('telemetryOptimizerKpi');
    if (!logRowsEl) return;

    const history = Array.isArray(data?.history) ? data.history : [];
    if (!history.length) {
        logRowsEl.innerHTML = '<tr><td colspan="5" class="empty-state">Waiting for training results...</td></tr>';
        if (lossKpiEl) lossKpiEl.textContent = '0.0000';
        if (currentDdEl) currentDdEl.textContent = '0.00%';
        if (maxDdEl) maxDdEl.textContent = '0.00%';
        if (vetoEl) vetoEl.textContent = '0';
        if (replayEl) replayEl.textContent = '0';
        if (optEl) optEl.textContent = data?.training_config?.optimizer_name || '--';
        return;
    }

    const latest = history[history.length - 1] || {};
    const latestCurveResult = findLatestCurveResult(history);
    const ddStats = computeDrawdownStats(latestCurveResult?.equity_curve);

    if (lossKpiEl) lossKpiEl.textContent = fmtFixed(latest.predictor_loss ?? 0, 4, "0.0000");
    if (currentDdEl) currentDdEl.textContent = formatPctSigned(ddStats.currentDrawdown);
    if (maxDdEl) maxDdEl.textContent = formatPctSigned(ddStats.maxDrawdown);
    if (vetoEl) vetoEl.textContent = String(latest.veto_count ?? 0);
    if (replayEl) replayEl.textContent = String(latest.replay_eval_count ?? latest.optimizer_replays ?? 0);
    if (optEl) {
        optEl.textContent = latest.optimizer_name || data?.training_config?.optimizer_name || '--';
    }

    const recent = history.slice(-12).reverse();
    logRowsEl.innerHTML = recent.map((row) => {
        const ep = Number.isFinite(Number(row.episode_id)) ? Number(row.episode_id) : "--";
        const epochLabel = row.total_epochs && row.epoch
            ? `${ep}.${row.epoch}/${row.total_epochs}`
            : String(ep);
        const pnl = Number(row.realized_pnl ?? 0);
        const pnlText = `${pnl >= 0 ? "+" : ""}${formatDollar(pnl)}`;
        const lossText = fmtFixed(row.predictor_loss ?? 0, 4, "0.0000");
        const dd = Array.isArray(row.equity_curve) ? computeDrawdownStats(row.equity_curve).maxDrawdown : null;
        const ddText = dd === null ? '--' : formatPctSigned(dd);
        const pnlClass = pnl < 0 ? 'negative' : 'positive';
        return `
            <tr>
                <td>${escapeHtml(epochLabel)}</td>
                <td title="${escapeHtml(row.winning_agent || '--')}">${escapeHtml((row.winning_agent || '--').replace('___class__', ''))}</td>
                <td class="${pnlClass}">${escapeHtml(pnlText)}</td>
                <td>${escapeHtml(lossText)}</td>
                <td>${escapeHtml(ddText)}</td>
            </tr>
        `;
    }).join('');
}

function updateTelemetryDrawthru(data) {
    const metaEl = document.getElementById('telemetryDbWindowMeta');
    const rowsEl = document.getElementById('telemetryDbWindowRows');
    if (!rowsEl) return;

    const bars = Array.isArray(data?.preview_bars) ? data.preview_bars : [];
    if (!data || data.status !== 'ok' || bars.length === 0) {
        if (metaEl) metaEl.textContent = data?.error || data?.db_path || 'Waiting for drawthru snapshot...';
        rowsEl.innerHTML = '<tr><td colspan="5" class="empty-state">Waiting for DuckDB preview bars...</td></tr>';
        return;
    }

    if (metaEl) {
        metaEl.textContent = `${data.preview_symbol || '--'} • ${data.table || '--'} • ${bars.length} bars`;
    }

    const recentBars = bars.slice(-12).reverse();
    rowsEl.innerHTML = recentBars.map((b) => `
        <tr>
            <td>${escapeHtml((b.timestamp || '--').replace('T', ' ').slice(11, 19))}</td>
            <td>${escapeHtml(fmtFixed(b.open, 4))}</td>
            <td>${escapeHtml(fmtFixed(b.high, 4))}</td>
            <td>${escapeHtml(fmtFixed(b.low, 4))}</td>
            <td>${escapeHtml(fmtFixed(b.close, 4))}</td>
        </tr>
    `).join('');
}

function renderSampleLog(samples) {
    const body = document.getElementById('sampleLog');
    if (!body) return;
    if (!Array.isArray(samples) || samples.length === 0) {
        body.innerHTML = '<tr><td colspan="4" class="empty-state">Waiting for sampling events...</td></tr>';
        return;
    }

    const recent = samples.slice(-30).reverse();
    body.innerHTML = recent.map((s) => {
        const symbol = s.symbol || (Array.isArray(s.symbols) ? s.symbols.join(',') : '--');
        const windowLabel = s.window || s.bar_window || s.bar_window_len || '--';
        const lenLabel = s.len || s.length || s.extent_len || '--';
        const ts = (s.timestamp || s.ts || '').toString();
        const timeLabel = ts ? ts.replace('T', ' ').slice(0, 19) : '--';
        return `
            <tr>
                <td>${escapeHtml(symbol)}</td>
                <td>${escapeHtml(windowLabel)}</td>
                <td>${escapeHtml(lenLabel)}</td>
                <td>${escapeHtml(timeLabel)}</td>
            </tr>
        `;
    }).join('');
}

function updateControlRuntimeSnapshot(data) {
    const cfg = data?.training_config || {};
    const setText = (id, value) => {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    };
    setText('ctlSnapshotOptimizer', cfg.optimizer_name || '--');
    setText('ctlSnapshotLr', Number.isFinite(Number(cfg.learning_rate)) ? Number(cfg.learning_rate).toExponential(2) : '--');
    setText('ctlSnapshotWd', Number.isFinite(Number(cfg.weight_decay)) ? Number(cfg.weight_decay).toExponential(2) : '--');
    setText('ctlSnapshotState', (data?.status || '--').toUpperCase());
}

function populateRealtimeControls(config, options = {}) {
    const markInitialized = options.markInitialized !== false;
    if (!config) return;
    document.querySelectorAll('[data-control-key]').forEach((input) => {
        const key = input.getAttribute('data-control-key');
        if (!key || !(key in config)) return;
        const value = config[key];
        if (input.type === 'checkbox') {
            input.checked = Boolean(value);
        } else if (value !== undefined && value !== null) {
            input.value = String(value);
        }
    });
    if (markInitialized) {
        settingsInitializedFromState = true;
    }
}

function syncRealtimeControlsFromState(data) {
    if (!data) return;
    const config = data.training_config || data.runtime_control_defaults;
    if (!config) return;
    if (!settingsInitializedFromState) {
        populateRealtimeControls(config);
        setControlStatus("Runtime controls synced from daemon. Edit values and click Apply.", "neutral");
    }
}

function loadSaneDefaultsIntoControls() {
    const daemonDefaults = lastStateData?.runtime_control_defaults;
    populateRealtimeControls(daemonDefaults || SANE_RUNTIME_CONTROL_DEFAULTS);
    setControlStatus("Loaded sane default runtime controls. Click Apply to push them to the daemon.", "neutral");
}

function collectRealtimeControlUpdates() {
    const updates = {};
    const base = lastStateData?.training_config || {};

    document.querySelectorAll('[data-control-key]').forEach((input) => {
        const key = input.getAttribute('data-control-key');
        if (!key) return;

        let value;
        if (input.type === 'checkbox') {
            value = Boolean(input.checked);
        } else if (input.value === '') {
            return;
        } else if (input.type === 'number') {
            const n = Number(input.value);
            if (!Number.isFinite(n)) return;
            value = Number.isInteger(n) ? n : n;
        } else {
            value = input.value;
        }

        if (Object.prototype.hasOwnProperty.call(base, key) && base[key] === value) {
            return;
        }
        updates[key] = value;
    });

    return updates;
}

async function submitRealtimeControls(event) {
    event.preventDefault();

    const applyBtn = document.getElementById('applyControlsBtn');
    const updates = collectRealtimeControlUpdates();
    if (!updates || Object.keys(updates).length === 0) {
        setControlStatus("No runtime control changes to apply.", "neutral");
        return;
    }

    if (applyBtn) applyBtn.disabled = true;
    setControlStatus("Applying runtime controls to trainer daemon...", "neutral");

    try {
        const res = await fetch('/api/control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ updates }),
        });
        const payload = await res.json().catch(() => ({}));
        if (!res.ok || payload.ok === false) {
            const errText = payload?.errors
                ? Object.entries(payload.errors).map(([k, v]) => `${k}: ${v}`).join(' | ')
                : (payload?.message || `HTTP ${res.status}`);
            setControlStatus(`Control apply failed: ${errText}`, "error");
            if (payload?.training_config) {
                lastStateData = { ...(lastStateData || {}), training_config: payload.training_config };
            }
            return;
        }

        if (payload.training_config) {
            lastStateData = { ...(lastStateData || {}), training_config: payload.training_config };
            populateRealtimeControls(payload.training_config);
        }
        const warnings = Array.isArray(payload.warnings) && payload.warnings.length
            ? ` Warnings: ${payload.warnings.join(' ')}`
            : '';
        const appliedKeys = Object.keys(payload.applied || updates).join(', ');
        setControlStatus(`Applied: ${appliedKeys}.${warnings}`, payload.warnings?.length ? "neutral" : "success");

        if (lastStateData) {
            updateControlRuntimeSnapshot(lastStateData);
        }
    } catch (err) {
        console.error("Failed to apply realtime controls", err);
        setControlStatus("Control apply failed: daemon unavailable.", "error");
    } finally {
        if (applyBtn) applyBtn.disabled = false;
    }
}

function refreshRealtimeControlsFromDaemon() {
    if (lastStateData?.training_config) {
        populateRealtimeControls(lastStateData.training_config);
        setControlStatus("Controls refreshed from latest daemon state snapshot.", "neutral");
        return;
    }
    if (lastStateData?.runtime_control_defaults) {
        populateRealtimeControls(lastStateData.runtime_control_defaults);
        setControlStatus("Loaded daemon defaults while trainer is booting.", "neutral");
        return;
    }
    pollState();
}

function initRealtimeControls() {
    const form = document.getElementById('realtimeControlsForm');
    if (form) {
        form.addEventListener('submit', submitRealtimeControls);
    }
    const refreshBtn = document.getElementById('refreshControlsBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', refreshRealtimeControlsFromDaemon);
    }
    const saneDefaultsBtn = document.getElementById('saneDefaultsBtn');
    if (saneDefaultsBtn) {
        saneDefaultsBtn.addEventListener('click', loadSaneDefaultsIntoControls);
    }
    if (!settingsInitializedFromState) {
        populateRealtimeControls(SANE_RUNTIME_CONTROL_DEFAULTS, { markInitialized: false });
    }
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
initDrawthruPreviewChart();
initRealtimeControls();
pollState();
pollCache();
pollDrawthru();
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
        radar.innerHTML = '<div class="empty-state">No codec data available</div>';
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
