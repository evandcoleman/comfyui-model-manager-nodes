import { app } from "../../scripts/app.js";

// Maps node type -> folder key used by the backend
const NODE_FOLDER_MAP = {
    ModelManagerCheckpointLoader: "checkpoints",
    ModelManagerLoRALoader: "loras",
    ModelManagerMultiLoRALoader: "loras",
    ModelManagerVAELoader: "vae",
    ModelManagerClearCache: null,
    ModelManagerImageUpload: "checkpoints",
};

// Maps folder -> combo widget name on the node
const FOLDER_COMBO_MAP = {
    checkpoints: "ckpt_name",
    loras: "lora_name",
    vae: "vae_name",
};

// Node types whose combo widget name differs from FOLDER_COMBO_MAP
const NODE_COMBO_OVERRIDE = {
    ModelManagerImageUpload: "model_name",
};

const MODEL_MANAGER_NODE_TYPES = Object.keys(NODE_FOLDER_MAP);

const modelManagerState = {
    connected: false,
    api_url: null,
};

const trackedWidgets = new Set(); // all auth widgets

// ---------------------------------------------------------------------------
// Model cache — per-folder model data fetched from our API
// ---------------------------------------------------------------------------

const modelCache = {}; // folder -> { models: [...], values: [...], baseModels: [...] }

function modelComboValue(m) {
    const vid = m.versionId;
    return vid ? `${m.id}@${vid}:${m.name}` : `${m.id}:${m.name}`;
}

function getCacheValues(folder, baseModel) {
    const cache = modelCache[folder];
    if (!cache) return [];
    if (!baseModel || baseModel === "All") return cache.values;
    return cache.models
        .filter(m => m.baseModel === baseModel)
        .map(modelComboValue);
}

function getBaseModelOptions(folder) {
    const cache = modelCache[folder];
    if (!cache || cache.baseModels.length === 0) return ["All"];
    return ["All", ...cache.baseModels];
}

async function refreshModels(folder) {
    if (!modelManagerState.connected) return;
    try {
        const resp = await fetch(`/model-manager/models/${folder}`);
        const data = await resp.json();
        if (data.models) {
            const models = data.models;
            modelCache[folder] = {
                models,
                values: models.map(modelComboValue),
                baseModels: [...new Set(models.map(m => m.baseModel).filter(Boolean))].sort(),
            };
        }
    } catch (e) {
        console.warn(`Model Manager: failed to refresh ${folder}`, e);
    }
}

async function refreshAllModels() {
    await Promise.all(["checkpoints", "loras", "vae"].map(refreshModels));
    updateAllNodeCombos();
}

// Directly update combo widget values on all existing graph nodes
function updateAllNodeCombos() {
    if (!app.graph?._nodes) return;
    for (const node of app.graph._nodes) {
        updateNodeCombo(node);
    }
    app.graph.setDirtyCanvas(true, true);
}

function updateNodeCombo(node) {
    const folder = NODE_FOLDER_MAP[node.type];
    if (!folder) return;
    if (!modelCache[folder]) return;

    // Update base_model widget options (if present)
    const baseModelWidget = node.widgets?.find(w => w.name === "base_model");
    if (baseModelWidget) {
        baseModelWidget.options.values = getBaseModelOptions(folder);
    }

    // Multi-LoRA doesn't have a standard combo — its dropdowns are built on the fly
    if (node.type === "ModelManagerMultiLoRALoader") return;

    // Update model combo values
    const comboName = NODE_COMBO_OVERRIDE[node.type] || FOLDER_COMBO_MAP[folder];
    if (!comboName) return;

    const baseModel = baseModelWidget?.value;
    const values = getCacheValues(folder, baseModel);
    const comboValues = values.length > 0 ? values : ["(no models found)"];

    // Update the widget on this node (if it's still visible as a widget)
    const comboWidget = node.widgets?.find(w => w.name === comboName);
    if (comboWidget) {
        comboWidget.options = comboWidget.options || {};
        comboWidget.options.values = comboValues;
    }

    // Keep input.widget.config in sync so Primitive nodes get current values
    // when they connect (or are already connected).
    const input = (node.inputs || []).find(i => i.name === comboName);
    if (input?.widget?.config) {
        input.widget.config[0] = comboValues;
    }

    // Update any upstream node (Primitive/Reroute) already connected to this input
    if (input?.link != null) {
        updateUpstreamPrimitive(input.link, comboValues);
    }
}

// Walk upstream through a link chain (handles Reroute nodes) and update
// the originating Primitive node's combo widget with fresh values.
function updateUpstreamPrimitive(linkId, comboValues) {
    const link = app.graph.links?.[linkId];
    if (!link) return;
    const srcNode = app.graph.getNodeById(link.origin_id);
    if (!srcNode) return;

    // If this is a Reroute, walk further upstream
    if (srcNode.type === "Reroute") {
        const rInput = srcNode.inputs?.[0];
        if (rInput?.link != null) {
            updateUpstreamPrimitive(rInput.link, comboValues);
        }
        return;
    }

    // Update every combo-like widget on the source node (Primitive has one widget)
    for (const w of (srcNode.widgets || [])) {
        if (w.type === "combo" || (w.options && Array.isArray(w.options?.values))) {
            w.options = w.options || {};
            w.options.values = comboValues;
        }
    }
}

// ---------------------------------------------------------------------------
// Shared UI helpers
// ---------------------------------------------------------------------------

function makeOverlay() {
    const overlay = document.createElement("div");
    Object.assign(overlay.style, {
        position: "fixed",
        top: "0",
        left: "0",
        width: "100%",
        height: "100%",
        background: "rgba(0,0,0,0.5)",
        zIndex: "10000",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
    });
    return overlay;
}

function makeDialogBox(width = "400px") {
    const dialog = document.createElement("div");
    Object.assign(dialog.style, {
        background: "#2a2a2a",
        borderRadius: "8px",
        padding: "24px",
        minWidth: "320px",
        maxWidth: width,
        color: "#eee",
        fontFamily: "sans-serif",
        boxShadow: "0 4px 24px rgba(0,0,0,0.5)",
    });
    return dialog;
}

function makeTitle(text) {
    const title = document.createElement("h3");
    title.textContent = text;
    Object.assign(title.style, { margin: "0 0 16px 0", fontSize: "16px" });
    return title;
}

function makeErrorDiv() {
    const errorDiv = document.createElement("div");
    Object.assign(errorDiv.style, {
        color: "#ff6b6b",
        marginBottom: "12px",
        fontSize: "13px",
        display: "none",
    });
    return errorDiv;
}

function makeField(label, type, placeholder) {
    const container = document.createElement("div");
    container.style.marginBottom = "12px";
    const lbl = document.createElement("label");
    lbl.textContent = label;
    Object.assign(lbl.style, {
        display: "block",
        marginBottom: "4px",
        fontSize: "13px",
        color: "#aaa",
    });
    const input = document.createElement("input");
    input.type = type;
    input.placeholder = placeholder;
    Object.assign(input.style, {
        width: "100%",
        padding: "8px",
        border: "1px solid #555",
        borderRadius: "4px",
        background: "#1a1a1a",
        color: "#eee",
        fontSize: "14px",
        boxSizing: "border-box",
    });
    container.appendChild(lbl);
    container.appendChild(input);
    return { container, input };
}

function makeButton(text, primary) {
    const btn = document.createElement("button");
    btn.textContent = text;
    Object.assign(btn.style, {
        padding: "8px 16px",
        border: "none",
        borderRadius: "4px",
        cursor: "pointer",
        fontSize: "14px",
        background: primary ? "#4a9eff" : "#555",
        color: "#fff",
    });
    return btn;
}

function makeBtnRow() {
    const row = document.createElement("div");
    Object.assign(row.style, {
        display: "flex",
        justifyContent: "flex-end",
        gap: "8px",
        marginTop: "16px",
    });
    return row;
}

// ---------------------------------------------------------------------------
// State management
// ---------------------------------------------------------------------------

async function fetchStatus() {
    try {
        const resp = await fetch("/model-manager/status");
        const data = await resp.json();
        modelManagerState.connected = data.connected;
        modelManagerState.api_url = data.api_url;
    } catch (e) {
        console.warn("Model Manager: failed to fetch status", e);
    }
}

function updateAllWidgets() {
    for (const w of trackedWidgets) {
        if (w.onModelManagerStateChange) w.onModelManagerStateChange();
    }
    if (app.graph) app.graph.setDirtyCanvas(true, true);
}

// ---------------------------------------------------------------------------
// Connect dialog
// ---------------------------------------------------------------------------

function showConnectDialog() {
    const overlay = makeOverlay();
    const dialog = makeDialogBox();
    const errorDiv = makeErrorDiv();

    const apiUrl = makeField("API URL", "text", "http://localhost:3000");
    const apiKey = makeField("API Key", "password", "");

    if (modelManagerState.api_url) apiUrl.input.value = modelManagerState.api_url;

    const btnRow = makeBtnRow();
    const cancelBtn = makeButton("Cancel", false);
    const connectBtn = makeButton("Connect", true);

    cancelBtn.onclick = () => overlay.remove();
    overlay.onclick = (e) => { if (e.target === overlay) overlay.remove(); };

    connectBtn.onclick = async () => {
        const url = apiUrl.input.value.trim();
        const key = apiKey.input.value.trim();

        if (!url || !key) {
            errorDiv.textContent = "Both fields are required.";
            errorDiv.style.display = "block";
            return;
        }

        connectBtn.disabled = true;
        connectBtn.textContent = "Connecting...";
        errorDiv.style.display = "none";

        try {
            const resp = await fetch("/model-manager/connect", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ api_url: url, api_key: key }),
            });
            const data = await resp.json();

            if (!resp.ok) {
                errorDiv.textContent = data.error || "Connection failed";
                errorDiv.style.display = "block";
                connectBtn.disabled = false;
                connectBtn.textContent = "Connect";
                return;
            }

            modelManagerState.connected = true;
            modelManagerState.api_url = url;
            overlay.remove();
            updateAllWidgets();
            await refreshAllModels();
        } catch (e) {
            errorDiv.textContent = "Connection failed: " + e.message;
            errorDiv.style.display = "block";
            connectBtn.disabled = false;
            connectBtn.textContent = "Connect";
        }
    };

    btnRow.appendChild(cancelBtn);
    btnRow.appendChild(connectBtn);

    dialog.appendChild(makeTitle("Connect to Model Manager"));
    dialog.appendChild(errorDiv);
    dialog.appendChild(apiUrl.container);
    dialog.appendChild(apiKey.container);
    dialog.appendChild(btnRow);
    overlay.appendChild(dialog);
    document.body.appendChild(overlay);

    (apiUrl.input.value ? apiKey.input : apiUrl.input).focus();
}

// ---------------------------------------------------------------------------
// Disconnect
// ---------------------------------------------------------------------------

async function doDisconnect() {
    if (!confirm("Disconnect from Model Manager?")) return;

    try {
        await fetch("/model-manager/disconnect", { method: "POST" });
    } catch (e) {
        console.warn("Model Manager: disconnect request failed", e);
    }

    modelManagerState.connected = false;
    modelManagerState.api_url = null;
    // Clear caches
    for (const key of Object.keys(modelCache)) delete modelCache[key];
    updateAllWidgets();
    updateAllNodeCombos();
}

// ---------------------------------------------------------------------------
// Dynamic LoRA slot management (rgthree-style architecture)
// ---------------------------------------------------------------------------

// -- Canvas drawing helpers ------------------------------------------------

function roundRect(ctx, x, y, w, h, r) {
    if (ctx.roundRect) {
        ctx.beginPath();
        ctx.roundRect(x, y, w, h, r);
    } else {
        ctx.beginPath();
        ctx.rect(x, y, w, h);
    }
}

function drawSwitch(ctx, x, y, on, partial) {
    const W = SWITCH_W;
    const H = SWITCH_H;
    const R = H / 2;
    const knobR = R - 2;

    ctx.fillStyle = on ? "#4a9eff" : partial ? "rgba(74,158,255,0.35)" : "#555";
    roundRect(ctx, x, y, W, H, [R]);
    ctx.fill();

    const knobX = on || partial ? x + W - R : x + R;
    ctx.fillStyle = "#eee";
    ctx.beginPath();
    ctx.arc(knobX, y + R, knobR, 0, Math.PI * 2);
    ctx.fill();
}

function truncateText(ctx, text, maxWidth) {
    if (ctx.measureText(text).width <= maxWidth) return text;
    const ellipsis = "\u2026";
    while (text.length > 1 && ctx.measureText(text + ellipsis).width > maxWidth) {
        text = text.slice(0, -1);
    }
    return text + ellipsis;
}

// -- Zone layout constants -------------------------------------------------

const MARGIN = 15;
const SWITCH_W = 22;
const SWITCH_H = 12;
const SWITCH_PAD = 4;
const ARROW_W = 14;
const STRENGTH_NUM_W = 36;
const STRENGTH_COL_W = ARROW_W + STRENGTH_NUM_W + ARROW_W; // one full strength column

function loraZones(width, dual) {
    const switchX = MARGIN + SWITCH_PAD;
    const nameX = switchX + SWITCH_W + 8;

    if (dual) {
        // Two strength columns: | < model > | < clip > |
        const col2End = width - MARGIN;
        const col2ArrowRight = col2End - ARROW_W;
        const col2Num = col2ArrowRight - STRENGTH_NUM_W;
        const col2ArrowLeft = col2Num - ARROW_W;

        const col1End = col2ArrowLeft - 2; // small gap between columns
        const col1ArrowRight = col1End - ARROW_W;
        const col1Num = col1ArrowRight - STRENGTH_NUM_W;
        const col1ArrowLeft = col1Num - ARROW_W;

        const nameEnd = col1ArrowLeft - 4;
        return {
            toggle:       { x: MARGIN, w: nameX - MARGIN },
            name:         { x: nameX,  w: nameEnd - nameX },
            arrowLeft:    { x: col1ArrowLeft, w: ARROW_W },
            strengthNum:  { x: col1Num, w: STRENGTH_NUM_W },
            arrowRight:   { x: col1ArrowRight, w: ARROW_W },
            arrowLeft2:   { x: col2ArrowLeft, w: ARROW_W },
            strengthNum2: { x: col2Num, w: STRENGTH_NUM_W },
            arrowRight2:  { x: col2ArrowRight, w: ARROW_W },
        };
    }

    // Single mode — same as before
    const arrowRightEnd = width - MARGIN;
    const arrowRightX = arrowRightEnd - ARROW_W;
    const numEnd = arrowRightX;
    const numX = numEnd - STRENGTH_NUM_W;
    const arrowLeftEnd = numX;
    const arrowLeftX = arrowLeftEnd - ARROW_W;
    const nameEnd = arrowLeftX - 4;
    return {
        toggle:     { x: MARGIN, w: nameX - MARGIN },
        name:       { x: nameX,  w: nameEnd - nameX },
        arrowLeft:  { x: arrowLeftX, w: ARROW_W },
        strengthNum:{ x: numX, w: STRENGTH_NUM_W },
        arrowRight: { x: arrowRightX, w: ARROW_W },
    };
}

// -- Interaction helpers ---------------------------------------------------

function isDualMode(node) {
    return node.properties?.showStrengths === "Model & Clip";
}

function getFilteredLoraModels(node) {
    const cache = modelCache.loras;
    if (!cache) return [];
    const baseModel = node.widgets?.find(w => w.name === "base_model")?.value;
    if (!baseModel || baseModel === "All") return cache.models;
    return cache.models.filter(m => m.baseModel === baseModel);
}

function buildGroupedLoraMenu(models, onSelect) {
    // Group LoRA models by baseModel into nested submenus
    const rootItems = [];
    rootItems.push({ content: "None", callback: () => onSelect("None") });

    const groups = {};
    for (const m of models) {
        const group = m.baseModel || "Other";
        if (!groups[group]) groups[group] = [];
        groups[group].push(m);
    }

    const groupNames = Object.keys(groups).sort();
    if (groupNames.length <= 1) {
        // No grouping needed — flat list
        for (const m of models) {
            const value = modelComboValue(m);
            rootItems.push({
                content: m.name,
                callback: () => onSelect(value),
            });
        }
    } else {
        for (const group of groupNames) {
            const submenu = groups[group].map(m => {
                const value = modelComboValue(m);
                return {
                    content: m.name,
                    callback: () => onSelect(value),
                };
            });
            rootItems.push({
                content: group,
                has_submenu: true,
                submenu: { options: submenu },
            });
        }
    }

    return rootItems;
}

function showLoraDropdown(event, widget, node) {
    const models = getFilteredLoraModels(node);
    const items = buildGroupedLoraMenu(models, (v) => {
        widget.value = { ...widget.value, lora: v, on: v !== "None" };
        node.setDirtyCanvas(true);
    });
    new LiteGraph.ContextMenu(items, {
        event: event,
        scale: app.canvas.ds?.scale || 1,
    });
}

function showStrengthInput(event, widget, node, key = "strength") {
    const input = document.createElement("input");
    input.type = "number";
    input.value = (widget.value[key] ?? widget.value.strength).toFixed(2);
    input.step = "0.05";
    input.min = "-20";
    input.max = "20";
    Object.assign(input.style, {
        position: "fixed",
        left: (event.clientX - 40) + "px",
        top: (event.clientY - 12) + "px",
        width: "60px",
        padding: "4px 6px",
        border: "1px solid #4a9eff",
        borderRadius: "4px",
        background: "#1a1a1a",
        color: "#eee",
        fontSize: "13px",
        zIndex: "20000",
        textAlign: "center",
        outline: "none",
    });

    function apply() {
        const v = parseFloat(input.value);
        if (!isNaN(v)) {
            widget.value = { ...widget.value, [key]: Math.max(-20, Math.min(20, v)) };
        }
        input.remove();
        node.setDirtyCanvas(true);
    }

    input.onblur = apply;
    input.onkeydown = (e) => {
        if (e.key === "Enter") { e.preventDefault(); apply(); }
        if (e.key === "Escape") input.remove();
    };

    document.body.appendChild(input);
    input.focus();
    input.select();
}

// -- Widget factories ------------------------------------------------------

function getLoraWidgets(node) {
    return (node.widgets || []).filter(w => w.name.startsWith("lora_") && w._isLoraSlot);
}

function createLoraWidget(node, index, initialValue) {
    const val = initialValue || { on: true, lora: "None", strength: 1.0, strengthTwo: null };
    const cache = modelCache.loras;
    const comboValues = cache ? ["None", ...cache.values] : ["None"];
    // Create as combo (for addWidget initialization), then change type to prevent
    // LiteGraph's combo-specific processing from resetting our object value
    const w = node.addWidget("combo", `lora_${index}`, "None", null, { values: comboValues });
    w.type = "lora_slot";
    w.value = val;
    w.options.values = undefined; // prevent LiteGraph combo picker on dblclick
    w._isLoraSlot = true;
    w._lastStrengthClick = 0;
    w._lastStrengthClick2 = 0;
    w._dragState = null; // { key, startX, startVal }

    w.draw = function (ctx, _node, width, y, H) {
        this.last_y = y;
        const dual = isDualMode(node);
        const z = loraZones(width, dual);
        const switchX = MARGIN + SWITCH_PAD;
        const switchY = y + (H - SWITCH_H) / 2;
        const v = this.value;
        const on = v.on !== false;

        // background
        ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR || "#2a2a2a";
        ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR || "#666";
        roundRect(ctx, MARGIN, y, width - MARGIN * 2, H, [4]);
        ctx.fill();
        ctx.stroke();

        // toggle switch
        drawSwitch(ctx, switchX, switchY, on, false);

        const savedAlpha = ctx.globalAlpha;
        if (!on) ctx.globalAlpha = 0.4;

        // LoRA name — extract name after "id@vid:" or "id:" prefix
        const rawName = v.lora === "None" ? "None" : v.lora.replace(/^[^:]*:/, "");
        ctx.fillStyle = on ? (LiteGraph.WIDGET_TEXT_COLOR || "#ddd") : "#666";
        ctx.font = "12px sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(truncateText(ctx, rawName, z.name.w), z.name.x, y + H / 2);

        // Strength column 1 (model strength, or the only strength in single mode)
        const secColor = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR || "#999";
        ctx.fillStyle = secColor;
        ctx.font = "11px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("\u25C0", z.arrowLeft.x + z.arrowLeft.w / 2, y + H / 2);

        ctx.fillStyle = on ? secColor : "#555";
        ctx.font = "11px monospace";
        ctx.textAlign = "center";
        ctx.fillText(v.strength.toFixed(2), z.strengthNum.x + z.strengthNum.w / 2, y + H / 2);

        ctx.fillStyle = secColor;
        ctx.font = "11px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("\u25B6", z.arrowRight.x + z.arrowRight.w / 2, y + H / 2);

        // Strength column 2 (clip strength, dual mode only)
        if (dual && z.arrowLeft2) {
            const s2 = v.strengthTwo ?? v.strength;
            ctx.fillStyle = secColor;
            ctx.font = "11px sans-serif";
            ctx.textAlign = "center";
            ctx.fillText("\u25C0", z.arrowLeft2.x + z.arrowLeft2.w / 2, y + H / 2);

            ctx.fillStyle = on ? secColor : "#555";
            ctx.font = "11px monospace";
            ctx.textAlign = "center";
            ctx.fillText(s2.toFixed(2), z.strengthNum2.x + z.strengthNum2.w / 2, y + H / 2);

            ctx.fillStyle = secColor;
            ctx.font = "11px sans-serif";
            ctx.textAlign = "center";
            ctx.fillText("\u25B6", z.arrowRight2.x + z.arrowRight2.w / 2, y + H / 2);
        }

        ctx.globalAlpha = savedAlpha;
    };

    w.mouse = function (event, pos) {
        const t = event.type;
        const dual = isDualMode(node);
        const z = loraZones(node.size[0], dual);
        const x = pos[0];

        // --- Drag-to-change strength ---
        if (t === "pointerdown" || t === "mousedown") {
            // Check if pointer is on a strength number zone
            if (x >= z.strengthNum.x && x < z.strengthNum.x + z.strengthNum.w) {
                this._dragState = { key: "strength", startX: event.canvasX ?? event.clientX, startVal: this.value.strength };
            } else if (dual && z.strengthNum2 && x >= z.strengthNum2.x && x < z.strengthNum2.x + z.strengthNum2.w) {
                this._dragState = { key: "strengthTwo", startX: event.canvasX ?? event.clientX, startVal: this.value.strengthTwo ?? this.value.strength };
            } else {
                this._dragState = null;
            }
            return true;
        }

        if (t === "pointermove" || t === "mousemove") {
            if (this._dragState) {
                const dx = (event.canvasX ?? event.clientX) - this._dragState.startX;
                if (Math.abs(dx) > 2) {
                    const delta = dx * 0.01;
                    const newVal = Math.max(-20, Math.min(20, this._dragState.startVal + delta));
                    this.value = { ...this.value, [this._dragState.key]: Math.round(newVal * 100) / 100 };
                    this._dragState._dragged = true;
                    node.setDirtyCanvas(true);
                }
                return true;
            }
            return false;
        }

        const isUp = t === "pointerup" || t === "mouseup";
        const isDblClick = t === "dblclick";

        if (isUp && this._dragState?._dragged) {
            this._dragState = null;
            return true;
        }
        this._dragState = null;

        if (!isUp && !isDblClick) return false;

        // dblclick — handle strength zones
        if (isDblClick) {
            if (x >= z.strengthNum.x && x < z.strengthNum.x + z.strengthNum.w) {
                showStrengthInput(event, this, node, "strength");
                return true;
            }
            if (dual && z.strengthNum2 && x >= z.strengthNum2.x && x < z.strengthNum2.x + z.strengthNum2.w) {
                showStrengthInput(event, this, node, "strengthTwo");
                return true;
            }
            return false;
        }

        // toggle switch
        if (x < z.toggle.x + z.toggle.w) {
            this.value = { ...this.value, on: !this.value.on };
            node.setDirtyCanvas(true);
            return true;
        }

        // --- Strength column 1 ---
        if (x >= z.arrowLeft.x && x < z.arrowLeft.x + z.arrowLeft.w) {
            const s = Math.max(-20, this.value.strength - 0.05);
            this.value = { ...this.value, strength: Math.round(s * 100) / 100 };
            node.setDirtyCanvas(true);
            return true;
        }
        if (x >= z.strengthNum.x && x < z.strengthNum.x + z.strengthNum.w) {
            const now = Date.now();
            if (now - this._lastStrengthClick < 500) {
                this._lastStrengthClick = 0;
                showStrengthInput(event, this, node, "strength");
            } else {
                this._lastStrengthClick = now;
            }
            return true;
        }
        if (x >= z.arrowRight.x && x < z.arrowRight.x + z.arrowRight.w) {
            const s = Math.min(20, this.value.strength + 0.05);
            this.value = { ...this.value, strength: Math.round(s * 100) / 100 };
            node.setDirtyCanvas(true);
            return true;
        }

        // --- Strength column 2 (dual mode) ---
        if (dual && z.arrowLeft2) {
            if (x >= z.arrowLeft2.x && x < z.arrowLeft2.x + z.arrowLeft2.w) {
                const s2 = (this.value.strengthTwo ?? this.value.strength) - 0.05;
                this.value = { ...this.value, strengthTwo: Math.round(Math.max(-20, s2) * 100) / 100 };
                node.setDirtyCanvas(true);
                return true;
            }
            if (x >= z.strengthNum2.x && x < z.strengthNum2.x + z.strengthNum2.w) {
                const now = Date.now();
                if (now - this._lastStrengthClick2 < 500) {
                    this._lastStrengthClick2 = 0;
                    showStrengthInput(event, this, node, "strengthTwo");
                } else {
                    this._lastStrengthClick2 = now;
                }
                return true;
            }
            if (x >= z.arrowRight2.x && x < z.arrowRight2.x + z.arrowRight2.w) {
                const s2 = (this.value.strengthTwo ?? this.value.strength) + 0.05;
                this.value = { ...this.value, strengthTwo: Math.round(Math.min(20, s2) * 100) / 100 };
                node.setDirtyCanvas(true);
                return true;
            }
        }

        // name zone — open dropdown
        showLoraDropdown(event, this, node);
        return true;
    };

    w.serializeValue = function () {
        const v = { ...this.value };
        if (!isDualMode(node)) {
            delete v.strengthTwo;
        }
        return v;
    };

    return w;
}

function createToggleAllWidget(node) {
    const w = node.addWidget("combo", "toggle_all", "", null, { values: [] });
    w.type = "lora_toggle_all";
    w.serialize = false;
    w._isToggleAll = true;

    w.draw = function (ctx, _node, width, y, H) {
        this.last_y = y;
        const dual = isDualMode(node);
        const switchX = MARGIN + SWITCH_PAD;
        const switchY = y + (H - SWITCH_H) / 2;
        const z = loraZones(width, dual);
        const loraW = getLoraWidgets(node);
        const allOn = loraW.length > 0 && loraW.every(lw => lw.value.on !== false);
        const someOn = loraW.some(lw => lw.value.on !== false);

        ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR || "#2a2a2a";
        ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR || "#666";
        roundRect(ctx, MARGIN, y, width - MARGIN * 2, H, [4]);
        ctx.fill();
        ctx.stroke();

        drawSwitch(ctx, switchX, switchY, allOn, !allOn && someOn);

        ctx.fillStyle = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR || "#999";
        ctx.font = "11px sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText("Toggle All", z.name.x, y + H / 2);

        // Strength column header labels
        ctx.font = "10px sans-serif";
        ctx.textAlign = "center";
        ctx.fillStyle = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR || "#999";
        if (dual && z.strengthNum2) {
            ctx.fillText("Model", z.strengthNum.x + z.strengthNum.w / 2, y + H / 2);
            ctx.fillText("Clip", z.strengthNum2.x + z.strengthNum2.w / 2, y + H / 2);
        } else {
            ctx.fillText("Strength", z.strengthNum.x + z.strengthNum.w / 2, y + H / 2);
        }
    };

    w.mouse = function (event) {
        const t = event.type;
        if (t === "pointerdown" || t === "mousedown") return true;
        if (t !== "pointerup" && t !== "mouseup") return false;
        const loraW = getLoraWidgets(node);
        const allOn = loraW.length > 0 && loraW.every(lw => lw.value.on !== false);
        const newState = !allOn;
        for (const lw of loraW) {
            lw.value = { ...lw.value, on: newState };
        }
        node.setDirtyCanvas(true);
        return true;
    };

    return w;
}

function createAddLoraButton(node) {
    const btn = node.addWidget("button", "add_lora", "Add LoRA", (_, __, ___, event) => {
        const models = getFilteredLoraModels(node);
        if (models.length === 0) {
            // No LoRAs available — add a blank slot
            addLoraSlot(node, btn, "None");
            return;
        }
        const items = buildGroupedLoraMenu(models, (v) => {
            addLoraSlot(node, btn, v);
        });
        new LiteGraph.ContextMenu(items, {
            event: event,
            scale: app.canvas.ds?.scale || 1,
        });
    });
    btn.serialize = false;
    btn._isAddLora = true;
    return btn;
}

function addLoraSlot(node, btn, loraName) {
    const loraW = getLoraWidgets(node);
    const newIndex = loraW.length + 1;
    const dual = isDualMode(node);
    const newW = createLoraWidget(node, newIndex, {
        on: loraName !== "None",
        lora: loraName,
        strength: 1.0,
        strengthTwo: dual ? 1.0 : null,
    });
    // Move new widget before the Add LoRA button
    const widgets = node.widgets;
    const newWIdx = widgets.indexOf(newW);
    const btnIdx = widgets.indexOf(btn);
    if (newWIdx > btnIdx) {
        widgets.splice(newWIdx, 1);
        widgets.splice(btnIdx, 0, newW);
    }
    fitNodeHeight(node);
    node.setDirtyCanvas(true);
}

// -- Renumber lora widget names to keep them sequential --------------------

function fitNodeHeight(node) {
    const computed = node.computeSize();
    node.setSize([node.size[0], computed[1]]);
}

function renumberLoraWidgets(node) {
    const loraW = getLoraWidgets(node);
    for (let i = 0; i < loraW.length; i++) {
        loraW[i].name = `lora_${i + 1}`;
    }
}

// -- Base model filter widget for LoRA nodes --------------------------------

function addBaseModelWidget(node) {
    const w = node.addWidget("combo", "base_model", "All", () => {
        updateNodeCombo(node);
        node.setDirtyCanvas(true);
    }, { values: getBaseModelOptions("loras") });
    w.serialize = false;
    return w;
}

// -- Setup & lifecycle -----------------------------------------------------

function setupLoraNode(node) {
    // Initialize the showStrengths property
    if (!node.properties) node.properties = {};
    if (!node.properties.showStrengths) node.properties.showStrengths = "Single";

    // Remove any auto-created optional widgets/inputs from FlexibleOptionalInputType
    node.widgets = (node.widgets || []).filter(w =>
        w.name === "model" || w.name === "clip" ||
        w.name === "mm_auth" || w.name === "mm_refresh"
    );
    if (node.inputs) {
        node.inputs = node.inputs.filter(input =>
            input.name === "model" || input.name === "clip"
        );
    }

    // Base model filter
    addBaseModelWidget(node);

    // Create Toggle All
    createToggleAllWidget(node);

    // Create one default LoRA slot
    createLoraWidget(node, 1, { on: true, lora: "None", strength: 1.0, strengthTwo: null });

    // Create Add LoRA button
    createAddLoraButton(node);

    fitNodeHeight(node);

    // Handle property changes (mode transitions)
    const origOnPropertyChanged = node.onPropertyChanged;
    node.onPropertyChanged = function (name, value, prevValue) {
        if (origOnPropertyChanged) origOnPropertyChanged.call(this, name, value, prevValue);
        if (name === "showStrengths") {
            const loraW = getLoraWidgets(node);
            if (value === "Model & Clip") {
                // Switching to dual: set strengthTwo = strength on all widgets
                for (const lw of loraW) {
                    if (lw.value.strengthTwo == null) {
                        lw.value = { ...lw.value, strengthTwo: lw.value.strength };
                    }
                }
            } else {
                // Switching to single: clear strengthTwo
                for (const lw of loraW) {
                    lw.value = { ...lw.value, strengthTwo: null };
                }
            }
            node.setDirtyCanvas(true);
        }
    };

    // Workflow load — restore slots from saved widget values
    const origConfigure = node.configure;
    node.configure = function (info) {
        // Strip dynamic LoRA widgets, stash trailing widgets (auth/refresh) for re-appending
        const trailingWidgets = [];
        node.widgets = (node.widgets || []).filter(w => {
            if (w._isLoraSlot || w._isToggleAll || w._isAddLora) return false;
            if (w.name === "mm_auth" || w.name === "mm_refresh") {
                trailingWidgets.push(w);
                return false;
            }
            return true;
        });
        if (node.inputs) {
            node.inputs = node.inputs.filter(input =>
                input.name === "model" || input.name === "clip"
            );
        }

        origConfigure?.call(this, info);

        // Restore showStrengths property
        if (info.properties?.showStrengths) {
            node.properties.showStrengths = info.properties.showStrengths;
        }

        // Rebuild from saved widgets_values
        const savedValues = info.widgets_values || [];
        const loraSlotValues = savedValues.filter(
            v => v && typeof v === "object" && "lora" in v
        );

        createToggleAllWidget(node);

        if (loraSlotValues.length > 0) {
            for (let i = 0; i < loraSlotValues.length; i++) {
                const sv = loraSlotValues[i];
                createLoraWidget(node, i + 1, {
                    on: sv.on !== false,
                    lora: sv.lora || "None",
                    strength: typeof sv.strength === "number" ? sv.strength : 1.0,
                    strengthTwo: sv.strengthTwo ?? null,
                });
            }
        } else {
            createLoraWidget(node, 1, { on: true, lora: "None", strength: 1.0, strengthTwo: null });
        }

        createAddLoraButton(node);

        // Re-append trailing widgets in correct order
        node.widgets.push(...trailingWidgets);

        fitNodeHeight(node);
    };

    // Right-click context menu
    const origGetExtraMenuOptions = node.getExtraMenuOptions;
    node.getExtraMenuOptions = function (canvas, options) {
        if (origGetExtraMenuOptions) origGetExtraMenuOptions.call(this, canvas, options);

        const nodeY = canvas.graph_mouse[1] - node.pos[1];
        const widgetH = LiteGraph.NODE_WIDGET_HEIGHT || 20;

        const loraW = getLoraWidgets(node);
        let slotIdx = -1;
        for (let i = 0; i < loraW.length; i++) {
            const ly = loraW[i].last_y;
            if (ly == null) continue;
            if (nodeY >= ly && nodeY < ly + widgetH) { slotIdx = i; break; }
        }
        if (slotIdx < 0) return;

        const menuItems = [];

        // Toggle On/Off
        const isOn = loraW[slotIdx].value.on !== false;
        menuItems.push({
            content: isOn ? "Disable LoRA" : "Enable LoRA",
            callback: () => {
                loraW[slotIdx].value = { ...loraW[slotIdx].value, on: !isOn };
                node.setDirtyCanvas(true);
            },
        });

        if (slotIdx > 0) {
            menuItems.push({
                content: "Move LoRA Up",
                callback: () => {
                    const widgets = node.widgets;
                    const aIdx = widgets.indexOf(loraW[slotIdx - 1]);
                    const bIdx = widgets.indexOf(loraW[slotIdx]);
                    [widgets[aIdx], widgets[bIdx]] = [widgets[bIdx], widgets[aIdx]];
                    renumberLoraWidgets(node);
                    node.setDirtyCanvas(true);
                },
            });
        }

        if (slotIdx < loraW.length - 1) {
            menuItems.push({
                content: "Move LoRA Down",
                callback: () => {
                    const widgets = node.widgets;
                    const aIdx = widgets.indexOf(loraW[slotIdx]);
                    const bIdx = widgets.indexOf(loraW[slotIdx + 1]);
                    [widgets[aIdx], widgets[bIdx]] = [widgets[bIdx], widgets[aIdx]];
                    renumberLoraWidgets(node);
                    node.setDirtyCanvas(true);
                },
            });
        }

        menuItems.push({
            content: "Remove LoRA",
            callback: () => {
                const idx = node.widgets.indexOf(loraW[slotIdx]);
                if (idx >= 0) node.widgets.splice(idx, 1);
                renumberLoraWidgets(node);
                fitNodeHeight(node);
                node.setDirtyCanvas(true);
            },
        });

        options.unshift(...menuItems, null);
    };
}

// ---------------------------------------------------------------------------
// Extension registration
// ---------------------------------------------------------------------------

app.registerExtension({
    name: "Comfy.ModelManager",

    async setup() {
        await fetchStatus();
        if (modelManagerState.connected) {
            await refreshAllModels();
        }
    },

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!MODEL_MANAGER_NODE_TYPES.includes(nodeData.name)) return;

        const folderKey = NODE_FOLDER_MAP[nodeData.name];
        const isMultiLoraNode = nodeData.name === "ModelManagerMultiLoRALoader";
        const isSingleLoraNode = nodeData.name === "ModelManagerLoRALoader";
        const isUploadNode = nodeData.name === "ModelManagerImageUpload";

        // Register the showStrengths property for the multi-LoRA node
        if (isMultiLoraNode) {
            nodeType.prototype.properties = {
                ...(nodeType.prototype.properties || {}),
                showStrengths: "Single",
            };
            // Expose as a combo widget in the Properties panel
            const propWidgets = nodeType.prototype.widgets_info || {};
            propWidgets["showStrengths"] = { widget: "combo", values: ["Single", "Model & Clip"] };
            nodeType.prototype.widgets_info = propWidgets;
        }

        const origOnCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            if (origOnCreated) origOnCreated.apply(this, arguments);

            const node = this;

            // --- Dynamic LoRA slot management (multi-LoRA) ---
            if (isMultiLoraNode) {
                setupLoraNode(node);

                // API JSON loading workaround: ComfyUI's API format doesn't call
                // configure(), so we detect & restore from the raw widgets array
                setTimeout(() => {
                    const hasLoraSlots = (node.widgets || []).some(w => w._isLoraSlot);
                    if (!hasLoraSlots) return; // already configured properly
                    // Check if any lora slot has a string value (API JSON format)
                    const apiSlots = (node.widgets || []).filter(
                        w => w._isLoraSlot && typeof w.value === "string"
                    );
                    if (apiSlots.length > 0) {
                        // Re-run setup — the API format set string values instead of objects
                        for (const aw of apiSlots) {
                            const name = aw.value;
                            aw.value = {
                                on: name !== "None",
                                lora: name,
                                strength: 1.0,
                                strengthTwo: isDualMode(node) ? 1.0 : null,
                            };
                        }
                        node.setDirtyCanvas(true);
                    }
                }, 16);
            }

            // --- Base model filter for single LoRA ---
            if (isSingleLoraNode) {
                const loraWidget = node.widgets?.find(w => w.name === "lora_name");
                if (loraWidget) {
                    const bmWidget = addBaseModelWidget(node);
                    // Move base_model before lora_name
                    const widgets = node.widgets;
                    const bmIdx = widgets.indexOf(bmWidget);
                    const lnIdx = widgets.indexOf(loraWidget);
                    if (bmIdx > lnIdx) {
                        widgets.splice(bmIdx, 1);
                        widgets.splice(lnIdx, 0, bmWidget);
                    }
                    // Initialize combo values from cache
                    updateNodeCombo(node);
                }
            }

            // --- Upload Image: force wildcard type on metadata inputs ---
            if (isUploadNode) {
                // Ensure LiteGraph sees these as "*" so combo outputs can connect
                for (const input of (node.inputs || [])) {
                    if (input.name === "sampler" || input.name === "scheduler") {
                        input.type = "*";
                    }
                }
            }

            // --- Auth button ---
            const authWidget = node.addWidget("button", "mm_auth", null, () => {
                if (modelManagerState.connected) {
                    doDisconnect();
                } else {
                    showConnectDialog();
                }
            });
            authWidget.serialize = false;

            authWidget.onModelManagerStateChange = () => {
                authWidget.name = modelManagerState.connected
                    ? "Model Manager: Connected"
                    : "Connect to Model Manager";
            };
            authWidget.onModelManagerStateChange();
            trackedWidgets.add(authWidget);

            // --- Refresh models button ---
            const refreshWidget = node.addWidget("button", "mm_refresh", "Refresh Models", async () => {
                if (!modelManagerState.connected) return;
                refreshWidget.name = "Refreshing...";
                node.setDirtyCanvas(true);
                try {
                    await fetch("/model-manager/refresh-models", { method: "POST" });
                    await refreshAllModels();
                } catch (e) {
                    console.warn("Model Manager: refresh failed", e);
                }
                refreshWidget.name = "Refresh Models";
                node.setDirtyCanvas(true);
            });
            refreshWidget.serialize = false;

            // --- Cleanup ---
            const origOnRemoved = node.onRemoved;
            node.onRemoved = function () {
                trackedWidgets.delete(authWidget);
                if (origOnRemoved) origOnRemoved.apply(this, arguments);
            };
        };
    },
});
