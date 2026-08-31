/**
 * H3 Modality LoRA Loader - dynamic slot UI (PlagueKind-style).
 *
 * Each row holds one LoRA plus its strength only; the audio/video/text
 * strengths remain the global node inputs above the slot list. Row state
 * is serialized into the hidden ``stack_data`` widget so workflows stay
 * portable, and the LoRA list is refreshed live from the server.
 */
import { app } from "../../scripts/app.js";

const NODE_TYPE = "H3ModalityLoraLoader";
const MAX_SLOTS = 10;
const MIN_SIZE = [420, 120];
let _loraCache = ["None"];
// Every on-canvas node instance registers a refresh callback here so its
// dropdown/warning state can be updated live, without needing a new node
// to be created or the page to be reloaded.
const _liveInstances = new Set();

function _notifyLiveInstances() {
    for (const refresh of _liveInstances) {
        try { refresh(); } catch (e) { console.warn("H3 LoRA instance refresh failed", e); }
    }
}

async function getLoraList(nodeData) {
    try {
        const list = nodeData?.input?.hidden?.available_loras?.[0];
        if (Array.isArray(list)) {
            _loraCache = ["None", ...list];
            _notifyLiveInstances();
        }
    } catch (e) {
        console.warn("LoRA fetch failed", e);
    }
}

function loraBasename(p) {
    const clean = p ? String(p).replace(/\\/g, "/") : "";
    return clean ? clean.split("/").pop() : "";
}

function loraFolder(p) {
    const clean = p ? String(p).replace(/\\/g, "/") : "";
    return clean && clean.includes("/") ? clean.substring(0, clean.lastIndexOf("/")) : "";
}

function loraDisplayName(p, allSlots) {
    const base = loraBasename(p);
    if (!base) return "None";
    const others = allSlots ? allSlots.filter(s => {
        const l = s.getLora ? s.getLora() : null;
        return l && l !== p && loraBasename(l) === base;
    }) : [];
    return others.length > 0 ? loraFolder(p).split("/").pop() + "/" + base : base;
}

app.registerExtension({
    name: "ComfyUI.H3ModalityLoraLoader.DynamicSlotUI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;
        await getLoraList(nodeData);

        const orig = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            orig?.apply(this, arguments);

            const node = this;
            const baseHeight = node.size[1];
            const BTN_H = 40;

            function minHeight() {
                return Math.max(MIN_SIZE[1], baseHeight + BTN_H);
            }

            node.onResize = function (size) {
                size[0] = Math.max(MIN_SIZE[0], size[0]);
                size[1] = Math.max(minHeight(), size[1]);
                this.size = size;
            };

            node.properties = node.properties || {};

            const stackIndex = node.widgets.findIndex(w => w.name === "stack_data");
            let stackWidget = stackIndex !== -1 ? node.widgets[stackIndex] : null;
            if (stackWidget) {
                stackWidget.computeSize = () => [0, -4];
                stackWidget.draw = () => {};
            }

            let initialData = [];
            try {
                const raw = node.properties["stack_data"] || stackWidget?.value || "[]";
                initialData = JSON.parse(raw);
            } catch {}

            const inputStyle = "background:var(--comfy-input-bg);color:var(--fg-color);border:1px solid var(--border-color);border-radius:4px;padding:0 6px;font-size:12px;cursor:pointer;height:26px;";

            const container = document.createElement("div");
            Object.assign(container.style, {
                display: "flex",
                flexDirection: "column",
                gap: "4px",
                width: "100%",
                padding: "4px",
                position: "relative",
                boxSizing: "border-box",
                fontFamily: "var(--font)",
                color: "var(--fg-color)"
            });

            let slots = [];
            let _rafPending = false;
            function syncSize() {
                if (_rafPending) return;
                _rafPending = true;
                requestAnimationFrame(() => {
                    _rafPending = false;
                    const targetH = baseHeight + container.scrollHeight + 12;
                    node.setSize([Math.max(node.size[0], MIN_SIZE[0]), Math.max(targetH, minHeight())]);
                });
            }

            function refreshAllDisplayNames() {
                slots.forEach(s => s.refreshDisplayName?.());
            }

            function refreshLoraState() {
                for (const s of slots) {
                    s.checkMissing?.();
                    s.refreshDisplayName?.();
                }
            }
            _liveInstances.add(refreshLoraState);
            const origOnRemoved = node.onRemoved;
            node.onRemoved = function () {
                _liveInstances.delete(refreshLoraState);
                origOnRemoved?.apply(this, arguments);
            };

            function syncData() {
                const data = slots.map(s => s.getValue());
                const json = JSON.stringify(data);
                if (stackWidget) stackWidget.value = json;
                node.properties["stack_data"] = json;
                node.onPropertyChanged?.("stack_data", json);
                refreshAllDisplayNames();
                syncSize();
            }

            async function refreshCache() {
                try {
                    const response = await fetch("/object_info/" + NODE_TYPE, { cache: "no-store" });
                    if (!response.ok) return;
                    const info = await response.json();
                    const refreshed = info?.[NODE_TYPE]?.input?.hidden?.available_loras?.[0]
                        || info?.input?.hidden?.available_loras?.[0];
                    if (Array.isArray(refreshed) && refreshed.length) {
                        _loraCache = ["None", ...refreshed];
                        slots.forEach(s => s.checkMissing?.());
                        _notifyLiveInstances();
                    }
                } catch (e) {
                    console.error(e);
                }
            }

            function sortTree(items) {
                items.sort((a, b) => {
                    const af = a.has_submenu ? 1 : 0;
                    const bf = b.has_submenu ? 1 : 0;
                    if (af !== bf) return af - bf;
                    return a.content.localeCompare(b.content, undefined, { numeric: true, sensitivity: "base" });
                });
                items.forEach(item => {
                    if (item.has_submenu && item.submenu?.options) {
                        sortTree(item.submenu.options);
                    }
                });
            }

            function buildMenuTree(list, onSelect) {
                const tree = [];
                const folders = new Map();

                for (const item of list) {
                    if (item === "None") {
                        tree.push({ content: "None", callback: () => onSelect("None") });
                        continue;
                    }

                    const clean = item.replace(/\\/g, "/");
                    const parts = clean.split("/");
                    let current = tree;
                    for (let i = 0; i < parts.length - 1; i++) {
                        const part = parts[i];
                        const folderKey = parts.slice(0, i + 1).join("/");
                        if (!folders.has(folderKey)) {
                            const existing = current.find(x => x.content === "📁 " + part);
                            if (!existing) {
                                existing = { content: "📁 " + part, has_submenu: true, submenu: { options: [] } };
                                current.push(existing);
                            }
                            folders.set(folderKey, existing);
                        }
                        current = folders.get(folderKey).submenu.options;
                    }

                    current.push({ content: parts[parts.length - 1], callback: () => onSelect(item) });
                }

                sortTree(tree);
                return tree;
            }

            function openLoraMenu(e, onSelect) {
                const list = _loraCache;
                const searchIndex = [];
                let noneAdded = false;
                for (const item of list) {
                    if (item === "None") {
                        if (!noneAdded) {
                            searchIndex.push({ display: "None", fullPath: "None", isFolder: false });
                            noneAdded = true;
                        }
                        continue;
                    }
                    const parts = item.split(/[/\\]/);
                    for (let i = 1; i < parts.length; i++) {
                        const folderPath = parts.slice(0, i).join("/");
                        if (!searchIndex.find(x => x.display === "📁 " + folderPath)) {
                            searchIndex.push({ display: "📁 " + folderPath, fullPath: null, isFolder: true, folderPrefix: folderPath });
                        }
                    }
                    searchIndex.push({ display: item.split(/[/\\]/).pop(), fullPath: item, isFolder: false });
                }
                const tree = buildMenuTree(list, onSelect);
                const menu = new LiteGraph.ContextMenu(tree, { event: e, scale: 1.2 });

                requestAnimationFrame(() => {
                    const root = menu?.root;
                    if (!root) return;
                    const header = document.createElement("div");
                    header.style.cssText = "display:flex;justify-content:space-between;align-items:center;padding:5px 8px;border-bottom:1px solid #444;";
                    const box = document.createElement("input");
                    box.placeholder = "Search LoRA...";
                    box.style.cssText = `flex:1;padding:4px;background:#222;color:white;border:1px solid #444;border-radius:4px;font-size:12px;`;
                    const refreshBtn = document.createElement("button");
                    refreshBtn.innerHTML = "🔄";
                    refreshBtn.title = "Refresh LoRA Cache";
                    refreshBtn.style.cssText = "margin-left:6px;padding:2px 6px;background:#333;border:none;border-radius:3px;cursor:pointer;";
                    refreshBtn.onclick = (ev) => {
                        ev.stopPropagation();
                        refreshCache();
                        menu.close?.();
                    };
                    header.append(box, refreshBtn);
                    root.prepend(header);
                    const flatList = document.createElement("div");
                    flatList.style.cssText = "display:none;max-height:320px;overflow-y:auto;min-width:320px;width:100%;box-sizing:border-box;";
                    root.appendChild(flatList);
                    const treeEntries = Array.from(root.querySelectorAll(".litemenu-entry"));
                    let currentFolderFilter = null;
                    function renderSearchResults(q) {
                        flatList.innerHTML = "";
                        const matches = searchIndex.filter(entry => {
                            if (currentFolderFilter && !entry.isFolder) {
                                return entry.fullPath && entry.fullPath.startsWith(currentFolderFilter + "/");
                            }
                            return entry.display.toLowerCase().includes(q);
                        });
                        for (const entry of matches) {
                            const el = document.createElement("div");
                            el.className = "litemenu-entry";
                            el.textContent = entry.display;
                            el.style.cssText = "padding:4px 8px;cursor:pointer;font-size:12px;white-space:nowrap;width:100%;box-sizing:border-box;";
                            if (entry.isFolder) {
                                el.style.fontStyle = "italic";
                                el.addEventListener("mousedown", (ev) => {
                                    ev.preventDefault();
                                    ev.stopPropagation();
                                    currentFolderFilter = entry.folderPrefix;
                                    box.value = entry.folderPrefix + "/";
                                    renderSearchResults(box.value.toLowerCase().trim());
                                });
                            } else {
                                el.addEventListener("mousedown", (ev) => {
                                    ev.preventDefault();
                                    ev.stopPropagation();
                                    onSelect(entry.fullPath);
                                    menu.close?.();
                                });
                            }
                            flatList.appendChild(el);
                        }
                    }
                    box.addEventListener("input", () => {
                        const q = box.value.toLowerCase().trim();
                        treeEntries.forEach(el => el.style.display = q ? "none" : "");
                        flatList.style.display = q ? "block" : "none";
                        renderSearchResults(q);
                    });
                    box.focus();
                });
            }

            let dropIndicator = null;
            let dragRow = null;

            // Rows a drop can land between, in visual order and without the row
            // being dragged, so the index below indexes this exact list.
            function dropRows() {
                return slots.map(s => s.row).filter(r => r !== dragRow);
            }

            function dropIndex(rows, y) {
                for (const [i, el] of rows.entries()) {
                    const box = el.getBoundingClientRect();
                    if (y < box.top + box.height / 2) return i;
                }
                return rows.length;
            }

            function showDropIndicator(rows, index) {
                if (!dropIndicator) {
                    dropIndicator = document.createElement("div");
                    dropIndicator.style.cssText = "position:absolute;height:3px;background:var(--primary-color);border-radius:2px;box-shadow:0 0 4px var(--primary-color);pointer-events:none;z-index:10;";
                    container.appendChild(dropIndicator);
                }
                // An insert at the end of the list draws under the last row; the
                // coordinates are container-relative or the line lands offscreen.
                let edge;
                if (rows[index]) edge = rows[index].getBoundingClientRect().top;
                else if (rows.length) edge = rows[rows.length - 1].getBoundingClientRect().bottom;
                else edge = addBtn.getBoundingClientRect().top;
                dropIndicator.style.top = `${edge - container.getBoundingClientRect().top - 2}px`;
                dropIndicator.style.left = "4px";
                dropIndicator.style.right = "4px";
            }

            function hideDropIndicator() {
                if (dropIndicator) {
                    dropIndicator.remove();
                    dropIndicator = null;
                }
            }

            // One dragover/drop pair on the container instead of one per row: the
            // rows sit in a flex gap, and per-row handlers go quiet exactly where
            // the user aims, leaving no indicator and the previous one stale.
            container.addEventListener("dragover", e => {
                if (!dragRow) return;
                e.preventDefault();
                e.stopPropagation();
                e.dataTransfer.dropEffect = "move";
                const rows = dropRows();
                showDropIndicator(rows, dropIndex(rows, e.clientY));
            });

            container.addEventListener("drop", e => {
                if (!dragRow) return;
                e.preventDefault();
                e.stopPropagation();
                const row = dragRow;
                const from = slots.findIndex(s => s.row === row);
                if (from === -1) return;
                const rows = dropRows();
                const index = dropIndex(rows, e.clientY);
                // ``slots`` is the order written to stack_data, so it is reordered
                // together with the DOM instead of being left behind by the move.
                const [slot] = slots.splice(from, 1);
                slots.splice(index, 0, slot);
                container.insertBefore(row, rows[index] ?? addBtn);
                hideDropIndicator();
                syncData();
            });

            function makeDraggable(row) {
                row.draggable = false;
                row.addEventListener("dragstart", e => {
                    e.stopPropagation();
                    row.style.opacity = "0.5";
                    dragRow = row;
                    e.dataTransfer.effectAllowed = "move";
                    e.dataTransfer.setData("text/plain", "");
                });
                // The mouse is released over the drop target, never over the
                // handle, so the flag has to be cleared here or the row keeps
                // hijacking every later click inside its own inputs.
                row.addEventListener("dragend", e => {
                    e.stopPropagation();
                    row.draggable = false;
                    row.style.opacity = "1";
                    dragRow = null;
                    hideDropIndicator();
                });
            }

            function addSlot(data = { on: true, lora: "None", str: 1.0 }) {
                if (slots.length >= MAX_SLOTS) return;
                // Backfill fields missing from rows saved before the field existed.
                data = { on: true, lora: "None", str: 1.0, ...data };

                const row = document.createElement("div");
                row.style.cssText = "display:flex;align-items:center;gap:6px;width:100%;min-height:28px;background:var(--comfy-menu-bg);padding:4px;border-radius:4px;border:1px solid var(--border-color);transition:all 0.15s ease;box-sizing:border-box;";

                const handle = document.createElement("div");
                handle.textContent = "⋮";
                handle.style.cssText = "color:#777;font-size:18px;cursor:grab;padding:0;user-select:none;width:10px;text-align:center;flex-shrink:0;";

                handle.addEventListener("mouseenter", () => {
                    row.style.background = "var(--comfy-input-bg)";
                    row.style.borderColor = "var(--primary-color)";
                });
                handle.addEventListener("mouseleave", () => {
                    row.style.background = "var(--comfy-menu-bg)";
                    row.style.borderColor = "var(--border-color)";
                });
                handle.addEventListener("mousedown", () => { row.draggable = true; });
                handle.addEventListener("mouseup", () => { row.draggable = false; });

                const chk = document.createElement("input");
                chk.type = "checkbox";
                chk.checked = data.on;
                chk.style.flexShrink = "0";

                chk.onchange = () => {
                    syncData();
                };

                const sel = document.createElement("div");
                sel.setAttribute("role", "button");
                sel.dataset.lora = data.lora;
                sel.style.cssText = inputStyle + "flex-grow:1;min-width:0;width:0;flex-shrink:1;display:flex;align-items:center;justify-content:space-between;cursor:pointer;overflow:hidden;user-select:none;white-space:nowrap;";

                sel.addEventListener("mouseenter", () => {
                    sel.style.background = "var(--comfy-menu-bg)";
                    sel.style.borderColor = "var(--primary-color)";
                });
                sel.addEventListener("mouseleave", () => {
                    sel.style.background = "var(--comfy-input-bg)";
                    checkMissing();
                });

                const selText = document.createElement("span");
                selText.textContent = loraDisplayName(data.lora, slots);
                selText.title = data.lora;
                selText.style.cssText = "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex-grow:1;text-align:left;min-width:0;margin-right:4px;";

                const warn = document.createElement("span");
                warn.textContent = "⚠";
                warn.style.cssText = "color:var(--error-text, #ff5555);font-size:11px;margin-right:2px;flex-shrink:0;display:none;font-weight:bold;";

                function checkMissing() {
                    const fullPath = sel.dataset.lora;
                    if (!fullPath || fullPath === "None") {
                        warn.style.display = "none";
                        sel.style.borderColor = "var(--border-color)";
                        return;
                    }
                    const norm = p => p ? p.replace(/[/\\]/g, "/").toLowerCase() : "";
                    const isMissing = !_loraCache.some(x => norm(x) === norm(fullPath));
                    if (isMissing) {
                        warn.style.display = "inline";
                        warn.title = `File missing from environment:\n${fullPath}`;
                        sel.style.borderColor = "var(--error-text, #ff5555)";
                    } else {
                        warn.style.display = "none";
                        sel.style.borderColor = "var(--border-color)";
                    }
                }

                const arrow = document.createElement("span");
                arrow.textContent = "▼";
                arrow.style.cssText = "flex-shrink:0;font-size:8px;opacity:0.6;";

                sel.append(warn, selText, arrow);
                sel.onclick = (e) => {
                    openLoraMenu(e, (fullPath) => {
                        sel.dataset.lora = fullPath;
                        selText.textContent = loraDisplayName(fullPath, slots);
                        selText.title = fullPath;
                        checkMissing();
                        syncData();
                    });
                };

                function showNumPopup(currentVal, label, onConfirm) {
                    const overlay = document.createElement("div");
                    overlay.style.cssText = "position:fixed;inset:0;background:rgba(0,0,0,0.55);z-index:9999;display:flex;align-items:center;justify-content:center;";
                    const panel = document.createElement("div");
                    panel.style.cssText = "background:var(--comfy-menu-bg);border:1px solid var(--border-color);border-radius:8px;padding:16px;display:flex;flex-direction:column;gap:10px;min-width:200px;box-shadow:0 4px 24px rgba(0,0,0,0.5);";
                    const title = document.createElement("div");
                    title.textContent = `Set ${label.replace(":", "")}`;
                    title.style.cssText = "font-size:13px;font-weight:bold;color:var(--fg-color);";

                    const popInp = document.createElement("input");
                    popInp.type = "text"; popInp.inputMode = "decimal"; popInp.value = currentVal;
                    popInp.style.cssText = inputStyle + "width:100%;box-sizing:border-box;font-size:14px;padding:6px 8px;";

                    const btnRow = document.createElement("div");
                    btnRow.style.cssText = "display:flex;gap:8px;justify-content:flex-end;";
                    const cancel = document.createElement("button");
                    cancel.textContent = "Cancel";
                    cancel.style.cssText = inputStyle + "cursor:pointer;padding:4px 12px;";
                    const ok = document.createElement("button");
                    ok.textContent = "OK";
                    ok.style.cssText = inputStyle + "cursor:pointer;padding:4px 12px;font-weight:bold;";
                    const close = () => document.body.removeChild(overlay);
                    cancel.onclick = close;
                    ok.onclick = () => {
                        const v = parseFloat(popInp.value);
                        if (!isNaN(v)) onConfirm(v);
                        close();
                    };
                    popInp.addEventListener("keydown", (e) => {
                        if (e.key === "Enter") ok.click();
                        if (e.key === "Escape") close();
                    });
                    overlay.addEventListener("mousedown", (e) => { if (e.target === overlay) close(); });
                    btnRow.append(cancel, ok);
                    panel.append(title, popInp, btnRow);
                    overlay.appendChild(panel);
                    document.body.appendChild(overlay);
                    requestAnimationFrame(() => { popInp.focus(); popInp.select(); });
                }

                function num(val, label) {
                    const wrap = document.createElement("div");
                    wrap.style.cssText = "display:flex;align-items:center;gap:2px;flex-shrink:0;";
                    const lbl = document.createElement("span");
                    lbl.textContent = label;
                    lbl.style.fontSize = "10px";

                    const inp = document.createElement("input");
                    inp.type = "text"; inp.inputMode = "decimal"; inp.value = Number.isFinite(Number(val)) ? Number(val).toFixed(2) : "0.00";
                    inp.style.cssText = inputStyle + "width:41px;text-align:center;flex-shrink:0;box-sizing:border-box;";

                    inp.addEventListener("change", syncData);
                    inp.addEventListener("input", syncData);
                    inp.addEventListener("click", (e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        showNumPopup(inp.value, label, (newVal) => {
                            inp.value = newVal.toFixed(2);
                            syncData();
                        });
                    });
                    wrap.append(lbl, inp);
                    return { wrap, inp };
                }

                const str = num(data.str, "S:");

                const rm = document.createElement("button");
                rm.innerHTML = "✖";
                rm.style.cssText = "background:transparent;color:var(--error-text);border:1px solid transparent;cursor:pointer;font-size:12px;padding:2px 6px;border-radius:4px;transition:all 0.1s ease;flex-shrink:0;";

                rm.addEventListener("mouseenter", () => {
                    rm.style.background = "var(--comfy-menu-bg)";
                    rm.style.borderColor = "var(--primary-color)";
                    rm.style.color = "#ff5555";
                });
                rm.addEventListener("mouseleave", () => {
                    rm.style.background = "transparent";
                    rm.style.borderColor = "transparent";
                    rm.style.color = "var(--error-text)";
                });

                const slotObj = {
                    row,
                    getValue: () => ({
                        on: chk.checked,
                        lora: sel.dataset.lora,
                        str: parseFloat(str.inp.value) || 0.0
                    }),
                    getLora: () => sel.dataset.lora,
                    refreshDisplayName: () => {
                        const lora = sel.dataset.lora;
                        selText.textContent = loraDisplayName(lora, slots);
                    },
                    checkMissing: checkMissing,
                    remove: () => {
                        row.remove();
                        slots = slots.filter(s => s !== slotObj);
                        syncData();
                    }
                };
                rm.onclick = slotObj.remove;
                row.append(handle, chk, sel, str.wrap, rm);
                slots.push(slotObj);
                makeDraggable(row);
                container.appendChild(row);

                checkMissing();
                syncData();
            }

            const addBtn = document.createElement("button");
            addBtn.textContent = "＋ Add LoRA";
            addBtn.style.cssText = inputStyle + "width:100%;cursor:pointer;font-weight:bold;transition:all 0.1s ease;";
            addBtn.addEventListener("mouseenter", () => {
                addBtn.style.background = "var(--comfy-menu-bg)";
                addBtn.style.borderColor = "var(--primary-color)";
            });
            addBtn.addEventListener("mouseleave", () => {
                addBtn.style.background = "var(--comfy-input-bg)";
                addBtn.style.borderColor = "var(--border-color)";
            });
            addBtn.onclick = () => addSlot();
            container.appendChild(addBtn);

            const loraUIWidget = node.addDOMWidget("lora_ui", "HTML", container);
            loraUIWidget.computeLayoutSize = () => ({
                minHeight: container.scrollHeight + 8,
                maxHeight: container.scrollHeight + 8,
                minWidth: 0
            });

            initialData.forEach(d => addSlot(d));
            requestAnimationFrame(syncSize);

            const origConfigure = node.configure?.bind(node);
            node.configure = function (data) {
                origConfigure?.(data);
                for (const s of [...slots]) s.row.remove();
                slots = [];
                try {
                    const raw = data?.properties?.["stack_data"] || "[]";
                    if (stackWidget) stackWidget.value = raw;
                    JSON.parse(raw).forEach(d => addSlot(d));
                } catch {}
                requestAnimationFrame(syncSize);
            };
        };
    }
});

