(() => {
  "use strict";

  const state = {
    router: null,
    telemetry: null,
    category: "All",
    search: "",
    includeUnlisted: false,
    consoleTab: "llama-swap",
    benchFile: "",
    busyModel: null,
    pendingSwitch: null,
    smokeResults: {},
    sse: null,
    logTimer: null,
    lastRunningKey: "",
  };

  const $ = (id) => document.getElementById(id);

  const escapeHtml = (val) =>
    String(val ?? "").replace(/[&<>'"]/g, (ch) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      "'": "&#39;",
      '"': "&quot;",
    })[ch]);

  function fmtGiB(mib) {
    if (mib == null || Number.isNaN(mib)) return "--";
    return (mib / 1024).toFixed(1);
  }

  function meterClass(pct, baseClass = "") {
    if (pct == null) return `meter-fill ${baseClass}`.trim();
    if (pct >= 95) return "meter-fill meter-crit";
    if (pct >= 85) return "meter-fill meter-warn";
    return `meter-fill ${baseClass}`.trim();
  }

  function showToast(msg, kind = "ok") {
    const bar = $("toast-bar");
    bar.className = `surface toast-bar ${kind === "err" ? "err" : kind === "warn" ? "warn" : ""}`.trim();
    $("toast-message").textContent = msg;
    bar.hidden = false;
  }

  function hideToast() {
    $("toast-bar").hidden = true;
  }

  function resolveDeepLinkHost() {
    const host = window.location.hostname;
    if (host && host !== "127.0.0.1" && host !== "localhost" && host !== "::1") {
      return host;
    }
    return state.router?.endpoints?.tailscale_host || "yeti-cachy.llama-alnair.ts.net";
  }

  function updateDeepLinks() {
    const host = resolveDeepLinkHost();
    const uiPort = state.router?.endpoints?.ui_port || 8089;
    const swapPort = state.router?.endpoints?.swap_port || 8080;
    const chatUrl = `http://${host}:${uiPort}/`;
    const comfyUrl = `http://${host}:${swapPort}/comfyui/`;
    $("link-chat-top").href = chatUrl;
    $("link-comfy-top").href = comfyUrl;
    $("hero-chat-link").href = chatUrl;
    $("hero-comfy-link").href = comfyUrl;
  }

  function renderTelemetry(t) {
    if (!t) return;
    state.telemetry = t;

    const sampleDate = new Date((t.ts || Date.now() / 1000) * 1000);
    $("last-sample-val").textContent = sampleDate.toLocaleTimeString();

    // GPU
    const gpu = t.gpu || {};
    if (gpu.available) {
      $("gpu-vram-val").textContent = `${Math.round(gpu.vram_used_mib ?? 0)} / ${Math.round(gpu.vram_total_mib ?? 0)} MiB`;
      const bar = $("gpu-vram-bar");
      bar.style.width = `${Math.min(100, gpu.vram_pct ?? 0)}%`;
      bar.className = meterClass(gpu.vram_pct);
      $("gpu-sub-val").textContent =
        `Util ${gpu.util_pct ?? "--"}% · ${gpu.temp_c ?? "--"} °C · ${gpu.power_w ?? "--"} / ${gpu.power_limit_w ?? "--"} W`;
    } else {
      $("gpu-vram-val").textContent = "UNAVAILABLE";
      $("gpu-vram-bar").style.width = "0%";
      $("gpu-sub-val").textContent = gpu.error || "nvidia-smi unavailable";
    }

    // RAM + zram
    const mem = t.memory || {};
    if (mem.available) {
      $("ram-val").textContent = `${fmtGiB(mem.used_mib)} / ${fmtGiB(mem.total_mib)} GiB`;
      const rbar = $("ram-bar");
      rbar.style.width = `${Math.min(100, mem.used_pct ?? 0)}%`;
      rbar.className = meterClass(mem.used_pct, "meter-blue");
      const zramGiB = fmtGiB(mem.zram_used_mib ?? mem.swap_used_mib);
      $("zram-sub-val").textContent = `Swap/zram: ${zramGiB} GiB · Avail: ${fmtGiB(mem.available_mib)} GiB`;
    }

    // CPU P/E
    const cpu = t.cpu || {};
    const pPct = cpu.p_core_pct;
    const ePct = cpu.e_core_pct;
    $("cpu-val").textContent = `P ${pPct != null ? pPct.toFixed(0) : "--"}% · E ${ePct != null ? ePct.toFixed(0) : "--"}%`;
    $("cpu-p-bar").style.width = `${Math.min(100, pPct ?? 0)}%`;
    $("cpu-p-bar").className = meterClass(pPct);
    $("cpu-e-bar").style.width = `${Math.min(100, ePct ?? 0)}%`;
    $("cpu-e-bar").className = meterClass(ePct, "meter-blue");

    // Governor + Tuning + Spill
    const tuning = t.tuning || {};
    const govPill = $("gov-pill");
    const govText = (tuning.governor || "unknown").toUpperCase();
    govPill.textContent = tuning.governor_healthy ? `GOV: ${govText}` : `ALERT: ${govText}`;
    govPill.className = tuning.governor_healthy ? "gov-pill" : "gov-pill gov-alert";
    govPill.title = tuning.governor_alert || "CPU frequency governor healthy (performance)";

    const swapChip = $("swappiness-chip");
    swapChip.textContent = `swap=${tuning.vm_swappiness ?? "?"}`;
    swapChip.className = tuning.swappiness_high ? "tuning-chip warn" : "tuning-chip";
    $("thp-chip").textContent = `THP=${tuning.thp || "?"}`;

    const io = t.io || {};
    const spill = $("spill-badge");
    if (io.spill_state === "idle") {
      spill.textContent = "NVMe: IDLE";
      spill.className = "spill-badge";
    } else if (io.spill_state === "spill") {
      spill.textContent = `SPILL ${io.read_mb_s} MB/s (${io.active_engine})`;
      spill.className = "spill-badge crit";
    } else if (io.spill_state === "elevated") {
      spill.textContent = `NVMe ${io.read_mb_s} MB/s (${io.active_engine})`;
      spill.className = "spill-badge warn";
    } else {
      spill.textContent = `NVMe ${io.read_mb_s} MB/s · OK`;
      spill.className = "spill-badge ok";
    }

    // Disks (local df -hlT)
    const disks = t.disks || [];
    if (disks.length) {
      $("disk-strip").innerHTML = disks
        .map((d) => {
          const cls = d.critical ? "disk-chip crit" : d.warning ? "disk-chip warn" : "disk-chip";
          return `<span class="${cls}" title="${escapeHtml(d.filesystem)} (${escapeHtml(d.fstype)}) ${escapeHtml(d.avail)} free">${escapeHtml(d.mount)} ${d.use_pct}%</span>`;
        })
        .join("");
    }
  }

  function renderResidentHero() {
    const r = state.router;
    if (!r) return;

    updateDeepLinks();
    const routerVal = $("router-status-val");
    if (r.reachable) {
      routerVal.textContent = `${r.running.length} active · ${r.catalog_total} defined`;
      routerVal.style.color = "var(--green)";
    } else {
      routerVal.textContent = "UNREACHABLE";
      routerVal.style.color = "var(--pink)";
    }

    const hero = document.querySelector(".hero-panel");
    const listEl = $("resident-list");
    const inspector = $("engine-inspector");
    const unloadBtn = $("unload-all-btn");

    if (!r.reachable) {
      hero.classList.add("idle");
      $("resident-title").textContent = "llama-swap unreachable (:8080)";
      $("resident-subtitle").textContent = r.error || "Router did not respond.";
      listEl.innerHTML = "";
      inspector.hidden = true;
      unloadBtn.disabled = true;
      return;
    }

    if (!r.running.length) {
      hero.classList.add("idle");
      $("resident-title").textContent = "VRAM IDLE — NO MODEL RESIDENT";
      $("resident-subtitle").textContent = "Select any profile below to wake on demand, or open ComfyUI.";
      listEl.innerHTML = "";
      inspector.hidden = true;
      unloadBtn.disabled = true;
      return;
    }

    hero.classList.remove("idle");
    unloadBtn.disabled = Boolean(state.busyModel);
    const primary = r.running[0];
    $("resident-title").textContent = r.running.map((item) => item.model).join(" + ");
    $("resident-subtitle").textContent = primary.name || `${primary.engine} · ${primary.state}`;

    listEl.innerHTML = r.running
      .map(
        (item) => `
        <div class="resident-chip">
          <strong>${escapeHtml(item.model)}</strong>
          <span class="badge badge-green">${escapeHtml(item.state)}</span>
          <span class="badge badge-blue">${escapeHtml(item.engine)}</span>
          <small>group:${escapeHtml(item.group)} · ttl:${item.ttl}s</small>
        </div>`
      )
      .join("");

    const cfg = r.active_engine_config;
    if (cfg) {
      inspector.hidden = false;
      $("inspector-path").textContent = cfg.config_path || `${cfg.engine} CLI flags`;
      const envEntries = Object.entries(cfg.env || {});
      $("inspector-env").innerHTML = envEntries.length
        ? envEntries.map(([k, v]) => `<span class="flag-chip">${escapeHtml(k)}=${escapeHtml(v)}</span>`).join("")
        : '<span class="muted">No explicit env overrides</span>';

      const visionParts = [];
      if (cfg.vision) {
        if (cfg.vision.exe) {
          visionParts.push(`Vision: ${cfg.vision.gpu === false ? "CPU" : "GPU"} (${cfg.vision.threads || "?"}t)`);
        } else {
          visionParts.push("Vision: enabled");
        }
      } else {
        visionParts.push("Vision: text-only");
      }
      if (cfg.sampling) {
        visionParts.push(`temp=${cfg.sampling.temperature} top_p=${cfg.sampling.top_p}`);
      }
      if (cfg.reasoning_budget_tokens) {
        visionParts.push(`think_budget=${cfg.reasoning_budget_tokens}`);
      }
      $("inspector-vision").textContent = visionParts.join(" · ");

      const args = cfg.args || [];
      $("inspector-args-count").textContent = String(args.length);
      $("inspector-args").textContent = args.join(" ") || "(none)";
    } else {
      inspector.hidden = true;
    }
  }

  function renderModelGrid() {
    const r = state.router;
    if (!r) return;
    $("unlisted-count").textContent = String(r.unlisted_count || 0);

    const models = r.catalog || [];
    const categories = ["All", ...new Set(models.map((m) => m.category))];
    if (!categories.includes(state.category)) state.category = "All";

    $("category-filters").innerHTML = categories
      .map(
        (cat) =>
          `<button class="filter ${cat === state.category ? "active" : ""}" data-category="${escapeHtml(cat)}">${escapeHtml(cat)}</button>`
      )
      .join("");

    const needle = state.search.trim().toLowerCase();
    const filtered = models.filter((m) => {
      if (state.category !== "All" && m.category !== state.category) return false;
      if (!needle) return true;
      const hay = [m.id, m.name, m.description, m.engine, m.group, ...(m.aliases || [])].join(" ").toLowerCase();
      return hay.includes(needle);
    });

    const container = $("model-groups");
    if (!filtered.length) {
      container.innerHTML = '<div class="surface empty-cell">No matching models in catalog.</div>';
      return;
    }

    const grouped = new Map();
    for (const m of filtered) {
      if (!grouped.has(m.category)) grouped.set(m.category, []);
      grouped.get(m.category).push(m);
    }

    container.innerHTML = [...grouped.entries()]
      .map(([cat, items]) => {
        const cards = items
          .map((m) => {
            const busy = state.busyModel === m.id;
            const anyBusy = Boolean(state.busyModel);
            const smoke = state.smokeResults[m.id];
            const aliases = (m.aliases || [])
              .map((a) => `<span class="badge">alias:${escapeHtml(a)}</span>`)
              .join("");
            const smokeHtml = smoke
              ? `<div class="smoke-result ${smoke.ok ? "ok" : "err"}">
                   ${smoke.ok ? "SMOKE OK" : "SMOKE FAIL"} · ${smoke.elapsed_s}s · HTTP ${smoke.http_status}${smoke.has_reasoning ? " · reasoning" : ""}
                   <div>${escapeHtml(smoke.snippet || smoke.error || "")}</div>
                 </div>`
              : "";
            return `
              <article class="surface model-card ${m.running ? "is-running" : ""}">
                <div>
                  <div class="model-card-head">
                    <span class="model-id">${escapeHtml(m.id)}</span>
                    ${m.running ? '<span class="badge badge-green">RESIDENT</span>' : ""}
                  </div>
                  <div class="model-name">${escapeHtml(m.name)}</div>
                  ${m.description ? `<p class="model-desc">${escapeHtml(m.description.slice(0, 175))}${m.description.length > 175 ? "…" : ""}</p>` : ""}
                  <div class="badge-row">
                    <span class="badge badge-blue">${escapeHtml(m.engine)}</span>
                    <span class="badge ${m.exclusive ? "" : "badge-green"}">${m.exclusive ? "exclusive" : "co-resident"}</span>
                    ${m.preload ? '<span class="badge badge-pink">preload</span>' : ""}
                    ${m.has_vision ? '<span class="badge badge-blue">vision</span>' : ""}
                    ${m.unlisted ? '<span class="badge badge-amber">unlisted</span>' : ""}
                    <span class="badge">ttl:${m.ttl}s</span>
                    ${aliases}
                  </div>
                </div>
                <div>
                  <div class="card-actions">
                    <button class="button ${m.running ? "" : "btn-green"}" data-wake="${escapeHtml(m.id)}" ${anyBusy ? "disabled" : ""}>
                      ${busy ? "Working..." : m.running ? "Resident" : "Wake / Switch"}
                    </button>
                    <button class="button btn-amber" data-smoke="${escapeHtml(m.id)}" ${anyBusy ? "disabled" : ""}>
                      Smoke
                    </button>
                  </div>
                  ${smokeHtml}
                </div>
              </article>`;
          })
          .join("");
        return `<div class="group-block"><div class="group-title">${escapeHtml(cat)} (${items.length})</div><div class="model-grid">${cards}</div></div>`;
      })
      .join("");
  }

  async function fetchRouter() {
    try {
      const q = state.includeUnlisted ? "?include_unlisted=1" : "";
      const res = await fetch(`/api/router${q}`, { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      state.router = await res.json();
      state.lastRunningKey = JSON.stringify((state.router.running || []).map((r) => `${r.model}:${r.state}`));
      renderResidentHero();
      renderModelGrid();
    } catch (err) {
      showToast(`Failed to query /api/router: ${err.message}`, "err");
    }
  }

  function requestWake(modelId) {
    const model = state.router?.catalog?.find((m) => m.id === modelId);
    const running = state.router?.running || [];
    const victims = running
      .filter((r) => r.model !== modelId && (model ? model.exclusive : true) && r.exclusive !== false)
      .map((r) => r.model);

    if (victims.length > 0) {
      state.pendingSwitch = modelId;
      $("confirm-text").textContent = `Switching to "${modelId}" will evict resident model(s): ${victims.join(", ")}.`;
      $("confirm-banner").hidden = false;
      $("confirm-banner").scrollIntoView({ behavior: "smooth", block: "nearest" });
      return;
    }
    executeWake(modelId);
  }

  async function executeWake(modelId) {
    $("confirm-banner").hidden = true;
    state.pendingSwitch = null;
    state.busyModel = modelId;
    renderResidentHero();
    renderModelGrid();
    showToast(`Waking ${modelId}... (may take up to 2 minutes on cold load)`, "warn");

    try {
      const res = await fetch(`/api/models/${encodeURIComponent(modelId)}/wake`, { method: "POST" });
      const data = await res.json();
      if (data.ok) {
        showToast(`Loaded ${modelId} in ${data.elapsed_s}s (HTTP ${data.http_status}).`, "ok");
      } else {
        showToast(`Wake failed for ${modelId}: ${data.error || `HTTP ${data.http_status}`}`, "err");
      }
    } catch (err) {
      showToast(`Wake request error: ${err.message}`, "err");
    } finally {
      state.busyModel = null;
      await fetchRouter();
      if (state.consoleTab !== "benches") fetchLogs();
    }
  }

  async function executeUnloadAll() {
    state.busyModel = "__unload__";
    renderResidentHero();
    renderModelGrid();
    showToast("Unloading resident models to free VRAM...", "warn");

    try {
      const res = await fetch("/api/models/unload", { method: "POST" });
      const data = await res.json();
      if (data.ok) {
        showToast(`Unloaded in ${data.elapsed_s}s.`, "ok");
      } else {
        showToast(`Unload failed: ${data.error || `HTTP ${data.http_status}`}`, "err");
      }
    } catch (err) {
      showToast(`Unload request error: ${err.message}`, "err");
    } finally {
      state.busyModel = null;
      await fetchRouter();
    }
  }

  async function executeSmoke(modelId) {
    state.busyModel = modelId;
    renderModelGrid();
    showToast(`Running smoke probe against ${modelId}...`, "warn");

    try {
      const res = await fetch(`/api/models/${encodeURIComponent(modelId)}/smoke`, { method: "POST" });
      const data = await res.json();
      state.smokeResults[modelId] = data;
      if (data.ok) {
        showToast(`Smoke OK for ${modelId} (${data.elapsed_s}s): ${data.snippet}`, "ok");
      } else {
        showToast(`Smoke failed for ${modelId}: ${data.error}`, "err");
      }
    } catch (err) {
      showToast(`Smoke request error: ${err.message}`, "err");
    } finally {
      state.busyModel = null;
      await fetchRouter();
    }
  }

  async function fetchLogs() {
    const source = state.consoleTab === "engine" ? "engine" : "llama-swap";
    const lines = $("log-lines-select").value || "100";
    try {
      const res = await fetch(`/api/logs?source=${encodeURIComponent(source)}&lines=${encodeURIComponent(lines)}`, {
        cache: "no-store",
      });
      const data = await res.json();
      const meta = data.path ? `${data.source} · ${data.path} (${data.count} lines)` : `${data.source}`;
      $("log-meta").textContent = data.note ? `${meta} — ${data.note}` : meta;
      const outEl = $("log-output");
      outEl.textContent = (data.lines || []).length ? data.lines.join("\n") : (data.note || "(no log output)");
      outEl.scrollTop = outEl.scrollHeight;
    } catch (err) {
      $("log-output").textContent = `Error fetching logs: ${err.message}`;
    }
  }

  async function fetchBenches() {
    try {
      const q = state.benchFile ? `?file=${encodeURIComponent(state.benchFile)}` : "";
      const res = await fetch(`/api/benches${q}`, { cache: "no-store" });
      const data = await res.json();

      const sel = $("bench-file-select");
      const current = state.benchFile;
      const opts = ['<option value="">All JSONL files</option>'].concat(
        (data.files || []).map((f) => `<option value="${escapeHtml(f)}" ${f === current ? "selected" : ""}>${escapeHtml(f)}</option>`)
      );
      sel.innerHTML = opts.join("");

      const tbody = $("bench-table-body");
      const runs = data.runs || [];
      if (!runs.length) {
        tbody.innerHTML = '<tr><td colspan="7" class="empty-cell">No JSONL benchmark rows found in logs/results/.</td></tr>';
        return;
      }

      tbody.innerHTML = runs
        .map((r) => {
          const tg = r.tg_steady != null ? `<strong class="accent">${r.tg_steady.toFixed(1)} t/s</strong>` : (r.hit_rate_pct != null ? `hit ${r.hit_rate_pct}%` : "--");
          const aux = r.edit_tg != null
            ? `edit ${r.edit_tg.toFixed(1)} t/s`
            : r.draft_acc_pct != null
            ? `acc ${r.draft_acc_pct}%`
            : r.io_mb_per_token != null
            ? `${r.io_mb_per_token} MB/tok`
            : "--";
          const ppParts = [
            r.pp2k != null ? r.pp2k.toFixed(0) : "--",
            r.pp8k != null ? r.pp8k.toFixed(0) : "--",
            r.pp_long != null ? `${r.pp_long.toFixed(0)} (${r.pp_long_label})` : "--",
          ].join(" / ");
          const vram = r.vram_mib != null ? `${r.vram_mib} MiB` : "--";
          return `
            <tr>
              <td>${escapeHtml(r.ts.replace("T", " ").slice(0, 19))}</td>
              <td><span class="badge">${escapeHtml(r.file)}</span></td>
              <td><strong>${escapeHtml(r.label)}</strong></td>
              <td>${tg}</td>
              <td>${escapeHtml(aux)}</td>
              <td>${escapeHtml(ppParts)}</td>
              <td>${escapeHtml(vram)}</td>
            </tr>`;
        })
        .join("");
    } catch (err) {
      $("bench-table-body").innerHTML = `<tr><td colspan="7" class="empty-cell">Error: ${escapeHtml(err.message)}</td></tr>`;
    }
  }

  function switchConsoleTab(tab) {
    state.consoleTab = tab;
    document.querySelectorAll("[data-console-tab]").forEach((btn) => {
      const active = btn.dataset.consoleTab === tab;
      btn.classList.toggle("active", active);
      btn.setAttribute("aria-selected", String(active));
    });
    const isBenches = tab === "benches";
    $("console-log-view").hidden = isBenches;
    $("log-controls").hidden = isBenches;
    $("console-bench-view").hidden = !isBenches;
    $("bench-controls").hidden = !isBenches;

    if (isBenches) {
      fetchBenches();
    } else {
      fetchLogs();
    }
  }

  function connectSSE() {
    if (state.sse) state.sse.close();
    const pill = $("sse-pill");
    const text = $("sse-status-text");

    const es = new EventSource("/api/events");
    state.sse = es;

    es.onopen = () => {
      pill.className = "status-pill live";
      text.textContent = "SSE LIVE (2s)";
    };

    es.addEventListener("snapshot", (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        pill.className = "status-pill live";
        text.textContent = "SSE LIVE (2s)";
        if (msg.telemetry) renderTelemetry(msg.telemetry);
        if (msg.router) {
          const newKey = JSON.stringify((msg.router.running || []).map((r) => `${r.model}:${r.state}`));
          if (newKey !== state.lastRunningKey || (state.router && state.router.reachable !== msg.router.reachable)) {
            fetchRouter();
          }
        }
      } catch {
        // Ignore transient parse issue
      }
    });

    es.onerror = () => {
      pill.className = "status-pill warn";
      text.textContent = "RECONNECTING...";
    };
  }

  // Event wiring
  document.addEventListener("click", (ev) => {
    const catBtn = ev.target.closest("[data-category]");
    if (catBtn) {
      state.category = catBtn.dataset.category;
      renderModelGrid();
      return;
    }
    const wakeBtn = ev.target.closest("[data-wake]");
    if (wakeBtn) {
      requestWake(wakeBtn.dataset.wake);
      return;
    }
    const smokeBtn = ev.target.closest("[data-smoke]");
    if (smokeBtn) {
      executeSmoke(smokeBtn.dataset.smoke);
      return;
    }
    const tabBtn = ev.target.closest("[data-console-tab]");
    if (tabBtn) {
      switchConsoleTab(tabBtn.dataset.consoleTab);
    }
  });

  $("model-search").addEventListener("input", (ev) => {
    state.search = ev.target.value;
    renderModelGrid();
  });
  $("toggle-unlisted").addEventListener("change", (ev) => {
    state.includeUnlisted = ev.target.checked;
    fetchRouter();
  });
  $("unload-all-btn").addEventListener("click", executeUnloadAll);
  $("confirm-proceed-btn").addEventListener("click", () => {
    if (state.pendingSwitch) executeWake(state.pendingSwitch);
  });
  $("confirm-cancel-btn").addEventListener("click", () => {
    state.pendingSwitch = null;
    $("confirm-banner").hidden = true;
  });
  $("toast-close-btn").addEventListener("click", hideToast);
  $("refresh-console-btn").addEventListener("click", fetchLogs);
  $("log-lines-select").addEventListener("change", fetchLogs);
  $("bench-file-select").addEventListener("change", (ev) => {
    state.benchFile = ev.target.value;
    fetchBenches();
  });
  $("refresh-bench-btn").addEventListener("click", fetchBenches);

  // Initial load + periodic log auto-refresh
  fetchRouter();
  fetch("/api/telemetry", { cache: "no-store" })
    .then((r) => r.json())
    .then(renderTelemetry)
    .catch(() => {});
  fetchLogs();
  connectSSE();

  state.logTimer = setInterval(() => {
    if (state.consoleTab !== "benches" && $("log-autofollow").checked) {
      fetchLogs();
    }
  }, 5000);
})();
