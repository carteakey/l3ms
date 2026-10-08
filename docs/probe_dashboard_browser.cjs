#!/usr/bin/env node
// Opt-in, dependency-free Firefox/WebDriver BiDi layout and keyboard probe.
// It serves only docs/ on loopback and uses a fresh profile for every run.
const assert = require("node:assert/strict");
const { spawn } = require("node:child_process");
const fs = require("node:fs");
const http = require("node:http");
const os = require("node:os");
const path = require("node:path");

const REPO_ROOT = path.resolve(__dirname, "..");
const DOCS_ROOT = __dirname;
const TOOLKIT_ROOT = path.join(REPO_ROOT, ".toolkit");
const OUTER_DEADLINE_MS = 90_000;
const CLEANUP_BUDGET_MS = 8_000;
const MAX_SCREENSHOT_BYTES = 8 * 1024 * 1024;
const MAX_BIDI_MESSAGE_BYTES = 2 * 1024 * 1024;
const MAX_BROWSER_STDERR_BYTES = 64 * 1024;
const MAX_REPORTED_EXCEPTIONS = 32;
const MAX_EXCEPTION_TEXT = 512;
const ALLOWED_FILES = new Map([
  ["/", ["index.html", "text/html; charset=utf-8"]],
  ["/index.html", ["index.html", "text/html; charset=utf-8"]],
  ["/probe-blank", [null, "text/html; charset=utf-8"]],
  ["/generated-models.js", ["generated-models.js", "text/javascript; charset=utf-8"]],
  ["/favicon.svg", ["favicon.svg", "image/svg+xml"]],
  ["/favicon.ico", ["favicon.svg", "image/svg+xml"]],
]);

function parseArgs(argv) {
  if (argv.length === 1 && argv[0] === "--help") {
    process.stdout.write("Opt-in Firefox dashboard probe. Run with: node docs/probe_dashboard_browser.cjs --run\n");
    process.exit(0);
  }
  if (argv.length !== 1 || argv[0] !== "--run") {
    throw new Error("This probe is opt-in; pass exactly --run (or --help).");
  }
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function makeDeadline() {
  const end = Date.now() + OUTER_DEADLINE_MS - CLEANUP_BUDGET_MS;
  return {
    remaining(label) {
      const ms = end - Date.now();
      if (ms <= 0) throw new Error(`Probe-work deadline reached during ${label}; cleanup reserve remains.`);
      return ms;
    },
    async wait(promise, label, capMs = Infinity) {
      const ms = Math.min(this.remaining(label), capMs);
      let timer;
      try {
        return await Promise.race([
          promise,
          new Promise((_, reject) => {
            timer = setTimeout(() => reject(new Error(`Timed out during ${label}.`)), ms);
          }),
        ]);
      } finally {
        clearTimeout(timer);
      }
    },
    end,
  };
}

function shortError(error) {
  const message = error instanceof Error ? error.message : String(error);
  return message.replace(/[\r\n\0]/g, " ").slice(0, 1200);
}

function startStaticServer(deadline) {
  const allowed = new Map([...ALLOWED_FILES].map(([urlPath, [file, contentType]]) => [
    urlPath,
    {
      body: file === null
        ? Buffer.from("<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><title>Probe fixture</title><body>Isolated local event fixture</body></html>", "utf8")
        : fs.readFileSync(path.join(DOCS_ROOT, file)),
      contentType,
    },
  ]));
  for (const item of allowed.values()) assert.ok(item.body.length <= 2 * 1024 * 1024, "dashboard asset unexpectedly large");

  const seenPaths = new Set();
  let requestCount = 0;
  const server = http.createServer((request, response) => {
    requestCount += 1;
    if (requestCount > 64) {
      response.writeHead(429, { "Connection": "close" }).end();
      return;
    }
    let pathname;
    try {
      pathname = new URL(request.url || "/", "http://127.0.0.1").pathname;
    } catch {
      response.writeHead(400, { "Connection": "close" }).end();
      return;
    }
    if (pathname.length > 256) {
      response.writeHead(414, { "Connection": "close" }).end();
      return;
    }
    seenPaths.add(pathname);
    const asset = allowed.get(pathname);
    if (!asset || (request.method !== "GET" && request.method !== "HEAD")) {
      response.writeHead(404, { "Connection": "close", "Cache-Control": "no-store" }).end();
      return;
    }
    response.writeHead(200, {
      "Content-Type": asset.contentType,
      "Content-Length": asset.body.length,
      "Cache-Control": "no-store",
      "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self' data:; connect-src 'none'; frame-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'",
      "Referrer-Policy": "no-referrer",
      "X-Content-Type-Options": "nosniff",
      "Connection": "close",
    });
    if (request.method === "HEAD") response.end();
    else response.end(asset.body);
  });
  server.requestTimeout = 2_000;
  server.headersTimeout = 2_000;

  const listening = new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      server.removeListener("error", reject);
      const address = server.address();
      assert.ok(address && typeof address === "object");
      resolve({ url: `http://127.0.0.1:${address.port}/`, port: address.port });
    });
  });
  return deadline.wait(listening, "local static server startup", 5_000).then(({ url, port }) => ({
    server,
    url,
    port,
    seenPaths,
    close: () => new Promise(resolve => {
      server.closeAllConnections();
      server.close(() => resolve());
    }),
  }));
}

function childHandleIsLive(child) {
  return Boolean(child && child.exitCode === null && child.signalCode === null);
}

function readProcIdentity(pid) {
  try {
    const stat = fs.readFileSync(`/proc/${pid}/stat`, "utf8");
    const tail = stat.slice(stat.lastIndexOf(")") + 2).trim().split(/\s+/);
    const startTime = tail[19]; // field 22 (starttime), after pid/comm/state.
    const processGroup = tail[2]; // field 5 (pgrp).
    if (!/^\d+$/.test(startTime || "") || !/^\d+$/.test(processGroup || "")) return { status: "unknown" };
    return { status: "present", startTime, processGroup: Number(processGroup) };
  } catch (error) {
    return error.code === "ENOENT" ? { status: "gone" } : { status: "unknown" };
  }
}

function observeProcessTree(tracker) {
  if (!tracker.rootPid) return;
  const queue = [tracker.rootPid];
  const visited = new Set();
  while (queue.length && tracker.pids.size < 512) {
    const pid = queue.pop();
    if (!Number.isSafeInteger(pid) || pid <= 1 || visited.has(pid)) continue;
    visited.add(pid);
    const identity = readProcIdentity(pid);
    if (identity.status === "present") tracker.pids.set(pid, identity.startTime);
    if (identity.status !== "present") continue;
    try {
      const children = fs.readFileSync(`/proc/${pid}/task/${pid}/children`, "utf8");
      for (const childPid of children.trim().split(/\s+/)) {
        if (/^\d+$/.test(childPid)) queue.push(Number(childPid));
      }
    } catch {
      // A process may exit between reading stat and its children file.
    }
  }
  if (queue.length) tracker.overflow = true;
}

function processTreeState(tracker) {
  const remaining = [];
  let unknown = false;
  for (const [pid, originalStartTime] of tracker.pids) {
    const current = readProcIdentity(pid);
    if (current.status === "unknown") unknown = true;
    else if (current.status === "present" && current.startTime === originalStartTime) remaining.push(pid);
  }
  return { remaining, unknown };
}

function processGroupState(groupId) {
  const members = [];
  let unknown = false;
  let entries;
  try { entries = fs.readdirSync("/proc"); }
  catch { return { members, unknown: true }; }
  for (const entry of entries) {
    if (!/^\d+$/.test(entry)) continue;
    const info = readProcIdentity(Number(entry));
    if (info.status === "unknown") {
      // We only need a full census of this isolated session's group; unreadable entries
      // cannot be assumed members, but mark verification indeterminate conservatively.
      unknown = true;
    } else if (info.status === "present" && info.processGroup === groupId) {
      members.push(Number(entry));
    }
  }
  return { members, unknown };
}

async function waitForFirefoxTreeExit(firefox, capMs) {
  const deadline = Date.now() + capMs;
  let lastGroupCheck = 0;
  let groupState = { members: [], unknown: false };
  let groupCheckedAfterLeaderExit = false;
  while (Date.now() < deadline) {
    observeProcessTree(firefox.tracker);
    const state = processTreeState(firefox.tracker);
    const leaderExited = !childHandleIsLive(firefox.child);
    if (leaderExited && (!groupCheckedAfterLeaderExit || Date.now() - lastGroupCheck >= 150)) {
      groupState = processGroupState(firefox.tracker.groupId);
      lastGroupCheck = Date.now();
      groupCheckedAfterLeaderExit = true;
    }
    if (leaderExited && groupCheckedAfterLeaderExit && state.remaining.length === 0 && !state.unknown &&
        groupState.members.length === 0 && !groupState.unknown && !firefox.tracker.overflow) {
      return { verified: true, remaining: 0 };
    }
    await sleep(40);
  }
  observeProcessTree(firefox.tracker);
  const state = processTreeState(firefox.tracker);
  if (!childHandleIsLive(firefox.child)) {
    groupState = processGroupState(firefox.tracker.groupId);
    groupCheckedAfterLeaderExit = true;
  }
  return {
    verified: !childHandleIsLive(firefox.child) && groupCheckedAfterLeaderExit && state.remaining.length === 0 && !state.unknown &&
      groupState.members.length === 0 && !groupState.unknown && !firefox.tracker.overflow,
    remaining: Math.max(state.remaining.length, groupState.members.length),
    unknown: state.unknown || groupState.unknown,
    overflow: firefox.tracker.overflow,
  };
}

async function stopFirefox(firefox) {
  if (!firefox?.child || !firefox.child.pid) {
    if (firefox?.tracker?.interval) clearInterval(firefox.tracker.interval);
    return { verified: true, remaining: 0 };
  }
  const { child, tracker } = firefox;
  observeProcessTree(tracker);
  if (childHandleIsLive(child)) {
    try { child.kill("SIGTERM"); } catch {}
  }
  let result = await waitForFirefoxTreeExit(firefox, 2_500);
  if (!result.verified && childHandleIsLive(child)) {
    // ChildProcess.kill targets only the still-live child handle; never signal a PID/PGID again after exit.
    try { child.kill("SIGKILL"); } catch {}
    result = await waitForFirefoxTreeExit(firefox, 3_500);
  }
  clearInterval(firefox.tracker.interval);
  if (!result.verified) {
    // Close our pipe handles so this harness can report uncertainty without waiting on an orphan.
    child.stdout?.destroy();
    child.stderr?.destroy();
  }
  return result;
}

function requestFirefoxStop(firefox) {
  if (firefox && childHandleIsLive(firefox.child)) {
    try { firefox.child.kill("SIGTERM"); } catch {}
  }
}

function launchFirefox(profileDirectory, deadline) {
  const args = [
    "--headless",
    "--no-remote",
    "--profile", profileDirectory,
    "--remote-debugging-port", "0",
  ];
  const child = spawn("firefox", args, {
    detached: true,
    stdio: ["ignore", "ignore", "pipe"],
    env: { ...process.env, MOZ_HEADLESS: "1" },
  });
  const tracker = { rootPid: child.pid, groupId: child.pid, pids: new Map(), overflow: false, interval: null };
  observeProcessTree(tracker);
  tracker.interval = setInterval(() => observeProcessTree(tracker), 25);
  tracker.interval.unref();
  let stderrBytes = 0;
  let pending = "";
  let debugUrl;
  let rejectStartup;
  const startup = new Promise((resolve, reject) => {
    rejectStartup = reject;
    child.stderr.on("data", chunk => {
      stderrBytes += chunk.length;
      const text = chunk.toString("utf8");
      pending = (pending + text).slice(-8192);
      const match = /WebDriver BiDi listening on (ws:\/\/127\.0\.0\.1:\d+)/.exec(pending);
      if (match && !debugUrl) {
        debugUrl = `${match[1]}/session`;
        resolve(debugUrl);
      }
      // Continue draining but never retain unbounded browser diagnostics.
      if (stderrBytes > MAX_BROWSER_STDERR_BYTES) pending = pending.slice(-1024);
    });
    child.once("error", () => reject(new Error("Firefox failed to start.")));
    child.once("exit", () => {
      if (!debugUrl) reject(new Error("Firefox exited before exposing its loopback BiDi endpoint."));
    });
  });
  return { child, tracker, ready: deadline.wait(startup, "Firefox BiDi startup", 15_000) };
}

class BidiClient {
  constructor(url, deadline) {
    this.url = url;
    this.deadline = deadline;
    this.socket = null;
    this.nextId = 1;
    this.pending = new Map();
    this.exceptions = [];
    this.overflowed = false;
  }

  async connect() {
    if (typeof WebSocket !== "function") throw new Error("Node's native WebSocket API is unavailable.");
    const socket = new WebSocket(this.url);
    this.socket = socket;
    socket.addEventListener("message", event => this.#onMessage(event.data));
    socket.addEventListener("close", () => this.#rejectPending(new Error("BiDi connection closed.")));
    socket.addEventListener("error", () => this.#rejectPending(new Error("BiDi connection failed.")));
    await this.deadline.wait(new Promise((resolve, reject) => {
      socket.addEventListener("open", resolve, { once: true });
      socket.addEventListener("error", () => reject(new Error("Could not connect to loopback BiDi.")), { once: true });
    }), "BiDi websocket connect", 8_000);
  }

  #onMessage(data) {
    const text = typeof data === "string" ? data : "";
    if (Buffer.byteLength(text, "utf8") > MAX_BIDI_MESSAGE_BYTES) {
      this.overflowed = true;
      this.#rejectPending(new Error("BiDi message exceeded the size bound."));
      try { this.socket.close(); } catch {}
      return;
    }
    let message;
    try { message = JSON.parse(text); } catch {
      this.#rejectPending(new Error("BiDi returned invalid JSON."));
      return;
    }
    if (Number.isSafeInteger(message.id)) {
      const pending = this.pending.get(message.id);
      if (!pending) return;
      this.pending.delete(message.id);
      clearTimeout(pending.timer);
      if (message.type === "success") pending.resolve(message.result || {});
      else {
        const code = String(message.error || "unknown error").replace(/[^a-zA-Z0-9 _-]/g, "").slice(0, 64);
        const detail = String(message.message || "").replace(/[\r\n\0]/g, " ").slice(0, 256);
        pending.reject(new Error(`BiDi command failed: ${pending.method} (${code})${detail ? `: ${detail}` : ""}.`));
      }
      return;
    }
    if (message.method === "log.entryAdded") this.#recordLog(message.params);
  }

  #recordLog(params) {
    if (!params || this.exceptions.length >= MAX_REPORTED_EXCEPTIONS) return;
    // BiDi log.entryAdded carries the log entry object directly in params.
    const entry = params;
    const isError = entry.type === "javascript" || entry.level === "error";
    if (!isError) return;
    this.exceptions.push({
      type: String(entry.type || "unknown").slice(0, 32),
      level: String(entry.level || "unknown").slice(0, 32),
      method: String(entry.method || "").slice(0, 32),
      text: String(entry.text || "").replace(/[\r\n\0]/g, " ").slice(0, MAX_EXCEPTION_TEXT),
      line: Number.isSafeInteger(entry.lineNumber) ? entry.lineNumber : null,
    });
  }

  #rejectPending(error) {
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timer);
      pending.reject(error);
    }
    this.pending.clear();
  }

  command(method, params = {}, capMs = 10_000) {
    const id = this.nextId++;
    const timeoutMs = Math.min(this.deadline.remaining(method), capMs);
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`Timed out during BiDi command ${method}.`));
      }, timeoutMs);
      this.pending.set(id, { resolve, reject, timer, method });
      try {
        this.socket.send(JSON.stringify({ id, method, params }));
      } catch {
        clearTimeout(timer);
        this.pending.delete(id);
        reject(new Error(`Could not send BiDi command ${method}.`));
      }
    });
  }

  async evaluate(context, expression) {
    const result = await this.command("script.evaluate", {
      expression,
      target: { context },
      awaitPromise: true,
      resultOwnership: "none",
    });
    const remote = result && result.result;
    if (!remote || remote.type !== "string" || typeof remote.value !== "string") {
      throw new Error("Dashboard probe evaluation did not return a string.");
    }
    return JSON.parse(remote.value);
  }

  close() {
    this.#rejectPending(new Error("BiDi client closed."));
    if (this.socket && this.socket.readyState < WebSocket.CLOSING) this.socket.close();
  }
}

const KEY = {
  ArrowLeft: "\uE012",
  ArrowRight: "\uE014",
  Home: "\uE011",
  End: "\uE010",
};

const STATE_EXPRESSION = `JSON.stringify((() => {
  const tabs = [...document.querySelectorAll('[role="tab"]')];
  const selected = tabs.find(tab => tab.getAttribute('aria-selected') === 'true');
  const panel = selected && document.getElementById(selected.getAttribute('aria-controls'));
  const width = document.documentElement.clientWidth;
  const cards = [...(panel?.querySelectorAll('.lane-card') || [])];
  const outside = cards.filter(card => {
    const rect = card.getBoundingClientRect();
    return rect.left < -1 || rect.right > width + 1;
  }).length;
  const bodyOverflow = document.documentElement.scrollWidth > width + 1 || document.body.scrollWidth > width + 1;
  const laneOverflow = [...document.querySelectorAll('.lane-card')].some(card => card.scrollWidth > card.clientWidth + 2);
  return {
    viewport: { width: innerWidth, height: innerHeight, clientWidth: width },
    selected: selected?.id || null,
    focused: document.activeElement?.id || null,
    panel: panel?.id || null,
    mediaCards: document.getElementById('media-list')?.childElementCount || 0,
    decisionCards: document.getElementById('decision-list')?.childElementCount || 0,
    dataMedia: window.L3MS_DASHBOARD?.media?.length || 0,
    dataDecisions: window.L3MS_DASHBOARD?.decisions?.length || 0,
    bodyOverflow,
    laneOverflow,
    outsideCards: outside,
    scrollWidth: document.documentElement.scrollWidth,
    tabCount: tabs.length
  };
})())`;

async function currentState(client, context) {
  return client.evaluate(context, STATE_EXPRESSION);
}

async function clickTab(client, context, id) {
  await client.evaluate(context, `JSON.stringify((() => {
    const element = document.getElementById(${JSON.stringify(id)});
    if (element) element.scrollIntoView({ block: 'center', inline: 'nearest', behavior: 'instant' });
    return Boolean(element);
  })())`);
  const bounds = await client.evaluate(context, `JSON.stringify((() => {
    const element = document.getElementById(${JSON.stringify(id)});
    if (!element) return null;
    const rect = element.getBoundingClientRect();
    return { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2, width: rect.width, height: rect.height };
  })())`);
  assert.ok(bounds && bounds.width > 0 && bounds.height > 0, `tab ${id} is not visibly clickable`);
  await client.command("input.performActions", {
    context,
    actions: [{
      type: "pointer",
      id: "dashboard-probe-pointer",
      parameters: { pointerType: "mouse" },
      actions: [
        { type: "pointerMove", x: Math.round(bounds.x), y: Math.round(bounds.y), duration: 0, origin: "viewport" },
        { type: "pointerDown", button: 0 },
        { type: "pointerUp", button: 0 },
      ],
    }],
  });
  await client.command("input.releaseActions", { context });
}

async function pressKey(client, context, key) {
  await client.command("input.performActions", {
    context,
    actions: [{
      type: "key",
      id: "dashboard-probe-keyboard",
      actions: [
        { type: "keyDown", value: KEY[key] },
        { type: "keyUp", value: KEY[key] },
      ],
    }],
  });
  await client.command("input.releaseActions", { context });
}

async function setViewport(client, context, width, height) {
  await client.command("browsingContext.setViewport", {
    context,
    viewport: { width, height },
    devicePixelRatio: 1,
  });
}

async function saveScreenshot(client, context, outputDirectory, name) {
  const result = await client.command("browsingContext.captureScreenshot", {
    context,
    origin: "document",
  }, 12_000);
  const encoded = result && result.data;
  assert.equal(typeof encoded, "string", "Firefox did not return a screenshot");
  assert.ok(encoded.length <= Math.ceil(MAX_SCREENSHOT_BYTES * 4 / 3) + 8, "screenshot exceeded encoded size bound");
  const image = Buffer.from(encoded, "base64");
  assert.ok(image.length > 8 && image.length <= MAX_SCREENSHOT_BYTES, "screenshot exceeded decoded size bound");
  assert.equal(image.subarray(0, 8).toString("hex"), "89504e470d0a1a0a", "screenshot is not PNG data");
  const output = path.join(outputDirectory, name);
  fs.writeFileSync(output, image, { flag: "wx", mode: 0o600 });
  return output;
}

async function main() {
  parseArgs(process.argv.slice(2));
  const deadline = makeDeadline();
  fs.mkdirSync(TOOLKIT_ROOT, { recursive: true, mode: 0o700 });
  const profileDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "l3ms-dashboard-firefox-"));
  fs.chmodSync(profileDirectory, 0o700);
  const outputDirectory = fs.mkdtempSync(path.join(TOOLKIT_ROOT, "dashboard-probe-"));
  fs.chmodSync(outputDirectory, 0o700);

  let staticServer;
  let firefox;
  let bidi;
  let lastState;
  let pass = false;
  let thrown;
  let firefoxCleanup = { verified: false, remaining: 0 };
  let profileRetained = false;
  let logFixtureObserved = false;
  let watchdogFired = false;
  const watchdog = setTimeout(() => {
    watchdogFired = true;
    try { bidi?.close(); } catch {}
    requestFirefoxStop(firefox);
    try { staticServer?.server?.closeAllConnections(); } catch {}
    process.exitCode = 1;
  }, OUTER_DEADLINE_MS - CLEANUP_BUDGET_MS);

  try {
    fs.writeFileSync(path.join(profileDirectory, "user.js"), [
      'user_pref("app.update.enabled", false);',
      'user_pref("app.update.auto", false);',
      'user_pref("toolkit.telemetry.enabled", false);',
      'user_pref("datareporting.healthreport.uploadEnabled", false);',
      'user_pref("browser.shell.checkDefaultBrowser", false);',
      'user_pref("browser.newtabpage.enabled", false);',
      'user_pref("network.dns.disablePrefetch", true);',
      'user_pref("network.prefetch-next", false);',
      'user_pref("network.captive-portal-service.enabled", false);',
    ].join("\n"), { flag: "wx", mode: 0o600 });

    staticServer = await startStaticServer(deadline);
    firefox = launchFirefox(profileDirectory, deadline);
    firefox.url = await firefox.ready;
    bidi = new BidiClient(firefox.url, deadline);
    await bidi.connect();
    const session = await bidi.command("session.new", {
      capabilities: { alwaysMatch: { browserName: "firefox" } },
    }, 10_000);
    assert.ok(session.sessionId, "Firefox did not create a BiDi session");
    await bidi.command("session.subscribe", { events: ["log.entryAdded"] });
    const tree = await bidi.command("browsingContext.getTree", {});
    const context = tree.contexts?.[0]?.context;
    assert.equal(typeof context, "string", "Firefox did not expose a top-level browsing context");
    await bidi.command("browsingContext.navigate", {
      context,
      url: `${staticServer.url}probe-blank`,
      wait: "complete",
    }, 8_000);

    // Prove the event subscriber sees both console errors and uncaught exceptions
    // on a plain local page before the dashboard loads, then clear only the fixture.
    await bidi.evaluate(context, `JSON.stringify((() => {
      console.error('controlled dashboard probe console.error fixture');
      setTimeout(() => { throw new Error('controlled dashboard probe uncaught fixture'); }, 0);
      return 'scheduled';
    })())`);
    const fixtureDeadline = Date.now() + Math.min(2_500, deadline.remaining("BiDi log event fixture"));
    const fixtureSeen = () => ({
      console: bidi.exceptions.some(entry => entry.type === "console" && entry.method === "error" && entry.level === "error"),
      javascript: bidi.exceptions.some(entry => entry.type === "javascript" && entry.level === "error"),
    });
    let seen = fixtureSeen();
    while ((!seen.console || !seen.javascript) && Date.now() < fixtureDeadline) {
      await sleep(25);
      seen = fixtureSeen();
    }
    assert.equal(seen.console, true, "BiDi did not report the controlled console.error fixture");
    assert.equal(seen.javascript, true, "BiDi did not report the controlled uncaught exception fixture");
    logFixtureObserved = true;
    await sleep(Math.min(50, deadline.remaining("clear controlled log fixture")));
    bidi.exceptions.length = 0;

    await bidi.command("browsingContext.navigate", {
      context,
      url: staticServer.url,
      wait: "complete",
    }, 12_000);

    let state = await currentState(bidi, context);
    lastState = state;
    assert.equal(state.tabCount, 6, "expected all six dashboard tabs");
    assert.ok(state.mediaCards > 0 && state.decisionCards > 0, "media or decision cards did not render");
    assert.equal(state.mediaCards, state.dataMedia, "media card count differs from the manifest data");
    assert.equal(state.decisionCards, state.dataDecisions, "decision card count differs from the manifest data");

    // Exercise actual pointer and keyboard input through WebDriver BiDi.
    await clickTab(bidi, context, "media-tab");
    state = await currentState(bidi, context);
    lastState = state;
    assert.equal(state.selected, "media-tab");
    assert.equal(state.focused, "media-tab");
    await pressKey(bidi, context, "ArrowRight");
    state = await currentState(bidi, context);
    lastState = state;
    assert.equal(state.selected, "decisions-tab", "ArrowRight did not move selection");
    assert.equal(state.focused, "decisions-tab", "ArrowRight did not move focus");
    await pressKey(bidi, context, "Home");
    state = await currentState(bidi, context);
    lastState = state;
    assert.equal(state.selected, "models-tab", "Home did not select the first tab");
    assert.equal(state.focused, "models-tab");
    await pressKey(bidi, context, "End");
    state = await currentState(bidi, context);
    lastState = state;
    assert.equal(state.selected, "archive-tab", "End did not select the last tab");
    assert.equal(state.focused, "archive-tab");

    await setViewport(bidi, context, 1440, 1200);
    for (const [id, label] of [["media-tab", "media"], ["decisions-tab", "decisions"]]) {
      await clickTab(bidi, context, id);
      state = await currentState(bidi, context);
      lastState = state;
      assert.equal(state.selected, id);
      assert.equal(state.focused, id);
      assert.equal(state.bodyOverflow, false, `${label} view has document-level horizontal overflow at desktop width`);
      assert.equal(state.laneOverflow, false, `${label} cards have internal horizontal overflow at desktop width`);
      assert.equal(state.outsideCards, 0, `${label} cards extend beyond the desktop viewport`);
      await saveScreenshot(bidi, context, outputDirectory, `desktop-${label}.png`);
    }

    await setViewport(bidi, context, 390, 844);
    for (const [id, label] of [["media-tab", "media"], ["decisions-tab", "decisions"]]) {
      await clickTab(bidi, context, id);
      state = await currentState(bidi, context);
      lastState = state;
      assert.equal(state.viewport.width, 390, "mobile viewport width did not apply");
      assert.equal(state.selected, id);
      assert.equal(state.focused, id);
      assert.equal(state.bodyOverflow, false, `${label} view has document-level horizontal overflow at mobile width`);
      assert.equal(state.laneOverflow, false, `${label} cards have internal horizontal overflow at mobile width`);
      assert.equal(state.outsideCards, 0, `${label} cards extend beyond the mobile viewport`);
      await saveScreenshot(bidi, context, outputDirectory, `mobile-${label}.png`);
    }

    await sleep(Math.min(150, deadline.remaining("console exception drain")));
    assert.deepEqual(bidi.exceptions, [], `browser reported console exceptions: ${JSON.stringify(bidi.exceptions)}`);
    assert.equal(bidi.overflowed, false, "BiDi output exceeded its message bound");
    for (const pathname of staticServer.seenPaths) {
      assert.ok(ALLOWED_FILES.has(pathname), `unexpected local path requested: ${pathname}`);
    }
    pass = true;
  } catch (error) {
    thrown = error;
  } finally {
    clearTimeout(watchdog);
    try { bidi?.close(); } catch {}
    try {
      firefoxCleanup = await stopFirefox(firefox);
      if (!firefoxCleanup.verified) {
        profileRetained = true;
        pass = false;
        if (!thrown) thrown = new Error("Firefox cleanup is uncertain; the isolated profile was retained.");
      }
    } catch (error) {
      profileRetained = true;
      pass = false;
      if (!thrown) thrown = new Error(`Firefox cleanup is uncertain; profile retained (${shortError(error)}).`);
    }
    try {
      if (staticServer) await Promise.race([staticServer.close(), sleep(1_500)]);
    } catch (error) { if (!thrown) thrown = error; pass = false; }
    if (firefoxCleanup.verified) {
      try { fs.rmSync(profileDirectory, { recursive: true, force: true }); }
      catch (error) { if (!thrown) thrown = error; pass = false; }
    }
  }

  if (!pass) {
    process.stderr.write(`Browser dashboard probe failed: ${shortError(thrown || new Error("unknown failure"))}\n`);
    if (bidi?.exceptions?.length) process.stderr.write(`Bounded browser exceptions: ${JSON.stringify(bidi.exceptions).slice(0, 4096)}\n`);
    process.stderr.write(`Artifacts retained in ${outputDirectory}\n`);
    if (profileRetained) process.stderr.write(`Isolated Firefox profile retained because process cleanup was not verified: ${profileDirectory}\n`);
    if (watchdogFired) process.stderr.write("The probe-work deadline fired; cleanup used the reserved outer-deadline window without signaling a reused PID/group.\n");
    process.exitCode = 1;
    return;
  }

  const report = {
    result: "passed",
    browser: "Firefox headless via native Node WebSocket/WebDriver BiDi",
    interactions: ["real pointer click", "ArrowRight", "Home", "End", "focus assertions"],
    viewportWidths: [1440, 390],
    cards: { media: lastState.mediaCards, decisions: lastState.decisionCards },
    logEntryFixture: logFixtureObserved ? "console.error and unhandled JavaScript error events verified" : "not run",
    consoleExceptions: 0,
    childAndProcessGroupCleanup: firefoxCleanup.verified ? "verified" : "uncertain",
    profileRemoved: !profileRetained,
    localRequests: [...staticServer.seenPaths].sort(),
    screenshots: [
      "desktop-media.png",
      "desktop-decisions.png",
      "mobile-media.png",
      "mobile-decisions.png",
    ].map(name => path.relative(REPO_ROOT, path.join(outputDirectory, name))),
  };
  process.stdout.write(`${JSON.stringify(report)}\n`);
}

main().catch(error => {
  process.stderr.write(`Browser dashboard probe could not start: ${shortError(error)}\n`);
  process.exitCode = 1;
});
