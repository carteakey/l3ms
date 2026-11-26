// Execute the shipped dashboard JavaScript against a small DOM fixture.
// This tests behavior, not browser layout/accessibility or mobile rendering.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const html = fs.readFileSync(path.join(__dirname, "index.html"), "utf8");
const elements = [];
const byId = new Map();
const listeners = new Map();
let document;

function toCamel(name) {
  return name.replace(/-([a-z])/g, (_, ch) => ch.toUpperCase());
}

class Element {
  constructor(attributes) {
    this.attributes = new Map(attributes);
    this.id = this.attributes.get("id");
    this.dataset = Object.fromEntries(attributes
      .filter(([key]) => key.startsWith("data-"))
      .map(([key, value]) => [toCamel(key.slice(5)), value]));
    this.classes = new Set((this.attributes.get("class") || "").split(/\s+/).filter(Boolean));
    this.classList = {
      contains: name => this.classes.has(name),
      toggle: (name, force) => {
        const active = force === undefined ? !this.classes.has(name) : force;
        if (active) this.classes.add(name); else this.classes.delete(name);
        return active;
      },
    };
    this.tabIndex = Number(this.attributes.get("tabindex") ?? -1);
    this.handlers = new Map();
    this.innerHTML = "";
    this.textContent = "";
    this.value = "";
    this.hidden = false;
  }
  setAttribute(name, value) { this.attributes.set(name, value); }
  hasAttribute(name) { return this.attributes.has(name); }
  toggleAttribute(name, force) {
    const present = force === undefined ? !this.attributes.has(name) : force;
    if (present) this.attributes.set(name, ""); else this.attributes.delete(name);
    return present;
  }
  addEventListener(type, handler) { this.handlers.set(type, handler); }
  focus() { document.activeElement = this; }
  closest(selector) {
    const match = /^\[data-([a-z-]+)\]$/.exec(selector);
    return match && this.attributes.has(`data-${match[1]}`) ? this : null;
  }
}

for (const tag of html.matchAll(/<[a-z][a-z0-9]*\b([^>]*)>/gi)) {
  const attributes = [...tag[1].matchAll(/([a-z][a-z0-9-]*)="([^"]*)"/gi)]
    .map(match => [match[1], match[2]]);
  const element = new Element(attributes);
  elements.push(element);
  if (element.id) byId.set(element.id, element);
}
const queryAll = selector => {
  assert.match(selector, /^\.[a-z-]+$/, `unexpected fixture selector ${selector}`);
  return elements.filter(element => element.classes.has(selector.slice(1)));
};
document = {
  activeElement: null,
  getElementById: id => {
    assert.ok(byId.has(id), `missing dashboard element ${id}`);
    return byId.get(id);
  },
  querySelectorAll: queryAll,
  querySelector: selector => queryAll(selector)[0] || null,
  addEventListener: (type, handler) => listeners.set(type, handler),
};
const context = vm.createContext({
  document,
  window: { matchMedia: () => ({ matches: false }) },
  navigator: { clipboard: { writeText: async () => {} } },
  CSS: { escape: value => value },
  setTimeout,
  console,
});
vm.runInContext(fs.readFileSync(path.join(__dirname, "generated-models.js"), "utf8"), context);
const scripts = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)]
  .map(match => match[1]).filter(script => script.trim());
assert.equal(scripts.length, 1, "review the fixture when dashboard scripts change");
vm.runInContext(scripts[0], context, { filename: "dashboard-inline.js", timeout: 5000 });

// 1. Five crisp tabs + keyboard navigation
const tabs = queryAll(".tab");
const panels = queryAll(".tab-panel");
assert.equal(tabs.length, 5);
assert.equal(panels.length, 5);
assert.deepEqual(tabs.map(t => t.dataset.tab), ["stack", "engines", "roofline", "hardware", "archive"]);

function assertSelected(index) {
  tabs.forEach((tab, position) => {
    assert.equal(tab.classList.contains("active"), position === index);
    assert.equal(tab.attributes.get("aria-selected"), String(position === index));
    assert.equal(tab.tabIndex, position === index ? 0 : -1);
    assert.equal(tab.attributes.get("aria-controls"), `${tab.dataset.tab}-panel`);
  });
  panels.forEach(panel => {
    assert.equal(panel.classList.contains("active"), panel.id === `${tabs[index].dataset.tab}-panel`);
    assert.equal(panel.attributes.get("aria-labelledby"), panel.id.replace(/-panel$/, "-tab"));
  });
}
function key(k, shouldPrevent = true) {
  let prevented = false;
  queryAll(".tabs")[0].handlers.get("keydown")({ key: k, preventDefault: () => { prevented = true; } });
  assert.equal(prevented, shouldPrevent);
}

assertSelected(0);
tabs[1].focus();
listeners.get("click")({ target: tabs[1] });
assertSelected(1);
key("ArrowRight");
assertSelected(2);
assert.equal(document.activeElement, tabs[2]);
key("Home");
assertSelected(0);
key("ArrowLeft");
assertSelected(4);
key("ArrowRight");
assertSelected(0);
key("End");
assertSelected(4);
key("a", false);
assertSelected(4);
panels[4].focus();
key("ArrowLeft", false);
assertSelected(4);

// 2. Stack catalog rendering across all lanes
const dash = context.window.L3MS_DASHBOARD;
assert.equal((byId.get("spotlight-grid").innerHTML.match(/<article\b/g) || []).length, dash.spotlight.length);
assert.equal((byId.get("media-list").innerHTML.match(/<article\b/g) || []).length, dash.media.length);
assert.equal((byId.get("decision-list").innerHTML.match(/<article\b/g) || []).length, dash.decisions.length);
assert.equal((byId.get("unmeasured-list").innerHTML.match(/<article\b/g) || []).length, dash.unmeasured.length);
assert.equal((byId.get("embeddings-list").innerHTML.match(/<article\b/g) || []).length, dash.embeddings.length);
assert.ok(byId.get("unmeasured-list").innerHTML.includes("not measured"));
assert.ok(!dash.models.some(model => model.tps == null));
assert.ok(byId.get("media-list").innerHTML.includes("Breeze"));
assert.ok(byId.get("decision-list").innerHTML.includes("No measured decision benchmark published."));

// 3. Engine-aware portable recipe labels (Strata/PolyStrata vs llama.cpp)
const tableHtml = byId.get("model-table-body").innerHTML;
assert.ok(tableHtml.includes("Portable Strata / PolyStrata Recipe"));
assert.ok(tableHtml.includes("Portable llama.cpp command"));
assert.ok(tableHtml.includes("73.0"));
assert.ok(tableHtml.includes("77.5"));
assert.ok(tableHtml.includes("99.7"));

// 4. Unified filter pill interaction (e.g., co-resident)
const coResBtn = queryAll(".filter").find(el => el.dataset.tier === "co-resident");
assert.ok(coResBtn);
listeners.get("click")({ target: coResBtn });
assert.ok(byId.get("embeddings-list").innerHTML.includes("embeddinggemma-2"));
assert.equal(byId.get("unmeasured-section").hidden, true);
byId.get("stack-reset").handlers.get("click")();
assert.equal((byId.get("media-list").innerHTML.match(/<article\b/g) || []).length, dash.media.length);

// 5. Engines, Roofline SVGs, Dual-Track Evolution toggle, Hardware & Archive
assert.equal((byId.get("engines-grid").innerHTML.match(/<article\b/g) || []).length, dash.engines.engines.length);
assert.ok(byId.get("promoted-switches-body").innerHTML.includes("STRATA_HC_Q8=1"));
assert.ok(byId.get("promoted-switches-body").innerHTML.includes("STRATA_PREFILL_NVFP4=w4a8"));
assert.ok(byId.get("rejected-switches-body").innerHTML.includes("STRATA_ROUTE_PRIOR=1.5"));

assert.ok(byId.get("evolution-svg").innerHTML.includes("<svg"));
assert.ok(byId.get("evolution-svg").innerHTML.includes("73.0"));
const inklingBtn = queryAll(".track-btn").find(el => el.dataset.evoTrack === "inkling");
assert.ok(inklingBtn);
listeners.get("click")({ target: inklingBtn });
assert.ok(byId.get("evolution-svg").innerHTML.includes("77.5"));
assert.ok(byId.get("evolution-svg").innerHTML.includes("99.7 edit"));

assert.ok(byId.get("roofline-bars-svg").innerHTML.includes("455.8"));
assert.ok(byId.get("roofline-ctx-svg").innerHTML.includes("<polyline"));
assert.ok(byId.get("colab-section").innerHTML.includes("L4"));
assert.ok(byId.get("colab-section").innerHTML.includes("G4"));
assert.equal((byId.get("archive-list").innerHTML.match(/<article\b/g) || []).length, dash.archived.length);

// 6. Zero legacy L2M2 / L3DS strings and clean HTML escaping
assert.ok(!/l2m2|l3ds/i.test(html));
assert.equal(vm.runInContext('escapeHtml("<script>&\\\"\\\'")', context),
  "&lt;script&gt;&amp;&quot;&#39;");
console.log("Dashboard DOM fixtures passed: 5-tab navigation, unified filters, engines, roofline SVGs, dual-track evolution, and escaping.");
