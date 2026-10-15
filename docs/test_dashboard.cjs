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

class Element {
  constructor(attributes) {
    this.attributes = new Map(attributes);
    this.id = this.attributes.get("id");
    this.dataset = Object.fromEntries(attributes
      .filter(([key]) => key.startsWith("data-"))
      .map(([key, value]) => [key.slice(5), value]));
    this.classes = new Set((this.attributes.get("class") || "").split(/\s+/));
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
  }
  setAttribute(name, value) { this.attributes.set(name, value); }
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

const tabs = queryAll(".tab");
const panels = queryAll(".tab-panel");
assert.equal(tabs.length, 7);
assert.equal(panels.length, 7);
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
function key(key, shouldPrevent = true) {
  let prevented = false;
  queryAll(".tabs")[0].handlers.get("keydown")({ key, preventDefault: () => { prevented = true; } });
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
assertSelected(6);
key("ArrowRight");
assertSelected(0);
key("End");
assertSelected(6);
key("a", false);
assertSelected(6);
panels[6].focus();
key("ArrowLeft", false);
assertSelected(6);

assert.equal((byId.get("media-list").innerHTML.match(/<article\b/g) || []).length,
  context.window.L3MS_DASHBOARD.media.length);
assert.equal((byId.get("decision-list").innerHTML.match(/<article\b/g) || []).length,
  context.window.L3MS_DASHBOARD.decisions.length);
assert.equal((byId.get("unmeasured-list").innerHTML.match(/<article\b/g) || []).length,
  context.window.L3MS_DASHBOARD.unmeasured.length);
assert.ok(byId.get("unmeasured-list").innerHTML.includes("not measured"));
assert.ok(!context.window.L3MS_DASHBOARD.models.some(model => model.tps == null));
assert.ok(byId.get("media-list").innerHTML.includes("Breeze"));
assert.ok(byId.get("decision-list").innerHTML.includes("No measured decision benchmark published."));
assert.equal(vm.runInContext('escapeHtml("<script>&\\\"\\\'")', context),
  "&lt;script&gt;&amp;&quot;&#39;");
console.log("Dashboard DOM fixtures passed: rendering, tab click, arrows, Home/End, focus and escaping.");
