import { Controller } from "@hotwired/stimulus";

/**
 * Dataset picker: switches between the curated list and Hub search, renders
 * the server-rendered list fragment, and announces the selection so the
 * dialog and analysis columns can react.
 */
export default class extends Controller {
  static targets = ["list", "query", "searchForm", "tab", "error"];

  connect() {
    this.mode = "curated";
    this.selectedId = null;
  }

  switchTab(event) {
    const mode = event.currentTarget.dataset.tab;
    if (mode === this.mode) return;

    this.mode = mode;
    this.searchFormTarget.classList.toggle("hidden", mode !== "search");

    // Active tab: filled surface. Inactive: muted text.
    const active = ["bg-base-300", "text-base-content", "shadow-sm"];
    const inactive = ["text-base-content/50", "hover:text-base-content"];
    this.tabTargets.forEach((tab) => {
      const isActive = tab.dataset.tab === mode;
      tab.classList.remove(...(isActive ? inactive : active));
      tab.classList.add(...(isActive ? active : inactive));
    });
  }

  async search(event) {
    event.preventDefault();
    const query = this.queryTarget.value.trim();
    await this.loadFragment(
      `${this.listTarget.dataset.searchUrl}?q=${encodeURIComponent(query)}`
    );
  }

  /** Fired on `dataset:selected`, so the chosen card stays highlighted. */
  highlight(event) {
    this.selectedId = event.detail.datasetId;
    const on = ["border-primary", "bg-primary/10", "ring-1", "ring-primary/40"];
    const off = [
      "border-base-300",
      "bg-base-200",
      "hover:border-base-content/40",
      "hover:bg-base-300",
    ];

    this.listTarget.querySelectorAll("[data-dataset-id]").forEach((card) => {
      const isSelected = card.dataset.datasetId === this.selectedId;
      card.classList.remove(...(isSelected ? off : on));
      card.classList.add(...(isSelected ? on : off));
    });
  }

  select(event) {
    const card = event.currentTarget;
    const dataset = {
      id: card.dataset.datasetId,
      default_config: card.dataset.defaultConfig || null,
      default_split: card.dataset.defaultSplit || null,
      note: card.dataset.note || null,
      curated: card.dataset.curated === "true",
    };

    // Announce on window: the other columns listen for this instead of being
    // wired to this controller directly.
    window.dispatchEvent(
      new CustomEvent("dataset:selected", { detail: { dataset, datasetId: dataset.id } })
    );
  }

  async loadFragment(url) {
    this.errorTarget.innerHTML = "";
    try {
      const response = await fetch(url, { headers: { Accept: "text/html" } });
      const html = await response.text();
      if (!response.ok) throw new Error(html || `HTTP ${response.status}`);
      this.listTarget.innerHTML = html;
    } catch (error) {
      this.errorTarget.innerHTML = `<div class="alert alert-error text-xs py-2 my-2">${error.message}</div>`;
    }
  }
}
