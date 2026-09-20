import { Controller } from "@hotwired/stimulus";

/**
 * Dialog explorer: loads the split list for the chosen dataset, pages through
 * rows, and swaps in the transcript. Every fragment is rendered server-side —
 * this controller only tracks which dataset/row is selected.
 */
export default class extends Controller {
  static targets = ["body", "splits", "config", "split"];

  static PAGE_SIZE = 10;

  connect() {
    this.dataset = null;
    this.offset = 0;
    this.selectedIndex = null;
    this.pending = {};
    // Capture the URL templates once, before any substitution. Rewriting the
    // attribute in place is a trap: after the first substitution the literal
    // `__DATASET__` is gone, so a second dataset can never be loaded.
    this.splitsTemplate = this.bodyTarget.dataset.splitsUrl;
    this.rowsTemplate = this.bodyTarget.dataset.rowsUrl;
  }

  loadDataset(event) {
    this.dataset = event.detail.dataset;
    this.offset = 0;

    // Clear the previous transcript immediately and stop any in-flight
    // analysis, whose transcript no longer matches the newly picked dataset.
    this.selectedIndex = null;
    this.stopStream();
    window.dispatchEvent(
      new CustomEvent("dialog:selected", {
        detail: { dialog: null, dataset: this.dataset },
      })
    );

    this.config = this.dataset.default_config || "";
    this.split = this.dataset.default_split || "";

    // Both fragments are loaded up front: the rows request carries
    // config/split explicitly, so it never depends on the splits fragment
    // having landed first.
    this.loadSplits();
    this.loadRows();
  }

  /** Ask the analysis controller to abort, if it is connected. */
  stopStream() {
    const element = document.querySelector('[data-controller~="analysis-stream"]');
    if (!element) return;
    this.application
      .getControllerForElementAndIdentifier(element, "analysis-stream")
      ?.stop();
  }

  reload() {
    if (this.hasConfigTarget) this.config = this.configTarget.value;
    if (this.hasSplitTarget) this.split = this.splitTarget.value;
    this.offset = 0;
    this.loadRows();
  }

  prev() {
    this.offset = Math.max(0, this.offset - this.constructor.PAGE_SIZE);
    this.loadRows();
  }

  next() {
    this.offset += this.constructor.PAGE_SIZE;
    this.loadRows();
  }

  select(event) {
    const index = Number(event.currentTarget.dataset.dialogIndex);
    this.selectedIndex = index;

    // Re-read the embedded payload right now: `loadRows` reads it after its
    // own await, so relying on the earlier value would hand the analysis
    // column a null dialog on the very first click.
    const dialogs = this.readDialogsFromRenderedCards();
    const dialog = dialogs.find((entry) => entry.index === index) ?? null;

    window.dispatchEvent(
      new CustomEvent("dialog:selected", {
        detail: { dialog, dataset: this.dataset, index },
      })
    );
    this.loadRows();
  }

  async loadSplits() {
    await this.swap(this.splitsTemplate.replace("__DATASET__", this.dataset.id), {
      target: this.splitsTarget,
      channel: "splits",
    });
  }

  async loadRows() {
    const params = new URLSearchParams({
      offset: String(this.offset),
      length: String(this.constructor.PAGE_SIZE),
    });
    if (this.config) params.set("config", this.config);
    if (this.split) params.set("split", this.split);
    if (this.selectedIndex !== null) params.set("selected", String(this.selectedIndex));

    await this.swap(
      `${this.rowsTemplate.replace("__DATASET__", this.dataset.id)}?${params.toString()}`,
      { target: this.bodyTarget, channel: "rows" }
    );
  }

  /** Read the dialog payloads the server embedded in the rendered fragment. */
  readDialogsFromRenderedCards() {
    const embedded = this.bodyTarget.querySelector("[data-dialogs-json]");
    if (!embedded) return [];
    // The JSON is the element's *text content* (a `<script type="application/json">`
    // block), not an attribute value.
    try {
      return JSON.parse(embedded.textContent || "[]");
    } catch {
      return [];
    }
  }

  /**
   * Fetch a fragment into `target`.
   *
   * `channel` matters: splits and rows are requested in parallel, so they need
   * independent sequence numbers. A single counter would make whichever
   * response arrived second look "stale" and silently drop the other.
   */
  async swap(url, { target, channel }) {
    const id = (this.pending[channel] = (this.pending[channel] || 0) + 1);

    let html;
    let ok = true;
    try {
      const response = await fetch(url, { headers: { Accept: "text/html" } });
      html = await response.text();
      ok = response.ok;
    } catch (error) {
      html = error.message;
      ok = false;
    }

    // A slow request must not clobber a newer one, and both a failure and a
    // stale response must be dropped silently rather than written elsewhere.
    if (id !== this.pending[channel]) return;

    target.innerHTML = ok
      ? html
      : `<div class="alert alert-error text-xs py-2 my-2">${html}</div>`;
  }
}
