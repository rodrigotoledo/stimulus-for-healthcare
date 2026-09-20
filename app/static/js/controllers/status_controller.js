import { Controller } from "@hotwired/stimulus";

/** Polls the health fragment so the header dot reflects Ollama's state. */
export default class extends Controller {
  static targets = ["badge"];

  connect() {
    this.url = this.badgeTarget.dataset.refreshUrl;
    this.interval = Number(this.badgeTarget.dataset.interval || 15000);
    this.timer = setInterval(() => this.refresh(), this.interval);
  }

  disconnect() {
    clearInterval(this.timer);
  }

  async refresh() {
    try {
      const response = await fetch(this.url, { headers: { Accept: "text/html" } });
      if (response.ok) this.badgeTarget.innerHTML = await response.text();
    } catch {
      /* transient network failure: keep the last known state */
    }
  }
}
