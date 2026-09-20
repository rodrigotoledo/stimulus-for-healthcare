import { Controller } from "@hotwired/stimulus";

/**
 * Analysis stream: POSTs the selected transcript and appends server-rendered
 * HTML fragments emitted over SSE. The client never assembles markup — it only
 * appends what the server sends.
 */
export default class extends Controller {
  static targets = [
    "model",
    "language",
    "lens",
    "lensHint",
    "temperature",
    "temperatureLabel",
    "runButton",
    "stopButton",
    "output",
    "meta",
    "error",
    "hint",
    "config",
  ];

  connect() {
    this.dialog = null;
    this.dataset = null;
    this.abort = null;

    // Single-quoted attribute, so `'` is what needs escaping before parsing.
    // Wrapped in try/catch because a broken attribute must not take out the
    // whole controller (which would silently disable the analysis panel).
    try {
      this.lenses = JSON.parse(this.configTarget.dataset.lenses || "[]");
    } catch {
      this.lenses = [];
    }
  }

  setDialog(event) {
    const { dialog, dataset } = event.detail;
    this.dialog = dialog;
    this.dataset = dataset;

    const armed = Boolean(dialog);
    this.runButtonTarget.disabled = !armed;
    this.hintTarget.classList.toggle("hidden", armed);
    this.metaTarget.textContent = armed
      ? `Alvo: ${dataset?.id} · linha #${dialog.index} · ${dialog.messages.length} mensagens`
      : "";
    this.errorTarget.innerHTML = "";
  }

  describeLens() {
    const lens = this.lenses.find((entry) => entry.id === this.lensTarget.value);
    this.lensHintTarget.textContent = lens ? lens.description : "";
  }

  updateTemperature() {
    this.temperatureLabelTarget.textContent = Number(
      this.temperatureTarget.value
    ).toFixed(1);
  }

  async run() {
    if (!this.dialog) return;

    this.errorTarget.innerHTML = "";
    this.outputTarget.innerHTML = "";
    this.metaTarget.textContent = "";
    this.outputTarget.classList.remove("hidden");
    this.runButtonTarget.disabled = true;
    this.stopButtonTarget.classList.remove("hidden");

    this.abort = new AbortController();

    try {
      const response = await fetch(this.configTarget.dataset.analyzeUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify({
          model: this.modelTarget.value,
          lens: this.lensTarget.value,
          language: this.languageTarget.value,
          temperature: Number(this.temperatureTarget.value),
          dataset: this.dataset?.id || null,
          dialog_index: this.dialog.index,
          messages: this.dialog.messages,
        }),
        signal: this.abort.signal,
      });

      if (!response.ok) {
        const body = await response.text();
        throw new Error(body || `HTTP ${response.status}`);
      }

      await this.consume(response.body.getReader());
    } catch (error) {
      if (error.name !== "AbortError") this.showError(error.message);
    } finally {
      this.finish();
    }
  }

  /** Parse `event:`/`data:` frames and append the HTML they carry. */
  async consume(reader) {
    const decoder = new TextDecoder();
    let buffer = "";

    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        this.dispatchFrame(buffer.slice(0, boundary));
        buffer = buffer.slice(boundary + 2);
        boundary = buffer.indexOf("\n\n");
      }
    }
    if (buffer.trim()) this.dispatchFrame(buffer);
  }

  dispatchFrame(frame) {
    let event = "message";
    const dataLines = [];

    for (const line of frame.split("\n")) {
      if (line.startsWith("event: ")) event = line.slice(7).trim();
      else if (line.startsWith("data: ")) dataLines.push(line.slice(6));
    }
    if (!dataLines.length) return;

    const raw = dataLines.join("\n");

    if (event === "fragment") {
      // The server re-renders the whole accumulated reply on every flush, so
      // this replaces rather than appends. That is what lets partially
      // streamed markdown (`**bo`) stay hidden until it completes.
      this.outputTarget.innerHTML = raw;
    } else if (event === "error") {
      this.showError(raw);
    } else if (event === "done") {
      this.metaTarget.textContent = raw;
    }
  }

  stop() {
    this.abort?.abort();
    this.finish();
  }

  finish() {
    this.runButtonTarget.disabled = !this.dialog;
    this.stopButtonTarget.classList.add("hidden");
  }

  showError(message) {
    this.errorTarget.innerHTML = `<div class="alert alert-error text-xs py-2 my-2">${message}</div>`;
  }
}
