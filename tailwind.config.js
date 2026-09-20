/** @type {import('tailwindcss').Config} */
module.exports = {
  // Templates are the only source of class names: the Stimulus controllers
  // toggle classes that also appear here, so everything they use must be
  // present in the scanned markup (see the safelist below for the toggled ones).
  content: [
    "./app/templates/**/*.html",
    "./app/static/js/**/*.js",
  ],
  theme: {
    extend: {
      colors: {
        // Indigo-forward, deliberately dark. Not grey.
        ink: {
          950: "#080b18",
          900: "#0d1226",
          850: "#121838",
          800: "#1a2149",
          700: "#252e63",
          600: "#39438a",
          500: "#5a66b4",
          400: "#8b95d6",
          300: "#b7bee8",
          200: "#d8dcf4",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
    },
  },
  plugins: [require("daisyui")],
  daisyui: {
    logs: false,
    // Values go through culori's `oklch()` parser, so they must be CSS colors
    // (hex is fine). A bare "L% C H" triplet is NOT valid CSS and is rejected.
    themes: [
      {
        "hf-dark": {
          "color-scheme": "dark",
          primary: "#6366f1", // indigo 500
          "primary-content": "#0d1226",
          secondary: "#a855f7", // violet 500
          "secondary-content": "#0d1226",
          accent: "#22d3ee", // cyan 400
          "accent-content": "#0d1226",
          neutral: "#1a2149",
          "neutral-content": "#d8dcf4",
          "base-100": "#0d1226", // page background — deep indigo
          "base-200": "#121838", // raised surfaces
          "base-300": "#1a2149", // borders, dividers
          "base-content": "#d8dcf4",
          info: "#38bdf8",
          success: "#34d399",
          warning: "#fbbf24",
          error: "#fb7185",
          "--rounded-box": "0.75rem",
          "--rounded-btn": "0.5rem",
          "--rounded-badge": "0.375rem",
          "--border-btn": "1px",
          "--animation-btn": "0.15s",
        },
      },
    ],
  },
  safelist: [
    // Classes the Stimulus controllers add/remove at runtime. Tailwind's
    // scanner would otherwise never see them, and the toggles would silently
    // do nothing in a compiled build.
    "border-primary",
    "bg-primary/10",
    "ring-1",
    "ring-primary/40",
    "border-base-300",
    "bg-base-200",
    "hover:border-base-content/40",
    "hover:bg-base-300",
    "bg-base-300",
    "text-base-content",
    "shadow-sm",
    "text-base-content/50",
    "hover:text-base-content",
  ],
};
