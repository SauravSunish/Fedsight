"""Session-local presentation tokens shared by the browser and Plotly."""
from pathlib import Path

import streamlit as st


THEMES = {
    "case": {
        "bg-from": "#0A1220",
        "bg-to": "#0B1626",
        "sidebar": "#0D1624",
        "card": "#111F2F",
        "border": "#1F3446",
        "accent": "#3DD9C4",
        "warn": "#FFC36A",
        "critical": "#FF5475",
        "text": "#EFF6FC",
        "muted": "#B2C5D4",
        "on-accent": "#0A1220",
        "secondary": "#91BCFF",
        "tertiary": "#C5A4F3",
        "positive": "#9DD6AD",
    },
    "live": {
        "bg-from": "#120D1A",
        "bg-to": "#1A0F1F",
        "sidebar": "#150F1E",
        "card": "#1C1426",
        "border": "#33223F",
        "accent": "#FF5A5F",
        "warn": "#FFB020",
        "critical": "#FF2D55",
        "text": "#FFF4F7",
        "muted": "#CBB8CC",
        "on-accent": "#120D1A",
        "secondary": "#E7A4D6",
        "tertiary": "#A9B8FF",
        "positive": "#AACFA9",
    },
}


def tokens():
    return THEMES[st.session_state.get("mode", "case")]


def palette():
    t = tokens()
    return [t[k] for k in ("accent", "warn", "secondary", "tertiary", "positive")]


def translucent(color, alpha):
    r, g, b = bytes.fromhex(color[1:])
    return f"rgba({r},{g},{b},{alpha})"


def stylesheet():
    rules = []
    for mode, values in THEMES.items():
        selector = ":root" if mode == "case" else ':root[data-mode="live"], :root:has(.fs-theme[data-mode="live"])'
        declarations = " ".join(f"--{name}:{value};" for name, value in values.items())
        rules.append(f"{selector} {{{declarations}}}")
    return "\n".join(rules) + Path(__file__).with_name("theme.css").read_text(encoding="utf-8")


def apply_theme(mode):
    if mode not in THEMES:
        raise ValueError("Unknown workbench mode.")
    st.markdown(f'<style>{stylesheet()}</style><span hidden class="fs-theme" data-mode="{mode}"></span>',
                unsafe_allow_html=True)
    # Keep the mode allowlisted before writing it into the document.
    import inspect
    script = """
      const doc = win.document;
      doc.documentElement.dataset.mode = mode;
      if (!win.fsModeShortcut) {
        const triggerModeToggle = () => {
          const toggleInput = doc.querySelector(".st-key-mode_toggle label, .st-key-mode_toggle input");
          if (toggleInput) {
            toggleInput.click();
            return true;
          }
          const toggle = doc.querySelector(".st-key-mode_switch");
          const next = doc.documentElement.dataset.mode === "live" ? "CASE" : "LIVE";
          const button = toggle && [...toggle.querySelectorAll("button,label")].find(el => el.textContent.trim() === next);
          if (button) { button.click(); return true; }
          return false;
        };

        win.fsModeShortcut = event => {
          if (event.key.toLowerCase() !== "m" || event.repeat || event.isComposing ||
              event.ctrlKey || event.metaKey || event.altKey || event.shiftKey ||
              event.target.closest('input,textarea,select,[contenteditable=true],[role=dialog],[role=combobox]')) return;
          if (triggerModeToggle()) { event.preventDefault(); }
        };
        doc.addEventListener("keydown", win.fsModeShortcut);

        // Bind badge clicks so clicking CASE or LIVE triggers the smooth switch
        doc.addEventListener("click", e => {
          const badge = e.target.closest(".mode-badge-label");
          if (badge) {
            const isLive = doc.documentElement.dataset.mode === "live";
            const wantsLive = badge.classList.contains("live");
            const wantsCase = badge.classList.contains("case");
            if ((wantsLive && !isLive) || (wantsCase && isLive)) {
              triggerModeToggle();
            }
          }
        });
      }
    """
    inline = hasattr(st, "html") and "unsafe_allow_javascript" in inspect.signature(st.html).parameters
    target = "window" if inline else "window.parent"
    html = f'<script>((win, mode) => {{{script}}})({target}, "{mode}");</script>'
    if inline:
        st.html(html, unsafe_allow_javascript=True)
    else:
        from streamlit.components.v1 import html as component_html
        component_html(html, height=0, scrolling=False)


def plot(chart, *, container=st, **kwargs):
    t = tokens()
    chart.update_layout(template="plotly_dark", colorway=palette(), paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)", font_color=t["text"],
                        hoverlabel=dict(bgcolor=t["card"], bordercolor=t["muted"], font_color=t["text"]))
    chart.update_layout(legend=dict(bgcolor="rgba(0,0,0,0)", font_color=t["text"]))
    chart.update_traces(insidetextfont_color=t["on-accent"], selector=dict(type="pie"))
    chart.update_xaxes(gridcolor=t["border"], zerolinecolor=t["border"], linecolor=t["border"])
    chart.update_yaxes(gridcolor=t["border"], zerolinecolor=t["border"], linecolor=t["border"])
    return container.plotly_chart(chart, theme=None, **kwargs)
