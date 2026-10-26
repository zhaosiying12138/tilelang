(function () {
  "use strict";

  function showError(element, message, source) {
    element.classList.add("math-error");
    element.textContent = "公式渲染失败：" + message + "；原始公式：" + source;
  }

  function renderAllMath() {
    var elements = document.querySelectorAll(".math[data-display]");
    if (!elements.length) {
      return;
    }
    if (typeof window.katex === "undefined" || typeof window.katex.render !== "function") {
      elements.forEach(function (element) {
        showError(element, "内嵌 KaTeX 未加载", element.textContent);
      });
      return;
    }
    elements.forEach(function (element) {
      var source = element.textContent;
      try {
        window.katex.render(source, element, {
          displayMode: element.getAttribute("data-display") === "true",
          throwOnError: true,
          strict: "warn",
          output: "html"
        });
      } catch (error) {
        showError(element, error && error.message ? error.message : String(error), source);
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", renderAllMath, { once: true });
  } else {
    renderAllMath();
  }
}());
