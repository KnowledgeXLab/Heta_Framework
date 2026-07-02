(() => {
  const home = document.querySelector("[data-heta-home]");
  if (!home) {
    setupDocsHeader();
    return;
  }

  document.body.classList.add("heta-home-page");

  const root = home.querySelector("[data-heta-code-tabs]");
  if (!root) return;

  const title = root.querySelector("[data-heta-terminal-title]");
  const tabs = Array.from(root.querySelectorAll("[data-heta-code-tab]"));
  const panels = Array.from(root.querySelectorAll("[data-heta-code-panel]"));

  const activate = (name) => {
    for (const tab of tabs) {
      const active = tab.getAttribute("data-heta-code-tab") === name;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", active ? "true" : "false");
      if (active && title) {
        title.textContent = tab.getAttribute("data-title") || "";
      }
    }

    for (const panel of panels) {
      panel.classList.toggle(
        "is-active",
        panel.getAttribute("data-heta-code-panel") === name,
      );
    }
  };

  for (const tab of tabs) {
    tab.addEventListener("click", () => {
      const name = tab.getAttribute("data-heta-code-tab");
      if (name) activate(name);
    });
  }

})();

function setupDocsHeader() {
  document.body.classList.add("heta-docs-page");

  const header = document.querySelector(".md-header__inner");
  const isEnglish = document.documentElement.lang?.startsWith("en");
  setupCopyMarkdownButton(isEnglish);

  if (!header || header.querySelector(".heta-docs__links")) return;

  const links = document.createElement("nav");
  links.className = "heta-docs__links";
  links.setAttribute("aria-label", "Heta Framework 文档导航");
  links.innerHTML = `
    <a href="https://github.com/KnowledgeXLab/Heta_Framework">GitHub</a>
    <a href="https://knowledgexlab.github.io/">KnowledgeX Lab</a>
  `;

  const search = header.querySelector("[for='__search']");
  if (search) {
    header.insertBefore(links, search);
  } else {
    header.appendChild(links);
  }
}

function setupCopyMarkdownButton(isEnglish) {
  const source = document.querySelector("meta[name='heta-markdown-source']")?.content;
  const content = document.querySelector(".md-content__inner");
  if (!source || !content || content.querySelector(".heta-docs__page-actions")) return;

  const actions = document.createElement("div");
  actions.className = "heta-docs__page-actions";
  actions.innerHTML = `
    <button class="heta-docs__copy-markdown" type="button" data-heta-copy-markdown>
      ${isEnglish ? "Copy as Markdown" : "复制为 Markdown"}
    </button>
  `;

  const copyButton = actions.querySelector("[data-heta-copy-markdown]");
  if (copyButton) {
    copyButton.addEventListener("click", () => copyCurrentPageMarkdown(copyButton, isEnglish));
  }

  const firstHeading = content.querySelector("h1");
  if (firstHeading) {
    firstHeading.insertAdjacentElement("afterend", actions);
  } else {
    content.prepend(actions);
  }
}

async function copyCurrentPageMarkdown(button, isEnglish) {
  const source = document.querySelector("meta[name='heta-markdown-source']")?.content;
  if (!source) {
    setCopyButtonState(button, isEnglish ? "Unavailable" : "不可用");
    return;
  }

  const original = button.textContent;
  try {
    const response = await fetch(source, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Failed to load Markdown source: ${response.status}`);
    }
    const markdown = await response.text();
    await writeClipboard(markdown);
    setCopyButtonState(button, isEnglish ? "Copied" : "已复制", "success");
  } catch (error) {
    console.error(error);
    setCopyButtonState(button, isEnglish ? "Copy failed" : "复制失败", "error");
  } finally {
    window.setTimeout(() => {
      button.textContent = original;
      button.removeAttribute("data-copy-state");
    }, 1800);
  }
}

async function writeClipboard(text) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.left = "-9999px";
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand("copy");
  textarea.remove();
}

function setCopyButtonState(button, label, state) {
  button.textContent = label;
  button.setAttribute("data-copy-state", state);
}
