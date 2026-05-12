(() => {
  const dropzone     = document.getElementById("dropzone");
  const fileInput    = document.getElementById("fileInput");
  const browseBtn    = document.getElementById("browseBtn");
  const fileListEl   = document.getElementById("fileList");
  const interpretBtn = document.getElementById("interpretBtn");
  const btnLabel     = document.getElementById("btnLabel");
  const spinner      = document.getElementById("spinner");
  const resultsEl    = document.getElementById("results");

  /** @type {File[]} */
  let files = [];

  // ── File picking ──────────────────────────────────────────────────────────

  browseBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  dropzone.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", () => {
    addFiles([...fileInput.files]);
    fileInput.value = "";
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
  });

  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag-over"));

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
    addFiles([...e.dataTransfer.files]);
  });

  function addFiles(incoming) {
    const allowed = /\.(dcm|jpg|jpeg|png)$/i;
    incoming.filter((f) => allowed.test(f.name)).forEach((f) => {
      if (!files.find((x) => x.name === f.name && x.size === f.size)) {
        files.push(f);
      }
    });
    renderFileList();
  }

  function removeFile(index) {
    files.splice(index, 1);
    renderFileList();
  }

  function renderFileList() {
    fileListEl.innerHTML = "";
    files.forEach((f, i) => {
      const li = document.createElement("li");
      li.className = "file-item";
      li.innerHTML = `
        <span class="file-name">${escHtml(f.name)}</span>
        <span class="file-size">${fmtSize(f.size)}</span>
        <button class="file-remove" title="Remove" data-i="${i}">✕</button>`;
      fileListEl.appendChild(li);
    });

    fileListEl.querySelectorAll(".file-remove").forEach((btn) => {
      btn.addEventListener("click", () => removeFile(Number(btn.dataset.i)));
    });

    interpretBtn.disabled = files.length === 0;
  }

  // ── Interpret ─────────────────────────────────────────────────────────────

  interpretBtn.addEventListener("click", runInterpretation);

  async function runInterpretation() {
    if (files.length === 0) return;

    setRunning(true);
    resultsEl.innerHTML = "";

    const fd = new FormData();
    files.forEach((f) => fd.append("files", f));

    let response;
    try {
      response = await fetch("/api/interpret", { method: "POST", body: fd });
    } catch (err) {
      showError(`Network error: ${err.message}`);
      setRunning(false);
      return;
    }

    if (!response.ok) {
      showError(`Server error ${response.status}: ${await response.text()}`);
      setRunning(false);
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    /** @type {HTMLElement|null} */
    let activeBody = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop(); // keep incomplete last line

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        let event;
        try {
          event = JSON.parse(line.slice(6));
        } catch {
          continue;
        }
        activeBody = handleEvent(event, activeBody);
      }
    }

    setRunning(false);
  }

  /** @param {{type:string, index?:number, name?:string, text?:string, comparison?:boolean, message?:string}} event */
  function handleEvent(event, activeBody) {
    switch (event.type) {
      case "image_start": {
        if (activeBody) activeBody.classList.remove("streaming");
        const card = makeCard(event.name, `Image ${event.index + 1}`);
        resultsEl.appendChild(card.root);
        card.body.classList.add("streaming");
        return card.body;
      }

      case "token": {
        if (event.comparison) return activeBody; // comparison tokens go to comp card
        if (activeBody) activeBody.textContent += event.text;
        return activeBody;
      }

      case "image_done": {
        if (activeBody) activeBody.classList.remove("streaming");
        return null;
      }

      case "comparison_start": {
        if (activeBody) activeBody.classList.remove("streaming");
        const card = makeCard("Comparative Analysis", "Comparison", true);
        resultsEl.appendChild(card.root);
        card.body.classList.add("streaming");
        return card.body;
      }

      case "done": {
        if (activeBody) activeBody.classList.remove("streaming");
        return null;
      }

      case "error": {
        if (activeBody) activeBody.classList.remove("streaming");
        showError(event.message);
        return null;
      }

      default:
        // unknown event — handle comparison tokens that slip through
        if (event.comparison && activeBody) {
          activeBody.textContent += event.text ?? "";
        }
        return activeBody;
    }
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  function makeCard(filename, badge, isComparison = false) {
    const root = document.createElement("div");
    root.className = "result-card" + (isComparison ? " comparison" : "");

    const header = document.createElement("div");
    header.className = "card-header";
    header.innerHTML = `<span class="badge">${escHtml(badge)}</span>${escHtml(filename)}`;

    const body = document.createElement("div");
    body.className = "card-body";

    root.appendChild(header);
    root.appendChild(body);
    return { root, body };
  }

  function showError(msg) {
    const div = document.createElement("div");
    div.className = "error-card";
    div.textContent = `Error: ${msg}`;
    resultsEl.appendChild(div);
  }

  function setRunning(running) {
    interpretBtn.disabled = running;
    btnLabel.textContent = running ? "Interpreting…" : "Interpret";
    spinner.hidden = !running;
  }

  function fmtSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  }

  function escHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }
})();
