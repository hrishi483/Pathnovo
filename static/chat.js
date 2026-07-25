(() => {
  const pairSelect = document.getElementById("pair-select");
  const chatLog = document.getElementById("chat-log");
  const chatForm = document.getElementById("chat-form");
  const questionInput = document.getElementById("question-input");
  const sendButton = document.getElementById("send-button");
  const clearButton = document.getElementById("clear-button");
  const statusEl = document.getElementById("chat-status");
  const toolTrace = document.getElementById("tool-trace");

  function setStatus(message) {
    statusEl.textContent = message;
  }

  function appendBubble(role, text, className = role) {
    const bubble = document.createElement("div");
    bubble.className = `chat-bubble ${className}`;
    const roleLabel = document.createElement("span");
    roleLabel.className = "role";
    roleLabel.textContent = role;
    const body = document.createElement("div");
    body.textContent = text;
    bubble.append(roleLabel, body);
    chatLog.appendChild(bubble);
    chatLog.scrollTop = chatLog.scrollHeight;
    return bubble;
  }

  function renderToolTrace(toolCalls) {
    toolTrace.replaceChildren();
    if (!toolCalls || toolCalls.length === 0) {
      const empty = document.createElement("p");
      empty.className = "empty-trace";
      empty.textContent = "No tool calls for this answer.";
      toolTrace.appendChild(empty);
      return;
    }

    toolCalls.forEach((call, index) => {
      const details = document.createElement("details");
      details.className = "tool-call";
      details.open = index === 0;

      const summary = document.createElement("summary");
      summary.textContent = `${index + 1}. ${call.name}`;

      const argsPre = document.createElement("pre");
      argsPre.textContent = `Args\n${JSON.stringify(call.args, null, 2)}`;

      const resultPre = document.createElement("pre");
      resultPre.textContent = `Result\n${JSON.stringify(call.result, null, 2)}`;

      details.append(summary, argsPre, resultPre);
      toolTrace.appendChild(details);
    });
  }

  async function loadPairs() {
    const response = await fetch("/api/chat/pairs");
    if (!response.ok) {
      throw new Error(`Failed to load pairs (${response.status})`);
    }
    const data = await response.json();
    const pairs = data.pairs || [];
    pairSelect.replaceChildren();

    if (pairs.length === 0) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "No comparison pairs found";
      pairSelect.appendChild(option);
      pairSelect.disabled = true;
      setStatus("No artifact pairs under data/artifacts/delta. Run /compare first.");
      return;
    }

    pairs.forEach((pair, index) => {
      const option = document.createElement("option");
      option.value = pair.document_pair_id;
      option.textContent = `${pair.baseline_document_id.slice(0, 8)}… → ${pair.revision_document_id.slice(0, 8)}…`;
      if (index === 0) {
        option.selected = true;
      }
      pairSelect.appendChild(option);
    });
    pairSelect.disabled = false;
    setStatus("Ready. Ask a question about the selected document pair.");
  }

  chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = questionInput.value.trim();
    const documentPairId = pairSelect.value;
    if (!question || !documentPairId || sendButton.disabled) {
      return;
    }

    appendBubble("user", question);
    questionInput.value = "";
    sendButton.disabled = true;
    pairSelect.disabled = true;
    setStatus("Agent is thinking (tool loop may take a few seconds)...");

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          document_pair_id: documentPairId,
        }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = payload.detail || `Request failed (${response.status})`;
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }
      appendBubble("assistant", payload.answer || "(empty answer)");
      renderToolTrace(payload.tool_calls || []);
      setStatus(`Answered for pair ${payload.document_pair_id}.`);
    } catch (error) {
      appendBubble("assistant", error.message || String(error), "error");
      setStatus("Request failed.");
    } finally {
      sendButton.disabled = false;
      pairSelect.disabled = false;
      questionInput.focus();
    }
  });

  clearButton.addEventListener("click", () => {
    chatLog.replaceChildren();
    renderToolTrace([]);
    setStatus("Cleared. Ask another question.");
    questionInput.focus();
  });

  questionInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      chatForm.requestSubmit();
    }
  });

  loadPairs().catch((error) => {
    setStatus(error.message || String(error));
  });
})();
