const form = document.querySelector("#story-form");
const submit = document.querySelector("#submit");
const result = document.querySelector("#result");
const story = document.querySelector("#story");
const mode = document.querySelector("#mode");
const error = document.querySelector("#error");
const copy = document.querySelector("#copy");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  error.hidden = true;
  result.hidden = true;
  submit.disabled = true;
  submit.textContent = "ページをめくっています…";

  const payload = {
    name: document.querySelector("#name").value.trim(),
    place: document.querySelector("#place").value.trim(),
  };

  try {
    const response = await fetch("/api/story", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "生成できませんでした。");
    story.textContent = data.story;
    mode.textContent = data.mode === "demo" ? "見本モード（固定文）" : "この Mac で生成";
    result.hidden = false;
  } catch (reason) {
    error.textContent = reason.message;
    error.hidden = false;
  } finally {
    submit.disabled = false;
    submit.textContent = "一篇をひらく";
  }
});

copy.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(story.textContent);
    copy.textContent = "コピーしました";
    window.setTimeout(() => { copy.textContent = "物語をコピー"; }, 1600);
  } catch {
    copy.textContent = "選択してコピーしてください";
  }
});
