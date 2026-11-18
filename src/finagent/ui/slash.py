"""채팅 입력창 슬래시 명령 자동완성 목록 컴포넌트."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Final

from streamlit.components.v2 import component

if TYPE_CHECKING:
    from collections.abc import Sequence

JS: Final = """
const CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ";

function initial(ch) {
  const code = ch.charCodeAt(0) - 0xac00;
  return code >= 0 && code < 11172 ? CHO[Math.floor(code / 588)] : ch;
}

function matches(word, query) {
  if (query.length > word.length) return false;
  for (let i = 0; i < query.length; i += 1) {
    const q = query[i];
    if (word[i] !== q && !(CHO.includes(q) && initial(word[i]) === q)) return false;
  }
  return true;
}

function setup() {
  const menu = { commands: [], selector: "", items: [], index: 0, target: null };
  const box = document.createElement("div");
  box.className = "fa-slash";
  box.setAttribute("role", "listbox");
  box.hidden = true;
  document.body.appendChild(box);

  const close = () => {
    box.hidden = true;
    menu.items = [];
  };

  const choose = (item) => {
    const area = menu.target;
    const value = item.hint ? `${item.command} ` : item.command;
    const set = Object.getOwnPropertyDescriptor(
      HTMLTextAreaElement.prototype, "value"
    ).set;
    menu.chosen = value;
    set.call(area, value);
    area.dispatchEvent(new Event("input", { bubbles: true }));
    area.focus();
    close();
  };

  const render = () => {
    box.replaceChildren(
      ...menu.items.map((item, index) => {
        const row = document.createElement("div");
        row.className = "fa-slash-item";
        row.setAttribute("role", "option");
        row.setAttribute("aria-selected", String(index === menu.index));
        const name = document.createElement("b");
        name.textContent = item.command;
        const title = document.createElement("span");
        title.textContent = item.title;
        row.append(name, title);
        if (item.hint) {
          const hint = document.createElement("small");
          hint.textContent = item.hint;
          row.append(hint);
        }
        row.onmousedown = (event) => event.preventDefault();
        row.onclick = () => choose(item);
        return row;
      })
    );
    const frame = menu.target.closest('[data-testid="stChatInput"]') || menu.target;
    const rect = frame.getBoundingClientRect();
    box.style.left = `${rect.left}px`;
    box.style.width = `${rect.width}px`;
    box.style.bottom = `${window.innerHeight - rect.top + 6}px`;
    box.hidden = false;
  };

  const update = (area) => {
    const value = area.value;
    if (value === menu.chosen || !value.startsWith("/") || /\\s/.test(value)) {
      close();
      return;
    }
    menu.chosen = null;
    const query = value.slice(1).toLowerCase();
    const found = menu.commands.filter((item) =>
      item.names.some((name) => matches(name, query))
    );
    if (!found.length) {
      close();
      return;
    }
    if (menu.target !== area || found.length !== menu.items.length) menu.index = 0;
    menu.target = area;
    menu.items = found;
    render();
  };

  const own = (event) => event.target.matches?.(menu.selector);

  document.addEventListener("input", (event) => {
    if (own(event)) update(event.target);
  }, true);
  document.addEventListener("compositionend", (event) => {
    if (!own(event)) return;
    if (menu.pending && menu.items.length) {
      menu.pending = false;
      const item = menu.items[menu.index];
      setTimeout(() => choose(item), 0);
    }
  }, true);
  document.addEventListener("focusout", (event) => {
    if (own(event)) close();
  }, true);
  document.addEventListener("keydown", (event) => {
    if (box.hidden || event.target !== menu.target) return;
    const step = { ArrowDown: 1, ArrowUp: -1 }[event.key];
    if (step) {
      const count = menu.items.length;
      menu.index = (menu.index + step + count) % count;
      render();
    } else if (event.key === "Enter" || event.key === "Tab") {
      if (event.isComposing) {
        menu.pending = true;
        return;
      }
      choose(menu.items[menu.index]);
    } else if (event.key === "Escape") {
      close();
    } else {
      return;
    }
    if (!event.isComposing) event.preventDefault();
    event.stopPropagation();
  }, true);
  return menu;
}

export default function (component) {
  const { data } = component;
  window.__finagentSlash = window.__finagentSlash || setup();
  window.__finagentSlash.commands = data.commands;
  window.__finagentSlash.selector = data.selector;
}
"""

_COMPONENT: Final = component("finagent_slash", js=JS)


@dataclass(frozen=True, slots=True)
class Command:
    command: str
    title: str
    hint: str
    names: tuple[str, ...]


def slash_menu(input_key: str, commands: Sequence[Command]) -> None:
    _ = _COMPONENT(
        data={
            "selector": f".st-key-{input_key} textarea",
            "commands": [asdict(command) for command in commands],
        },
        key=f"{input_key}_slash",
    )
