"""The review window JS, run in Node.js (skipped without it; set OCR_TEST_NODE)."""

# The embedded JS keeps its own line lengths.
# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.owui.helpers import op, run_node
from tests.owui.test_draft_logic import ITEM, local_copy, server_draft

pytestmark = pytest.mark.p3


def core_script(body: str) -> str:
    return op.REVIEW_CORE_JS + "\n" + body


def test_sha256_matches_python(tmp_path: Path) -> None:
    samples = ["", "abc", "Привет, мир", "a" * 55, "b" * 56, "c" * 64, "d" * 1000, "𝑥 😀 \n\t"]
    result = run_node(
        core_script(f"console.log(JSON.stringify({json.dumps(samples)}.map(OCR_CORE.sha256Hex)));"),
        tmp_path,
    )
    assert result == [op.text_sha256(s) for s in samples]


def hint(start: int, end: int, hint_id: int = 1, state: str = "active") -> dict[str, Any]:
    return {"id": hint_id, "start": start, "end": end, "reasons": ["diff"], "state": state}


# (old text, new text, hint before, hint after) — section 8.3 rules.
SHIFT_CASES = [
    ("abc XYZ def", "Nabc XYZ def", hint(4, 7), hint(5, 8)),  # insert before: shifted
    ("abc XYZ def", "abc XYZ defN", hint(4, 7), hint(4, 7)),  # insert after: unchanged
    ("abc XYZ def", "abc XNYZ def", hint(4, 7), hint(4, 7, state="manual")),  # inside
    ("abc XYZ def", "abc NXYZ def", hint(4, 7), hint(5, 8)),  # exactly at the start: shifted
    ("abc XYZ def", "abc XYZN def", hint(4, 7), hint(4, 7)),  # exactly at the end: unchanged
    ("abc XYZ def", "c XYZ def", hint(4, 7), hint(2, 5)),  # deletion before
    ("abc XYZ def", "abc QQ def", hint(4, 7), hint(4, 7, state="manual")),  # replaced by paste
    ("ааааа", "аааааа", hint(2, 4), hint(2, 4)),  # repeated letters: prefix first
    ("ааааа", "аааа", hint(2, 4), hint(2, 4)),  # the deleted letter is taken from the end
    ("ааааа", "аааа", hint(3, 5), hint(3, 5, state="manual")),
    ("x XYZ", "𝑥x XYZ", hint(2, 5), hint(4, 7)),  # a non-BMP char counts as 2 UTF-16 units
]


def test_shift_hints_follows_section_8_3(tmp_path: Path) -> None:
    cases = [[old, new, [before]] for old, new, before, _ in SHIFT_CASES]
    result = run_node(
        core_script(
            f"const cases = {json.dumps(cases, ensure_ascii=False)};\n"
            "console.log(JSON.stringify(cases.map(c => OCR_CORE.shiftHints(c[2], c[0], c[1])[0])));"
        ),
        tmp_path,
    )
    assert result == [after for _, _, _, after in SHIFT_CASES]


def restore_cases() -> list[tuple[Any, Any]]:
    flight = {"expected_seq": 5, "rev": 7, "sha256": op.text_sha256("A")}
    return [
        (None, None),
        (local_copy("старое", base_seq=3, rev=5, sent_rev=5), server_draft("с сервера", 4)),
        (local_copy("дописал", base_seq=4, rev=9, sent_rev=7), server_draft("с сервера", 4)),
        (local_copy("B", base_seq=5, rev=9, sent_rev=6, in_flight=flight), server_draft("A", 6)),
        (local_copy("B", base_seq=5, rev=9, sent_rev=6, in_flight=flight), server_draft("C", 6)),
        (local_copy("мой", base_seq=3, rev=4, sent_rev=2), None),
        (None, server_draft("правка", 2, item_version="v1")),
        (None, server_draft("мой", 3, base_attempt_id=69, stale=True, stale_reason="keep_stale")),
        (local_copy("мой", base_seq=0, rev=2, sent_rev=0, base_attempt_id=69), None),
    ]


def test_restore_logic_matches_python_reference(tmp_path: Path) -> None:
    cases = restore_cases()
    payload = json.dumps([[ITEM, local, server] for local, server in cases], ensure_ascii=False)
    result = run_node(
        core_script(
            f"const cases = {payload};\n"
            "console.log(JSON.stringify(cases.map(c => ({"
            "choice: OCR_CORE.chooseDraftSource(c[1], c[2]),"
            "state: OCR_CORE.restoreState(c[0], c[1], c[2])}))));"
        ),
        tmp_path,
    )
    expected = [
        {
            "choice": op.choose_draft_source(local, server),
            "state": op.restore_state(ITEM, local, server),
        }
        for local, server in cases
    ]
    assert result == expected


def test_apply_draft_ack_matches_python_reference(tmp_path: Path) -> None:
    flight = {"expected_seq": 4, "rev": 5, "sha256": "x"}
    local = local_copy("текст", base_seq=4, rev=5, sent_rev=2, in_flight=flight)
    acks = [{"rev": 3, "seq": 5}, {"rev": 5, "seq": 6}, {"rev": None, "seq": 7}]
    result = run_node(
        core_script(
            f"const local = {json.dumps(local, ensure_ascii=False)};\n"
            f"const acks = {json.dumps(acks)};\n"
            "console.log(JSON.stringify(acks.map(a => OCR_CORE.applyDraftAck(local, a))));"
        ),
        tmp_path,
    )
    assert result == [op.apply_draft_ack(local, rev=a["rev"], seq=a["seq"]) for a in acks]


# A minimal DOM, enough to drive the overlay through one page.
FAKE_DOM = r"""
const listeners = {};
globalThis.window = globalThis;
globalThis.addEventListener = (t, fn) => { (listeners[t] = listeners[t] || []).push(fn); };
globalThis.removeEventListener = (t, fn) => { listeners[t] = (listeners[t] || []).filter(f => f !== fn); };
const store = new Map();
globalThis.localStorage = {
  getItem: k => (store.has(k) ? store.get(k) : null),
  setItem: (k, v) => store.set(k, String(v)),
  removeItem: k => store.delete(k),
  get length() { return store.size; },
  key: i => Array.from(store.keys())[i] || null,
};
function makeText(t) { return {nodeType: 3, textContent: String(t), children: [], parentNode: null}; }
function makeEl(tag) {
  const e = {tagName: tag.toUpperCase(), children: [], style: {}, attrs: {}, listeners: {},
    className: '', parentNode: null, _text: '', value: '', selectionStart: 0, selectionEnd: 0,
    scrollTop: 0, clientHeight: 400, clientWidth: 600, offsetTop: 0, naturalWidth: 0, naturalHeight: 0};
  e.appendChild = c => { c.parentNode = e; e.children.push(c); return c; };
  e.removeChild = c => { const i = e.children.indexOf(c); if (i >= 0) e.children.splice(i, 1); c.parentNode = null; return c; };
  Object.defineProperty(e, 'firstChild', {get: () => e.children[0] || null});
  Object.defineProperty(e, 'textContent', {
    get: () => e._text + e.children.map(c => c.textContent).join(''),
    set: v => { e.children = []; e._text = String(v); }
  });
  e.setAttribute = (k, v) => { e.attrs[k] = String(v); };
  e.getAttribute = k => (k in e.attrs ? e.attrs[k] : null);
  e.addEventListener = (t, fn) => { (e.listeners[t] = e.listeners[t] || []).push(fn); };
  e.removeEventListener = () => {};
  e.dispatch = t => (e.listeners[t] || []).forEach(fn => fn({preventDefault() {}, stopPropagation() {}}));
  e.contains = n => { while (n) { if (n === e) return true; n = n.parentNode; } return false; };
  e.getElementsByTagName = t => {
    const out = [];
    const walk = n => n.children.forEach(c => { if (c.tagName === t.toUpperCase()) out.push(c); walk(c); });
    walk(e);
    return out;
  };
  e.getBoundingClientRect = () => ({left: 0, top: 0, width: 600, height: 800});
  e.focus = () => {};
  e.setSelectionRange = (a, b) => { e.selectionStart = a; e.selectionEnd = b; };
  e.setRangeText = (text, a, b) => {
    e.value = e.value.slice(0, a) + text + e.value.slice(b);
    e.selectionStart = e.selectionEnd = a + text.length;
  };
  return e;
}
globalThis.document = {body: makeEl('body'), createElement: makeEl, createTextNode: makeText, execCommand: () => false};
function key(code, mods) {
  const ev = Object.assign({code: code, key: code, altKey: false, ctrlKey: false, metaKey: false,
    preventDefault() {}, stopPropagation() {}}, mods);
  (listeners.keydown || []).forEach(fn => fn(ev));
}
"""

SMOKE = r"""
const fs = require('fs');
const input = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const WINDOW_JS = input.window_js;
function exec(call) {
  const code = 'const CALL = ' + JSON.stringify(call) + ';\n' + WINDOW_JS;
  return new Function('return (async () => {' + code + '})()')();
}
(async () => {
  const out = {};
  out.collect = await exec({op: 'collect_outbox'});
  let pending = exec({op: 'show', page: input.item, images: {page: null, crops: {}}, reuse_images: false,
    can_back: false, autosave_idle_s: 10, acks: [], outbox_ack: []});
  const R = window.__ocrReview;
  const ta = R.ui.ta;
  out.mirror_marks = R.ui.mirror.getElementsByTagName('mark').length;
  ta.value = 'X' + ta.value;
  ta.selectionStart = ta.selectionEnd = 1;
  ta.dispatch('input');
  out.hint_after_insert = R.ps.hints[0].start;
  key('ArrowDown', {altKey: true});
  const draft1 = await pending;
  out.draft1 = {type: draft1.type, expected_seq: draft1.draft.expected_seq, starts_with_x: draft1.draft.text[0] === 'X'};
  pending = exec({op: 'continue', acks: [{event_id: draft1.id, page_id: draft1.page_id, type: 'draft', status: 'ok', seq: 1, rev: draft1.rev}], outbox_ack: []});
  const local = JSON.parse(localStorage.getItem('ocrReview.draft.' + input.item.page_id));
  out.local_after_ack = {base_seq: local.base_seq, synced: local.rev === local.sent_rev};
  key('Digit1', {altKey: true});
  const draft2 = await pending;
  out.draft2 = {expected_seq: draft2.draft.expected_seq, hint_state: R.ps.hints[0].state, has_variant: draft2.draft.text.indexOf(input.variant_a) >= 0};
  pending = exec({op: 'continue', acks: [{event_id: draft2.id, page_id: draft2.page_id, type: 'draft', status: 'ok', seq: 2, rev: draft2.rev}], outbox_ack: []});
  key('Enter', {ctrlKey: true});
  const done = await pending;
  out.done = {type: done.type, draft_seq: done.review.draft_seq, blocked: R.blocked, outbox: JSON.parse(localStorage.getItem('ocrReview.outbox')).length};
  out.close = await exec({op: 'close', acks: [{event_id: done.id, page_id: done.page_id, type: 'done', status: 'ok', request_id: done.review.request_id}], outbox_ack: []});
  out.after_close = {outbox: JSON.parse(localStorage.getItem('ocrReview.outbox')).length,
    local: localStorage.getItem('ocrReview.draft.' + input.item.page_id), body_children: document.body.children.length};
  console.log(JSON.stringify(out));
  process.exit(0);
})().catch(err => { console.error(err && err.stack || err); process.exit(1); });
"""


def test_window_runs_through_one_page(tmp_path: Path) -> None:
    text = "Строка один 47 опытов\nВторая строка"
    start = text.index("47 опытов")
    item = {
        "page_id": 5,
        "document_id": "doc1",
        "title": "Отчёт",
        "page_number": 3,
        "status": "flagged",
        "scope": "hints",
        "item_version": "v1",
        "base_attempt_id": 50,
        "base_revision_id": None,
        "text": text,
        "no_candidate": False,
        "printed_label": None,
        "hints": [
            {
                "id": 1,
                "start": start,
                "end": start + len("47 опытов"),
                "reasons": ["diff"],
                "variants": {"a": "41 опыт", "b": "47 опытов"},
                "crop": None,
            }
        ],
        "draft": None,
        "header": {"counter": "1 из 1", "title": "Отчёт", "page_label": "Стр. 3", "reasons": ""},
    }
    data = tmp_path / "input.json"
    data.write_text(
        json.dumps({"window_js": op.REVIEW_WINDOW_JS, "item": item, "variant_a": "41 опыт"}),
        encoding="utf-8",
    )
    result = run_node(FAKE_DOM + SMOKE, tmp_path, str(data))
    assert result["collect"] == {"type": "outbox", "entries": []}
    assert result["mirror_marks"] == 1
    assert result["hint_after_insert"] == start + 1
    assert result["draft1"] == {"type": "draft", "expected_seq": 0, "starts_with_x": True}
    assert result["local_after_ack"] == {"base_seq": 1, "synced": True}
    assert result["draft2"] == {"expected_seq": 1, "hint_state": "chosen_a", "has_variant": True}
    assert result["done"] == {"type": "done", "draft_seq": 2, "blocked": True, "outbox": 1}
    assert result["close"] == {"type": "closed"}
    assert result["after_close"] == {"outbox": 0, "local": None, "body_children": 0}


def test_window_code_is_valid_javascript(tmp_path: Path) -> None:
    code = op.window_code({"op": "close", "text": "строка с разделителем"})
    data = tmp_path / "code.js"
    data.write_text(code, encoding="utf-8")
    script = (
        "const fs = require('fs');\n"
        "const code = fs.readFileSync(process.argv[2], 'utf8');\n"
        "new Function('return (async () => {' + code + '})()');\n"
        "console.log(JSON.stringify({ok: true}));\n"
    )
    assert run_node(script, tmp_path, str(data)) == {"ok": True}
