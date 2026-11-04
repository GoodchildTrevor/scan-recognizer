"""Python references of the window's draft restore logic (section 8.4)."""

from __future__ import annotations

from typing import Any

import pytest

from tests.owui.helpers import op

pytestmark = pytest.mark.p3

ITEM: dict[str, Any] = {
    "page_id": 7,
    "item_version": "v2",
    "base_attempt_id": 70,
    "base_revision_id": None,
    "text": "Кандидат страницы",
    "printed_label": None,
    "hints": [{"id": 1, "start": 0, "end": 8, "reasons": ["diff"], "variants": None, "crop": None}],
}


def server_draft(text: str, seq: int, **extra: Any) -> dict[str, Any]:
    return {
        "seq": seq,
        "text": text,
        "text_sha256": op.text_sha256(text),
        "printed_label": "9 об.",
        "choices": {},
        "hints": [{"id": 1, "start": 0, "end": 3, "reasons": ["diff"], "state": "manual"}],
        "cursor": 4,
        "item_version": "v2",
        "base_attempt_id": 70,
        "base_revision_id": None,
        "stale": False,
        "stale_reason": None,
        **extra,
    }


def local_copy(
    text: str, *, base_seq: int, rev: int, sent_rev: int, **extra: Any
) -> dict[str, Any]:
    return {
        "page_id": 7,
        "text": text,
        "printed_label": "9 об.",
        "hints": [],
        "choices": {"1": "a"},
        "cursor": 2,
        "item_version": "v2",
        "base_attempt_id": 70,
        "base_revision_id": None,
        "base_seq": base_seq,
        "rev": rev,
        "sent_rev": sent_rev,
        "in_flight": None,
        **extra,
    }


def test_no_copies_open_the_page_text() -> None:
    state = op.restore_state(ITEM, None, None)
    assert state["mode"] == "edit"
    assert state["source"] == "item"
    assert state["text"] == ITEM["text"]
    assert state["hints"][0]["state"] == "active"
    assert state["base_seq"] == 0


def test_synced_local_copy_yields_to_server_draft() -> None:
    local = local_copy("старое", base_seq=3, rev=5, sent_rev=5)
    state = op.restore_state(ITEM, local, server_draft("с сервера", 4))
    assert state["source"] == "server"
    assert state["text"] == "с сервера"
    assert state["base_seq"] == 4
    assert not state["dirty"]


def test_unsent_local_edits_on_top_of_current_save_win() -> None:
    local = local_copy("дописал без связи", base_seq=4, rev=9, sent_rev=7)
    state = op.restore_state(ITEM, local, server_draft("с сервера", 4))
    assert state["source"] == "local"
    assert state["text"] == "дописал без связи"
    assert state["base_seq"] == 4
    assert state["dirty"]


def test_lost_confirmation_is_resolved_without_questions() -> None:
    # The server wrote A (seq 5 -> 6), the answer was lost, then the human typed B.
    in_flight = {"expected_seq": 5, "rev": 7, "sha256": op.text_sha256("A")}
    local = local_copy("B", base_seq=5, rev=9, sent_rev=6, in_flight=in_flight)
    state = op.restore_state(ITEM, local, server_draft("A", 6))
    assert state["mode"] == "edit"
    assert state["source"] == "local"
    assert state["text"] == "B"
    assert state["base_seq"] == 6
    assert state["sent_rev"] == 7
    assert state["in_flight"] is None
    assert state["dirty"]


def test_other_tab_save_is_a_real_conflict_with_both_versions() -> None:
    in_flight = {"expected_seq": 5, "rev": 7, "sha256": op.text_sha256("A")}
    local = local_copy("B", base_seq=5, rev=9, sent_rev=6, in_flight=in_flight)
    state = op.restore_state(ITEM, local, server_draft("C из другой вкладки", 6))
    assert state["mode"] == "conflict"
    assert state["text"] == "B"
    assert state["conflict"] == {
        "server_seq": 6,
        "server_text": "C из другой вкладки",
        "server_printed_label": "9 об.",
    }


def test_conflict_without_server_draft_offers_page_text() -> None:
    local = local_copy("мой текст", base_seq=3, rev=4, sent_rev=2)
    state = op.restore_state(ITEM, local, None)
    assert state["mode"] == "conflict"
    assert state["conflict"]["server_text"] == ITEM["text"]
    assert state["conflict"]["server_seq"] == 0


def test_changed_item_version_drops_hints_but_keeps_text_and_label() -> None:
    draft = server_draft("правленый текст", 2, item_version="v1")
    state = op.restore_state(ITEM, None, draft)
    assert state["mode"] == "edit"
    assert state["hints"] == []
    assert state["hints_outdated"]
    assert state["text"] == "правленый текст"
    assert state["printed_label"] == "9 об."
    assert state["hints_version"] == "v1"


def test_same_item_version_restores_shifted_hints() -> None:
    state = op.restore_state(ITEM, None, server_draft("правка", 2))
    assert state["hints"] == [
        {"id": 1, "start": 0, "end": 3, "reasons": ["diff"], "state": "manual"}
    ]
    assert not state["hints_outdated"]
    assert state["cursor"] == 4


def test_stale_server_draft_opens_with_the_reason() -> None:
    draft = server_draft(
        "мой текст", 3, base_attempt_id=69, stale=True, stale_reason="recognized_again"
    )
    state = op.restore_state(ITEM, None, draft)
    assert state["mode"] == "stale"
    assert state["stale_reason"] == "recognized_again"


def test_local_copy_on_an_old_base_is_stale() -> None:
    local = local_copy("мой текст", base_seq=0, rev=2, sent_rev=0, base_attempt_id=69)
    state = op.restore_state(ITEM, local, None)
    assert state["mode"] == "stale"
    assert state["stale_reason"] == "changed"


def test_confirmation_of_an_old_save_keeps_newer_edits_unsent() -> None:
    in_flight = {"expected_seq": 4, "rev": 5, "sha256": "x"}
    local = local_copy("текст", base_seq=4, rev=5, sent_rev=2, in_flight=in_flight)
    updated = op.apply_draft_ack(local, rev=3, seq=5)
    assert updated["base_seq"] == 5
    assert updated["sent_rev"] == 3
    assert updated["in_flight"] == in_flight
    assert updated["rev"] != updated["sent_rev"]
    synced = op.apply_draft_ack(updated, rev=5, seq=6)
    assert synced["in_flight"] is None
    assert synced["rev"] == synced["sent_rev"]


def test_choose_draft_source_cases() -> None:
    assert op.choose_draft_source(None, None) == {"source": "server", "base_seq": 0, "sent_rev": 0}
    local = local_copy("x", base_seq=2, rev=3, sent_rev=1)
    assert op.choose_draft_source(local, server_draft("y", 2))["source"] == "local"
    assert op.choose_draft_source(local, server_draft("y", 4))["source"] == "conflict"
