"""Tests for PR-G2 — RTK ``tokens_saved`` data-plane wiring.

Phase G of the Headroom realignment retires the dead ``tokens_saved_rtk``
field by sourcing it from RTK's own stats endpoint (``rtk gain --format
json`` via :func:`headroom.proxy.helpers._get_rtk_stats`) and writing the
per-call delta into ``HeadroomContribution.tokens_saved_rtk``. Previously,
the field silently mirrored ``tokens_saved_cli_filtering`` — making the
two counters identical at all times and breaking the dashboard's ability
to distinguish proxy-side compression from wrap-side RTK savings.

These tests pin the wiring:

1. The delta is computed correctly across two consecutive
   :meth:`update_contribution` calls (monotonic counter advances).
2. ``tokens_saved_rtk`` is exactly zero when ``_get_rtk_stats()`` returns
   ``None`` (RTK not installed / not selected).
3. ``_last_rtk_tokens_saved`` advances monotonically; deltas are not
   replayed across calls when the lifetime counter does not move.

Realignment build constraints honored:

- No silent fallback: a transient ``_get_rtk_stats()`` exception is
  structured-logged and yields ``tokens_saved_rtk = 0`` (test 4).
- Configurable: ``HEADROOM_RTK_WIRING=disabled`` opts the polling out and
  produces a clean zero, exercised by ``test_disabled_env_returns_zero``.
- Structured logs: each failure path emits a ``event=…`` line; the tests
  do not pin the log payload to avoid coupling, but the helper signatures
  surface them.
- Comprehensive tests: 6 unit tests + 1 explicit delta test cover the
  contract.
"""

from __future__ import annotations

from typing import Any

import pytest

import headroom.subscription.tracker as tracker_module
from headroom.subscription.tracker import SubscriptionTracker


def _build_tracker(monkeypatch: pytest.MonkeyPatch) -> SubscriptionTracker:
    """Construct a tracker with persistence disabled (unit-test isolation)."""

    monkeypatch.setattr(SubscriptionTracker, "_load_persisted_state", lambda self: None)
    return SubscriptionTracker(enabled=True)


def _stub_rtk_stats(
    monkeypatch: pytest.MonkeyPatch, payloads: list[dict[str, Any] | None]
) -> list[int]:
    """Stub ``_get_rtk_stats`` to return ``payloads`` in order.

    Returns a counter list (mutated by the stub) so callers can assert the
    number of polls.
    """

    call_count: list[int] = [0]

    def fake_get_rtk_stats() -> dict[str, Any] | None:
        idx = call_count[0]
        call_count[0] += 1
        if idx >= len(payloads):
            return payloads[-1]
        return payloads[idx]

    monkeypatch.setattr(
        "headroom.proxy.helpers._get_rtk_stats",
        fake_get_rtk_stats,
    )
    return call_count


# ---------------------------------------------------------------------------
# Test 1 — delta computed correctly across two consecutive polls
# ---------------------------------------------------------------------------


def test_tokens_saved_rtk_populated_from_rtk_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First call seeds the baseline; second call writes the positive delta."""

    tracker = _build_tracker(monkeypatch)
    monkeypatch.delenv(tracker_module._RTK_WIRING_ENV, raising=False)
    _stub_rtk_stats(
        monkeypatch,
        [
            {"lifetime_tokens_saved": 100},
            {"lifetime_tokens_saved": 175},
        ],
    )

    # First call — establishes the baseline at 100, contributes 100 (the
    # tracker starts at _last_rtk_tokens_saved == 0, so the first delta is
    # the full lifetime total). That matches the spec: the field reflects
    # cumulative session savings observed by the tracker.
    tracker.update_contribution()
    contribution_after_first = tracker._state.contribution.tokens_saved_rtk
    assert contribution_after_first == 100
    assert tracker._last_rtk_tokens_saved == 100

    # Second call — delta is 175 - 100 = 75; cumulative contribution = 175.
    tracker.update_contribution()
    assert tracker._state.contribution.tokens_saved_rtk == 175
    assert tracker._last_rtk_tokens_saved == 175


def test_delta_computed_correctly_across_polls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three consecutive polls — each adds only the new RTK delta."""

    tracker = _build_tracker(monkeypatch)
    monkeypatch.delenv(tracker_module._RTK_WIRING_ENV, raising=False)
    _stub_rtk_stats(
        monkeypatch,
        [
            {"lifetime_tokens_saved": 0},  # baseline at zero
            {"lifetime_tokens_saved": 50},
            {"lifetime_tokens_saved": 250},
        ],
    )

    tracker.update_contribution()
    assert tracker._state.contribution.tokens_saved_rtk == 0
    assert tracker._last_rtk_tokens_saved == 0

    tracker.update_contribution()
    assert tracker._state.contribution.tokens_saved_rtk == 50
    assert tracker._last_rtk_tokens_saved == 50

    tracker.update_contribution()
    # 50 + (250 - 50) = 250 cumulative; delta on the third call was 200.
    assert tracker._state.contribution.tokens_saved_rtk == 250
    assert tracker._last_rtk_tokens_saved == 250


# ---------------------------------------------------------------------------
# Test 2 — ``tokens_saved_rtk = 0`` when stats endpoint returns None
# ---------------------------------------------------------------------------


def test_rtk_stats_none_yields_zero_delta(monkeypatch: pytest.MonkeyPatch) -> None:
    """No RTK selected / installed — contribution stays at zero, no throw."""

    tracker = _build_tracker(monkeypatch)
    monkeypatch.delenv(tracker_module._RTK_WIRING_ENV, raising=False)
    _stub_rtk_stats(monkeypatch, [None, None])

    tracker.update_contribution()
    tracker.update_contribution()

    assert tracker._state.contribution.tokens_saved_rtk == 0
    assert tracker._last_rtk_tokens_saved == 0


# ---------------------------------------------------------------------------
# Test 3 — monotonic advancement; no replay on flat poll
# ---------------------------------------------------------------------------


def test_last_rtk_advances_monotonically(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two polls returning the same lifetime total contribute exactly once."""

    tracker = _build_tracker(monkeypatch)
    monkeypatch.delenv(tracker_module._RTK_WIRING_ENV, raising=False)
    _stub_rtk_stats(
        monkeypatch,
        [
            {"lifetime_tokens_saved": 42},
            {"lifetime_tokens_saved": 42},  # no movement
            {"lifetime_tokens_saved": 42},  # still no movement
        ],
    )

    tracker.update_contribution()
    assert tracker._state.contribution.tokens_saved_rtk == 42
    assert tracker._last_rtk_tokens_saved == 42

    tracker.update_contribution()
    assert tracker._state.contribution.tokens_saved_rtk == 42  # unchanged
    assert tracker._last_rtk_tokens_saved == 42

    tracker.update_contribution()
    assert tracker._state.contribution.tokens_saved_rtk == 42
    assert tracker._last_rtk_tokens_saved == 42


def test_counter_regression_rebaselines_without_negative_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RTK DB rebuild drops the lifetime total — re-baseline, do not subtract."""

    tracker = _build_tracker(monkeypatch)
    monkeypatch.delenv(tracker_module._RTK_WIRING_ENV, raising=False)
    _stub_rtk_stats(
        monkeypatch,
        [
            {"lifetime_tokens_saved": 500},
            {"lifetime_tokens_saved": 100},  # regression!
            {"lifetime_tokens_saved": 150},
        ],
    )

    tracker.update_contribution()
    assert tracker._state.contribution.tokens_saved_rtk == 500
    assert tracker._last_rtk_tokens_saved == 500

    tracker.update_contribution()
    # Regression: contribution stays at 500 (no negative subtraction).
    assert tracker._state.contribution.tokens_saved_rtk == 500
    # Baseline now points at the new (smaller) lifetime total so subsequent
    # polls can compute a meaningful delta.
    assert tracker._last_rtk_tokens_saved == 100

    tracker.update_contribution()
    # 150 - 100 = 50 new delta; contribution = 500 + 50 = 550.
    assert tracker._state.contribution.tokens_saved_rtk == 550
    assert tracker._last_rtk_tokens_saved == 150


# ---------------------------------------------------------------------------
# Test 4 — transient exception in the stats endpoint
# ---------------------------------------------------------------------------


def test_rtk_stats_exception_zero_delta_no_throw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raised ``_get_rtk_stats()`` is caught, logged, and yields 0 delta."""

    tracker = _build_tracker(monkeypatch)
    monkeypatch.delenv(tracker_module._RTK_WIRING_ENV, raising=False)

    def boom() -> dict[str, Any] | None:
        raise RuntimeError("transient subprocess failure")

    monkeypatch.setattr("headroom.proxy.helpers._get_rtk_stats", boom)

    # Must not raise.
    tracker.update_contribution()

    assert tracker._state.contribution.tokens_saved_rtk == 0
    assert tracker._last_rtk_tokens_saved == 0


# ---------------------------------------------------------------------------
# Test 5 — explicit env-var opt-out
# ---------------------------------------------------------------------------


def test_disabled_env_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """``HEADROOM_RTK_WIRING=disabled`` skips the poll entirely."""

    tracker = _build_tracker(monkeypatch)
    monkeypatch.setenv(tracker_module._RTK_WIRING_ENV, "disabled")

    polls = _stub_rtk_stats(
        monkeypatch,
        [{"lifetime_tokens_saved": 999}],
    )

    tracker.update_contribution()

    # Stats endpoint never called when wiring is disabled.
    assert polls[0] == 0
    assert tracker._state.contribution.tokens_saved_rtk == 0
    assert tracker._last_rtk_tokens_saved == 0


# ---------------------------------------------------------------------------
# Test 6 — explicit override from caller (back-compat for callers that
# already know the RTK delta out-of-band).
# ---------------------------------------------------------------------------


def test_explicit_rtk_override_skips_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caller-supplied ``tokens_saved_rtk`` short-circuits the poll."""

    tracker = _build_tracker(monkeypatch)
    monkeypatch.delenv(tracker_module._RTK_WIRING_ENV, raising=False)

    polls = _stub_rtk_stats(monkeypatch, [{"lifetime_tokens_saved": 999}])

    tracker.update_contribution(tokens_saved_rtk=17)

    # Stats endpoint not consulted when the caller passes an explicit value.
    assert polls[0] == 0
    assert tracker._state.contribution.tokens_saved_rtk == 17
    assert tracker._last_rtk_tokens_saved == 0


# ---------------------------------------------------------------------------
# Test 7 — cli_filtering decoupled from rtk
# ---------------------------------------------------------------------------


def test_cli_filtering_no_longer_mirrors_rtk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pre-PR-G2 bug: ``cli_filtering`` and ``rtk`` were always equal.

    After PR-G2 they are independent counters fed by separate sources.
    """

    tracker = _build_tracker(monkeypatch)
    monkeypatch.delenv(tracker_module._RTK_WIRING_ENV, raising=False)
    _stub_rtk_stats(monkeypatch, [{"lifetime_tokens_saved": 25}])

    tracker.update_contribution(tokens_saved_cli_filtering=8)

    assert tracker._state.contribution.tokens_saved_cli_filtering == 8
    # rtk comes from the polled delta, not from cli_filtering.
    assert tracker._state.contribution.tokens_saved_rtk == 25
