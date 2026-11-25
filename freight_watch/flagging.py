-"""Anomaly rule: rising vs own history AND out of the ordinary vs history or peers."""

from __future__ import annotations

import math

import pandas as pd

from .config import OWN_HISTORY_THRESHOLD_PCT, PEER_THRESHOLD_PCT

TRIGGER_OWN = "own_history"
TRIGGER_PEER = "peer"


def triggers_for(
    vs_own_pct: float,
    vs_peer_pct: float,
    own_threshold: float = OWN_HISTORY_THRESHOLD_PCT,
    peer_threshold: float = PEER_THRESHOLD_PCT,
) -> list[str]:
    """Which comparisons make this week anomalous. Empty list -> not flagged.

    A week with no prior history cannot be "rising", so it is never flagged.
    """
    if vs_own_pct is None or math.isnan(vs_own_pct) or vs_own_pct <= 0:
        return []
    triggers = []
    if vs_own_pct >= own_threshold:
        triggers.append(TRIGGER_OWN)
    if vs_peer_pct is not None and not math.isnan(vs_peer_pct) and vs_peer_pct >= peer_threshold:
        triggers.append(TRIGGER_PEER)
    return triggers


def anomalous_weeks(
    weekly: pd.DataFrame,
    own_threshold: float = OWN_HISTORY_THRESHOLD_PCT,
    peer_threshold: float = PEER_THRESHOLD_PCT,
) -> pd.DataFrame:
    weekly = weekly.copy()
    weekly["triggers"] = [triggers_for(o, p, own_threshold, peer_threshold)
                          for o, p in zip(weekly["vs_own_pct"], weekly["vs_peer_pct"])]
    flagged = weekly[weekly["triggers"].map(bool)]
    return flagged.sort_values(["week_of", "route"]).reset_index(drop=True)
