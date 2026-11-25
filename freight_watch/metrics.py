"""Weekly cost per tonne-km and the two graded baselines.

Definitions (from the brief, applied literally):
  * weekly cost per tonne-km = sum(freight cost) / sum(tonnes x km) over the route's
    shipments that week. This volume-weighted form reproduces all three sample rows
    exactly (e.g. Delhi-Jaipur 2024-11-11 -> 4.17, +35.5%, +21.0%).
  * vs own history = mean of the route's weekly values in the 8 calendar weeks
    strictly before this week (no look-ahead). Fewer than 8 -> all available.
  * vs similar routes = mean of the weekly values of *other* routes with the same
    route_type in the same week (the route itself is excluded).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BASELINE_WEEKS


def weekly_costs(shipments: pd.DataFrame, baseline_weeks: int = BASELINE_WEEKS) -> pd.DataFrame:
    grouped = shipments.groupby(["route", "route_type", "week_of"], as_index=False).agg(
        shipments=("shipment_id", "count"),
        freight_cost_inr=("freight_cost_inr", "sum"),
        tonne_km=("tonne_km", "sum"),
    )
    grouped["cost_per_tonne_km"] = grouped["freight_cost_inr"] / grouped["tonne_km"]
    weekly = grouped.sort_values(["route", "week_of"]).reset_index(drop=True)

    weekly = _add_own_history(weekly, baseline_weeks)
    weekly = _add_peer_average(weekly)

    weekly["vs_own_pct"] = (weekly["cost_per_tonne_km"] / weekly["own_history_avg"] - 1) * 100
    weekly["vs_peer_pct"] = (weekly["cost_per_tonne_km"] / weekly["peer_avg"] - 1) * 100
    return weekly


def _add_own_history(weekly: pd.DataFrame, baseline_weeks: int) -> pd.DataFrame:
    """Calendar-based trailing window: weeks in [week_of - N weeks, week_of)."""
    window = pd.Timedelta(weeks=baseline_weeks)
    averages, counts = [], []
    for _, route_weeks in weekly.groupby("route", sort=False):
        dates = route_weeks["week_of"].to_numpy()
        values = route_weeks["cost_per_tonne_km"].to_numpy()
        for current in dates:
            prior = (dates < current) & (dates >= current - window)
            counts.append(int(prior.sum()))
            averages.append(float(values[prior].mean()) if prior.any() else np.nan)
    # groupby(sort=False) preserves the frame's (route, week_of) ordering
    weekly["own_history_avg"] = averages
    weekly["prior_weeks_used"] = count
    return weekly


def _add_peer_average(weekly: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-out mean within (route_type, week_of)."""
    keys = ["route_type", "week_of"]
    total = weekly.groupby(keys)["cost_per_tonne_km"].transform("sum")
    count = weekly.groupby(keys)["cost_per_tonne_km"].transform("count")
    peers = count - 1
    weekly["peer_routes"] = peers
    weekly["peer_avg"] = np.where(peers > 0, (total - weekly["cost_per_tonne_km"]) / peers.where(peers > 0), np.nan)
    return weekly
