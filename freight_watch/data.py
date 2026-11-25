"""Loading and validating the shipment file. No LLM anywhere near this layer."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "shipment_id",
    "origin",
    "destination",
    "route_type",
    "quantity_tonnes",
    "distance_km",
    "freight_cost_inr",
    "shipment_date",
]
ROUTE_TYPES = {"Short", "Medium", "Long"}


class DataValidationError(ValueError):
    pass


def load_shipments(path: Path) -> pd.DataFrame:
    """Read shipments, drop rows that cannot produce a valid cost per tonne-km, add route + week_of.

    Invalid rows are dropped (and logged) rather than imputed: a made-up quantity
    would silently move the cost metric we are graded on.
    """
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"{path.name} is missing columns: {missing}")

    df["shipment_date"] = pd.to_datetime(df["shipment_date"], errors="coerce")
    for col in ("quantity_tonnes", "distance_km", "freight_cost_inr"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid = (
        df["shipment_date"].notna()
        & (df["quantity_tonnes"] > 0)
        & (df["distance_km"] > 0)
        & (df["freight_cost_inr"] > 0)
        & df["route_type"].isin(ROUTE_TYPES)
        & df["origin"].notna()
        & df["destination"].notna()
    )
    dropped = int((~valid).sum())
    if dropped:
        log.warning("Dropping %d invalid shipment rows", dropped)
    duplicates = int(df["shipment_id"].duplicated().sum())
    if duplicates:
        log.warning("Dropping %d duplicate shipment_id rows", duplicates)
        valid &= ~df["shipment_id"].duplicated()

    df = df[valid].copy()
    df["route"] = df["origin"].str.strip() + "-" + df["destination"].str.strip()
    df["week_of"] = monday_of(df["shipment_date"])
    df["tonne_km"] = df["quantity_tonnes"] * df["distance_km"]

    inconsistent = df.groupby("route")["route_type"].nunique()
    if (inconsistent > 1).any():
        raise DataValidationError(f"Routes with more than one route_type: {list(inconsistent[inconsistent > 1].index)}")
    return df


def monday_of(dates: pd.Series) -> pd.Series:
    """Weeks run Monday..Sunday; week_of is that Monday (time-of-day stripped)."""
    dates = dates.dt.normalize()
    return dates - pd.to_timedelta(dates.dt.weekday, unit="D")






































