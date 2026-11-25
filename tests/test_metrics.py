import pandas as pd
import pytest

from freight_watch.data import monday_of
from freight_watch.flagging import triggers_for
from freight_watch.metrics import weekly_costs


def shipments(rows):
    df = pd.DataFrame(rows, columns=["route", "route_type", "date", "tonnes", "km", "cost"])
    df["shipment_id"] = [f"S{i}" for i in range(len(df))]
    df["week_of"] = monday_of(pd.to_datetime(df["date"]))
    df["tonne_km"] = df["tonnes"] * df["km"]
    df["freight_cost_inr"] = df["cost"]
    return df


def test_weeks_run_monday_to_sunday():
    dates = pd.to_datetime(pd.Series(["2024-01-01", "2024-01-07", "2024-01-08", "2024-01-03 17:30"]), format="mixed")
    assert [d.date().isoformat() for d in monday_of(dates)] == ["2024-01-01", "2024-01-01", "2024-01-08", "2024-01-01"]


def test_weekly_cost_is_total_cost_over_total_tonne_km():
    df = shipments([
        ("A-B", "Short", "2024-01-01", 10, 100, 1000),  # 1.0 per tkm
        ("A-B", "Short", "2024-01-02", 30, 100, 6000),  # 2.0 per tkm
    ])
    weekly = weekly_costs(df)
    assert weekly["cost_per_tonne_km"].iloc[0] == pytest.approx(7000 / 4000)  # not mean(1.0, 2.0)


def _series(route, route_type, values, start="2024-01-01"):
    mondays = pd.date_range(start, periods=len(values), freq="7D")
    return [(route, route_type, d.date().isoformat(), 1, 1, v) for d, v in zip(mondays, values)]


def test_own_history_uses_only_prior_weeks_and_at_most_eight():
    values = [1, 1, 1, 1, 1, 1, 1, 1, 9, 2, 100]
    weekly = weekly_costs(shipments(_series("A-B", "Short", values))).set_index("week_of")
    weeks = list(weekly.index)
    assert pd.isna(weekly.loc[weeks[0], "own_history_avg"])          # no prior weeks
    assert weekly.loc[weeks[3], "own_history_avg"] == 1               # fewer than 8 -> all available
    assert weekly.loc[weeks[3], "prior_weeks_used"] == 3
    assert weekly.loc[weeks[8], "own_history_avg"] == 1               # the 9 itself is excluded
    assert weekly.loc[weeks[9], "own_history_avg"] == pytest.approx((7 * 1 + 9) / 8)  # week 0 dropped out
    # No look-ahead: the 100 in the final week appears in no baseline, including its own.
    assert weekly.loc[weeks[10], "own_history_avg"] == pytest.approx((6 * 1 + 9 + 2) / 8)


def test_calendar_window_skips_missing_weeks():
    rows = _series("A-B", "Short", [5]) + _series("A-B", "Short", [1], start="2024-03-11")  # 10 weeks later
    weekly = weekly_costs(shipments(rows))
    assert pd.isna(weekly["own_history_avg"].iloc[1])  # 10-week-old value is outside the 8-week window


def test_peer_average_excludes_the_route_itself_and_other_types():
    rows = (_series("A-B", "Short", [2.0]) + _series("C-D", "Short", [4.0]) + _series("E-F", "Short", [6.0])
            + _series("G-H", "Long", [100.0]))
    weekly = weekly_costs(shipments(rows)).set_index("route")
    assert weekly.loc["A-B", "peer_avg"] == 5.0
    assert weekly.loc["E-F", "peer_avg"] == 3.0
    assert pd.isna(weekly.loc["G-H", "peer_avg"])  # no other Long route that week


@pytest.mark.parametrize("own, peer, expected", [
    (16, 0, ["own_history"]),
    (5, 25, ["peer"]),
    (20, 30, ["own_history", "peer"]),
    (-1, 40, []),            # not rising -> never flagged
    (float("nan"), 50, []),  # no history -> cannot be rising
    (14.9, 19.9, []),
])
def test_flag_rule(own, peer, expected):
    assert triggers_for(own, peer) == expected
