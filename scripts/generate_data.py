"""Generate the synthetic ReplenishIQ retail dataset (seeded, reproducible).

Creates a SQLite database with three tables:
  inventory      (sku, store, on_hand, on_order, shelf_capacity, unit_cost)
  sales_history  (date, sku, store, units_sold)          ~180 days per SKU/store
  suppliers      (supplier, sku, lead_time_days, moq, case_pack)
plus a small `products` lookup (sku, name, category) and an empty
`purchase_orders` table that the MCP server writes to after human approval.

Run:  python -m scripts.generate_data
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DB_PATH  # noqa: E402

SEED = 42
HISTORY_DAYS = 180
LAST_SALES_DATE = date(2026, 10, 21)

# sku, name, category, unit_cost, base daily units (S001), supplier, lead_time, moq, case_pack
PRODUCTS = [
    ("GRO-001", "Whole Milk 1 gal", "Grocery", 2.80, 38, "DairyFresh Co", 2, 24, 4),
    ("GRO-002", "Large Eggs 12 ct", "Grocery", 2.10, 30, "DairyFresh Co", 2, 30, 15),
    ("GRO-003", "Sandwich Bread 20 oz", "Grocery", 1.60, 26, "Sunrise Bakery", 1, 24, 12),
    ("GRO-004", "Bananas 3 lb bag", "Grocery", 1.20, 45, "FreshFields Produce", 3, 40, 20),
    ("HH-001", "Paper Towels 6-roll", "Household", 6.50, 12, "CleanHome Supply", 5, 24, 6),
    ("HH-002", "Laundry Detergent 92 oz", "Household", 8.90, 9, "CleanHome Supply", 5, 16, 4),
    ("HH-003", "Dish Soap 19 oz", "Household", 3.10, 11, "CleanHome Supply", 5, 24, 12),
    ("PC-001", "Toothpaste 2-pack", "Personal Care", 3.40, 8, "BrightSmile Brands", 7, 24, 12),
    ("PC-002", "Shampoo 25 oz", "Personal Care", 4.20, 6, "BrightSmile Brands", 7, 12, 6),
    ("ELE-001", "AA Batteries 24-pack", "Electronics", 9.80, 5, "VoltMax Distribution", 10, 10, 10),
]
STORES = {"S001": 1.0, "S002": 0.7}  # store size multiplier on demand
# Weekly seasonality (Mon..Sun): weekend peak
DOW_FACTOR = np.array([0.90, 0.85, 0.90, 0.95, 1.10, 1.25, 1.05])
# Current stock as "days of cover" for S001 so some SKUs are clearly low
DAYS_OF_COVER = {
    "GRO-001": 2.0, "GRO-002": 5.5, "GRO-003": 1.5, "GRO-004": 2.5, "HH-001": 9.0,
    "HH-002": 3.0, "HH-003": 16.0, "PC-001": 6.0, "PC-002": 20.0, "ELE-001": 4.0,
}


def build_frames(seed: int = SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = [LAST_SALES_DATE - timedelta(days=HISTORY_DAYS - 1 - i) for i in range(HISTORY_DAYS)]
    t = np.arange(HISTORY_DAYS)

    products = pd.DataFrame(
        [(p[0], p[1], p[2]) for p in PRODUCTS], columns=["sku", "name", "category"]
    )
    suppliers = pd.DataFrame(
        [(p[5], p[0], p[6], p[7], p[8]) for p in PRODUCTS],
        columns=["supplier", "sku", "lead_time_days", "moq", "case_pack"],
    )

    sales_rows, inv_rows = [], []
    for sku, _name, _cat, cost, base, *_ in PRODUCTS:
        for store, mult in STORES.items():
            trend = 1 + 0.10 * (t / HISTORY_DAYS)  # mild growth over the period
            annual = 1 + 0.08 * np.sin(2 * np.pi * (t + 30) / 365)  # slow seasonal swing
            dow = np.array([DOW_FACTOR[d.weekday()] for d in dates])
            lam = base * mult * trend * annual * dow
            units = rng.poisson(lam)
            sales_rows += [(d.isoformat(), sku, store, int(u)) for d, u in zip(dates, units)]

            recent_avg = units[-28:].mean()
            cover = DAYS_OF_COVER[sku] * (1.3 if store == "S002" else 1.0)
            on_hand = int(round(recent_avg * cover))
            on_order = int(round(recent_avg * 1.0)) if sku in ("GRO-002", "PC-001") else 0
            shelf_capacity = int(np.ceil(recent_avg * 21 / 10) * 10)  # ~3 weeks of demand
            inv_rows.append((sku, store, on_hand, on_order, shelf_capacity, cost))

    inventory = pd.DataFrame(
        inv_rows, columns=["sku", "store", "on_hand", "on_order", "shelf_capacity", "unit_cost"]
    )
    sales = pd.DataFrame(sales_rows, columns=["date", "sku", "store", "units_sold"])
    return {"products": products, "inventory": inventory, "sales_history": sales, "suppliers": suppliers}


SCHEMA = """
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS inventory;
DROP TABLE IF EXISTS sales_history;
DROP TABLE IF EXISTS suppliers;
DROP TABLE IF EXISTS purchase_orders;
CREATE TABLE products (sku TEXT PRIMARY KEY, name TEXT NOT NULL, category TEXT NOT NULL);
CREATE TABLE inventory (
    sku TEXT NOT NULL, store TEXT NOT NULL, on_hand INTEGER NOT NULL, on_order INTEGER NOT NULL,
    shelf_capacity INTEGER NOT NULL, unit_cost REAL NOT NULL, PRIMARY KEY (sku, store));
CREATE TABLE sales_history (
    date TEXT NOT NULL, sku TEXT NOT NULL, store TEXT NOT NULL, units_sold INTEGER NOT NULL,
    PRIMARY KEY (date, sku, store));
CREATE TABLE suppliers (
    supplier TEXT NOT NULL, sku TEXT PRIMARY KEY, lead_time_days INTEGER NOT NULL,
    moq INTEGER NOT NULL, case_pack INTEGER NOT NULL);
CREATE TABLE purchase_orders (
    po_id TEXT NOT NULL, created_at TEXT NOT NULL, store TEXT NOT NULL, sku TEXT NOT NULL,
    quantity INTEGER NOT NULL, unit_cost REAL NOT NULL, approved_by TEXT, note TEXT);
"""


def write_db(frames: dict[str, pd.DataFrame], db_path: Path = DB_PATH) -> Path:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)
        for table, df in frames.items():
            df.to_sql(table, conn, if_exists="append", index=False)
    return db_path


def main() -> None:
    frames = build_frames()
    path = write_db(frames)
    print(f"Wrote {path}")
    for name, df in frames.items():
        print(f"  {name:<14} {len(df):>5} rows")


if __name__ == "__main__":
    main()
