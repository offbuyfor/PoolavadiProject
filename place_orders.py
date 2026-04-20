"""
place_orders.py — Semi-automatic Alpaca married-options order placer.

Two-pass workflow:
  Pass 1 (options):  read BQ → confirm per trade → market buy option
  Pass 2 (stock):    check option fills → confirm per trade → market stock order
                     → poll stock fill → place closing limit on stock

Usage:
  python place_orders.py --pass1                  # place option orders
  python place_orders.py --pass2                  # place stock + closing limits
  python place_orders.py --pass1 --dry-run        # simulate without submitting
  python place_orders.py --pass1 --date 2025-04-09

State between passes is saved to orders_state.json in this directory.

Env vars required:
  ALPACA_API_KEY
  ALPACA_SECRET_KEY
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import date

import pandas as pd

# ── BigQuery ──────────────────────────────────────────────────────────────────
try:
    from google.cloud import bigquery as _bq
except ImportError:
    _bq = None

# ── Alpaca ────────────────────────────────────────────────────────────────────
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        GetOptionContractsRequest,
        MarketOrderRequest,
        LimitOrderRequest,
    )
    from alpaca.trading.enums import (
        AssetStatus,
        ContractType,
        OrderSide,
        TimeInForce,
    )
    ALPACA_OK = True
except ImportError:
    ALPACA_OK = False

# ── Tabulate (optional) ───────────────────────────────────────────────────────
try:
    from tabulate import tabulate as _tab
    def _print_table(rows, headers):
        print(_tab(rows, headers=headers, tablefmt="rounded_outline"))
except ImportError:
    def _print_table(rows, headers):
        col_w = [max(len(str(h)), *(len(str(r[i])) for r in rows), 0)
                 for i, h in enumerate(headers)]
        fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
        print(fmt.format(*headers))
        print("  ".join("-" * w for w in col_w))
        for row in rows:
            print(fmt.format(*[str(v) for v in row]))


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

BQ_TABLE   = "modelraja.UPCOMING_EARNINGS_DATASET.FOR_EXTERNALfinal_portfolio_optimization_paper"
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orders_state.json")
STOCK_QTY  = 100   # flat 100 shares per married position
POLL_SEC   = 5     # seconds between fill polls
POLL_MAX   = 12    # max polls (12 × 5s = 60s timeout)
DIV        = "=" * 72


# ─────────────────────────────────────────────────────────────────────────────
# State file
# ─────────────────────────────────────────────────────────────────────────────

def _load_state() -> list:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return []


def _save_state(state: list) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)
    print(f"  State saved → {STATE_FILE}")


def _find(state: list, ticker: str, option_type: str) -> dict | None:
    for e in state:
        if e["ticker"] == ticker and e["option_type"] == option_type:
            return e
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Alpaca client
# ─────────────────────────────────────────────────────────────────────────────

def _client() -> "TradingClient":
    if not ALPACA_OK:
        sys.exit("ERROR: alpaca-py not installed.  pip install alpaca-py")
    key    = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_SECRET_KEY", "")
    if not key or not secret:
        sys.exit("ERROR: Set ALPACA_API_KEY and ALPACA_SECRET_KEY env vars.")
    return TradingClient(key, secret, paper=True)


# ─────────────────────────────────────────────────────────────────────────────
# BigQuery
# ─────────────────────────────────────────────────────────────────────────────

def _load_bq(trade_date: str) -> pd.DataFrame:
    if _bq is None:
        sys.exit("ERROR: google-cloud-bigquery not installed.")
    client = _bq.Client()
    query = f"""
        SELECT *
        FROM `{BQ_TABLE}`
        WHERE snapshot_date = DATE('{trade_date}')
          AND evaluation_status = 'PENDING_NEXT_DAY_DATA'
        ORDER BY prediction_prob DESC
    """
    print(f"Reading BQ table for {trade_date} ...")
    df = client.query(query).to_dataframe()
    print(f"  {len(df)} trade(s) found\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Alpaca helpers
# ─────────────────────────────────────────────────────────────────────────────

def _option_symbol(client, ticker: str, option_type: str,
                   strike: float, expiry) -> str:
    ctype      = ContractType.CALL if option_type == "call" else ContractType.PUT
    expiry_dt  = pd.to_datetime(expiry).date() if expiry else None
    req = GetOptionContractsRequest(
        underlying_symbols=[ticker],
        status=AssetStatus.ACTIVE,
        type=ctype,
        strike_price_gte=str(round(strike - 0.01, 2)),
        strike_price_lte=str(round(strike + 0.01, 2)),
        expiration_date=expiry_dt,
    )
    resp      = client.get_option_contracts(req)
    contracts = getattr(resp, "option_contracts", None) or []
    if not contracts:
        raise ValueError(
            f"No Alpaca contract: {ticker} {option_type} strike={strike} expiry={expiry}"
        )
    return min(contracts, key=lambda c: abs(float(c.strike_price) - strike)).symbol


def _market(client, symbol: str, qty: int, side: "OrderSide", dry_run: bool) -> dict:
    if dry_run:
        return {"id": "DRY-RUN"}
    o = client.submit_order(MarketOrderRequest(
        symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY,
    ))
    return {"id": str(o.id)}


def _limit(client, symbol: str, qty: int, side: "OrderSide",
           price: float, dry_run: bool) -> dict:
    if dry_run:
        return {"id": "DRY-RUN"}
    o = client.submit_order(LimitOrderRequest(
        symbol=symbol, qty=qty, side=side,
        time_in_force=TimeInForce.DAY,
        limit_price=round(price, 2),
    ))
    return {"id": str(o.id)}


def _poll_fill(client, order_id: str, dry_run: bool) -> float | None:
    """Poll until filled; return avg fill price or None on timeout."""
    if dry_run:
        return 99.99
    for attempt in range(1, POLL_MAX + 1):
        o      = client.get_order_by_id(order_id)
        status = str(o.status).lower()
        print(f"    poll {attempt}/{POLL_MAX}  status={status}")
        if status == "filled":
            return float(o.filled_avg_price)
        if status in ("cancelled", "expired", "rejected"):
            print(f"    Order ended: {status}")
            return None
        time.sleep(POLL_SEC)
    print("    Timeout — could not confirm fill.")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────

def _ask(prompt: str) -> str:
    while True:
        v = input(f"\n  {prompt} [y / n / q=quit] : ").strip().lower()
        if v in ("y", "n", "q"):
            return v


# ─────────────────────────────────────────────────────────────────────────────
# PASS 1 — Buy options
# ─────────────────────────────────────────────────────────────────────────────

def pass1(trade_date: str, dry_run: bool) -> None:
    df = _load_bq(trade_date)
    if df.empty:
        print(f"No pending trades found for {trade_date}.")
        return

    # Build display table
    rows = []
    for _, r in df.iterrows():
        premium   = float(r.get("options_price") or 0)
        invest    = float(r.get("Investment") or 0)
        contracts = math.floor(invest / (premium * 100)) if premium > 0 else 0
        rows.append([
            r.get("lookupvalue", ""),
            str(r.get("option_type", "")).upper(),
            r.get("calls_strike", ""),
            r.get("Option_Expiry_Date", ""),
            f"${premium:.2f}",
            contracts,
            f"${invest:,.0f}",
            f"{float(r.get('prediction_prob') or 0) * 100:.1f}%",
            r.get("Earnings_Date", ""),
        ])

    print(DIV)
    print(f"  PASS 1 — BUY OPTIONS{' [DRY RUN]' if dry_run else ''}")
    print(DIV)
    _print_table(rows, ["Ticker", "Type", "Strike", "Expiry",
                        "Premium", "Contracts", "Invest", "Conf%", "Earnings"])
    total = df["Investment"].apply(lambda x: float(x or 0)).sum()
    print(f"\n  Total capital at risk: ${total:,.0f}")
    print(DIV)

    if input("\nProceed to per-trade confirmation? [y/n] : ").strip().lower() != "y":
        print("Aborted.")
        return

    client = None if dry_run else _client()
    state  = _load_state()

    for _, r in df.iterrows():
        ticker      = str(r.get("lookupvalue", ""))
        option_type = str(r.get("option_type", "")).lower()
        strike      = float(r.get("calls_strike") or 0)
        expiry      = r.get("Option_Expiry_Date", "")
        premium     = float(r.get("options_price") or 0)
        invest      = float(r.get("Investment") or 0)
        contracts   = math.floor(invest / (premium * 100)) if premium > 0 else 0

        print(f"\n  {ticker} {option_type.upper()}"
              f"  Strike: {strike}  Expiry: {expiry}"
              f"  Premium: ${premium:.2f}  Contracts: {contracts}"
              f"  Est. cost: ${contracts * premium * 100:,.0f}")

        if contracts < 1:
            print("  SKIPPED — investment too small for 1 contract.")
            continue

        dec = _ask(f"Market buy {contracts} {option_type} contract(s) on {ticker}?")
        if dec == "q":
            print("Quit. Remaining trades skipped.")
            break
        if dec == "n":
            print("  Skipped.")
            continue

        try:
            sym = (_option_symbol(client, ticker, option_type, strike, expiry)
                   if not dry_run else f"DRY_{ticker}_{option_type.upper()}")
            res = _market(client, sym, contracts, OrderSide.BUY, dry_run)
            tag = "DRY RUN" if dry_run else "SUBMITTED"
            print(f"  [{tag}]  symbol={sym}  order_id={res['id']}")

            entry = {
                "ticker":             ticker,
                "option_type":        option_type,
                "option_symbol":      sym,
                "option_order_id":    res["id"],
                "option_fill_price":  None,
                "contracts":          contracts,
                "calls_strike":       strike,
                "Option_Expiry_Date": str(expiry),
                "quoted_premium":     premium,
                "snapshot_date":      trade_date,
                "stock_order_id":     None,
                "stock_fill_price":   None,
                "closing_order_id":   None,
            }
            existing = _find(state, ticker, option_type)
            if existing:
                existing.update(entry)
            else:
                state.append(entry)
            _save_state(state)

        except Exception as exc:
            print(f"  ERROR: {exc}")

    print(f"\nRun  python place_orders.py --pass2  after your options fill.")


# ─────────────────────────────────────────────────────────────────────────────
# PASS 2 — Stock market order + closing limit
# ─────────────────────────────────────────────────────────────────────────────

def pass2(dry_run: bool) -> None:
    state = _load_state()
    if not state:
        print(f"No state at {STATE_FILE}. Run --pass1 first.")
        return

    pending = [e for e in state
               if e.get("stock_order_id") is None
               and e.get("option_order_id") is not None]

    if not pending:
        print("Nothing pending — all stock legs already placed.")
        return

    print(DIV)
    print(f"  PASS 2 — STOCK LEG + CLOSING LIMIT{' [DRY RUN]' if dry_run else ''}")
    print(DIV)
    rows = []
    for e in pending:
        action = "SELL short" if e["option_type"] == "call" else "BUY long"
        rows.append([e["ticker"], e["option_type"].upper(),
                     action, STOCK_QTY, f"${e['quoted_premium']:.2f}",
                     e["option_order_id"]])
    _print_table(rows, ["Ticker", "Opt", "Stock Action", "Shares",
                        "Opt Premium", "Option Order ID"])
    print(DIV)

    if input("\nProceed to per-trade confirmation? [y/n] : ").strip().lower() != "y":
        print("Aborted.")
        return

    client = None if dry_run else _client()

    for entry in pending:
        ticker      = entry["ticker"]
        option_type = entry["option_type"]
        opt_id      = entry["option_order_id"]

        print(f"\n  {ticker} {option_type.upper()}  —  option order: {opt_id}")

        # ── A: verify option fill ─────────────────────────────────────────────
        if not dry_run:
            o          = client.get_order_by_id(opt_id)
            opt_status = str(o.status).lower()
            if opt_status != "filled":
                print(f"  Option not filled yet (status={opt_status}). Skipping.")
                continue
            opt_fill = float(o.filled_avg_price)
        else:
            opt_fill = entry["quoted_premium"]

        entry["option_fill_price"] = opt_fill
        print(f"  Option fill price: ${opt_fill:.2f}")

        # ── B: confirm + place stock market order ─────────────────────────────
        stock_side   = OrderSide.SELL if option_type == "call" else OrderSide.BUY
        action_label = "SELL 100 shares (short)" if option_type == "call" else "BUY 100 shares (long)"

        dec = _ask(f"Market {action_label} on {ticker}?")
        if dec == "q":
            print("Quit. Remaining trades skipped.")
            _save_state(state)
            break
        if dec == "n":
            print("  Skipped.")
            continue

        try:
            stock_res = _market(client, ticker, STOCK_QTY, stock_side, dry_run)
            entry["stock_order_id"] = stock_res["id"]
            _save_state(state)
            tag = "DRY RUN" if dry_run else "SUBMITTED"
            print(f"  [{tag}]  stock order_id={stock_res['id']}")

            # ── C: poll for stock fill ────────────────────────────────────────
            print("  Waiting for stock fill ...")
            stock_fill = _poll_fill(client, stock_res["id"], dry_run)

            if stock_fill is None:
                # Can't place closing limit without fill price — warn and bail
                print("  WARNING: Could not confirm stock fill price.")
                if option_type == "call":
                    print(f"  Manually place: BUY to cover {ticker} 100 shares")
                    print(f"  Limit = (your actual short fill price) − ${opt_fill:.2f}")
                else:
                    print(f"  Manually place: SELL {ticker} 100 shares")
                    print(f"  Limit = (your actual long fill price) + ${opt_fill:.2f}")
                continue

            entry["stock_fill_price"] = stock_fill
            print(f"  Stock fill price: ${stock_fill:.2f}")

            # ── D: closing limit on the stock ─────────────────────────────────
            # Profit on stock leg = option premium paid → position breaks even on stock
            if option_type == "call":
                # Shorted stock at S → buy to cover at S - opt_fill  (profit = premium)
                close_side  = OrderSide.BUY
                close_price = round(stock_fill - opt_fill, 2)
                close_desc  = f"BUY to cover {ticker} 100 @ ${close_price:.2f}"
            else:
                # Bought stock at S → sell at S + opt_fill  (profit = premium)
                close_side  = OrderSide.SELL
                close_price = round(stock_fill + opt_fill, 2)
                close_desc  = f"SELL {ticker} 100 @ ${close_price:.2f}"

            print(f"  Placing closing limit: {close_desc}")
            close_res = _limit(client, ticker, STOCK_QTY, close_side, close_price, dry_run)
            entry["closing_order_id"] = close_res["id"]
            _save_state(state)
            print(f"  [{tag}]  closing order_id={close_res['id']}")

        except Exception as exc:
            print(f"  ERROR: {exc}")
            _save_state(state)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{DIV}")
    print("  PASS 2 SUMMARY")
    print(DIV)
    for e in pending:
        parts = [f"{e['ticker']:8s} {e['option_type'].upper():4s}"]
        if e.get("option_fill_price") is not None:
            parts.append(f"opt_fill=${e['option_fill_price']:.2f}")
        if e.get("stock_order_id"):
            parts.append(f"stock_order={e['stock_order_id']}")
        if e.get("stock_fill_price") is not None:
            parts.append(f"stock_fill=${e['stock_fill_price']:.2f}")
        if e.get("closing_order_id"):
            parts.append(f"close_order={e['closing_order_id']}")
        print("  " + "  |  ".join(parts))
    print(DIV)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Alpaca married-options semi-automatic placer"
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--pass1", action="store_true",
                   help="Place option buy orders from BQ table")
    g.add_argument("--pass2", action="store_true",
                   help="Place stock leg + closing limit (run after options fill)")
    p.add_argument("--date",    default=str(date.today()),
                   help="Trade date YYYY-MM-DD (default: today, pass1 only)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show all steps without submitting to Alpaca")
    return p.parse_args()


def main() -> None:
    a = _args()
    if a.pass1:
        pass1(a.date, a.dry_run)
    else:
        pass2(a.dry_run)


if __name__ == "__main__":
    main()
