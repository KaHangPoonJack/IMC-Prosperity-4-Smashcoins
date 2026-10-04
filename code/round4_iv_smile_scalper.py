import math
from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List


# ── Pure-Python Black-Scholes (no numpy/scipy — competition safe) ─────────────

def _ncdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2))

def _bs_price(S: float, K: float, T: float, v: float) -> float:
    if T < 1e-9 or v < 1e-9:
        return max(0.0, S - K)
    sq = v * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * v * v * T) / sq
    return S * _ncdf(d1) - K * _ncdf(d1 - sq)

def _bs_delta(S: float, K: float, T: float, v: float) -> float:
    if T < 1e-9 or v < 1e-9:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + 0.5 * v * v * T) / (v * math.sqrt(T))
    return _ncdf(d1)

def _calc_iv(price: float, S: float, K: float, T: float) -> float:
    intrinsic = max(0.0, S - K)
    if price <= intrinsic + 0.01 or T < 1e-9:
        return float('nan')
    lo, hi = 1e-4, 4.0
    if _bs_price(S, K, T, hi) < price:
        return float('nan')
    for _ in range(80):
        m = (lo + hi) * 0.5
        p = _bs_price(S, K, T, m)
        if abs(p - price) < 1e-6:
            return m
        if p > price:
            hi = m
        else:
            lo = m
    return (lo + hi) * 0.5


# ── Book-walking helpers ──────────────────────────────────────────────────────

def _walk_buy(sym: str, od: OrderDepth, max_qty: int, max_price: float) -> List[Order]:
    """Buy up to max_qty lots, consuming ask levels in price order while price <= max_price."""
    result, rem = [], max_qty
    for px in sorted(od.sell_orders.keys()):
        if rem <= 0 or px > max_price:
            break
        qty = min(rem, abs(od.sell_orders[px]))
        if qty > 0:
            result.append(Order(sym, px, qty))
            rem -= qty
    return result

def _walk_sell(sym: str, od: OrderDepth, max_qty: int, min_price: float) -> List[Order]:
    """Sell up to max_qty lots, consuming bid levels in price order while price >= min_price."""
    result, rem = [], max_qty
    for px in sorted(od.buy_orders.keys(), reverse=True):
        if rem <= 0 or px < min_price:
            break
        qty = min(rem, od.buy_orders[px])
        if qty > 0:
            result.append(Order(sym, px, -qty))
            rem -= qty
    return result


class Trader:
    # ════════════════════════════════════════════════════════════════════════
    #  TUNABLE PARAMETERS
    # ════════════════════════════════════════════════════════════════════════

    # Volatility smile: fair_iv = SMILE_A*m² + SMILE_B*m + SMILE_C
    # m = ln(S/K) / sqrt(T_years)   (m = 0 at ATM)
    # To try the clean refit (R²=0.93): swap to 0.2081, 0.00370, 0.1904
    SMILE_A: float = 0.1762
    SMILE_B: float = 0.007389
    SMILE_C: float = 0.1946

    # Open/add when |iv_dev| exceeds ENTRY_THRESHOLD.
    # Close when |iv_dev| drops below EXIT_THRESHOLD.
    ENTRY_THRESHOLD: float = 0.015
    EXIT_THRESHOLD:  float = 0.005

    # Conviction sizing: target = clamp(BASE_QTY × |iv_dev|/ENTRY_THRESHOLD, 1, MAX_QTY)
    #   iv_dev = 0.015 → 25 lots   iv_dev = 0.045 → 75   iv_dev ≥ 0.090 → 150 (cap)
    BASE_QTY: int = 25
    MAX_QTY:  int = 150

    # Per-strike position cap. Hard limit = 300; leave 100 buffer.
    OPT_LIMIT: int = 200

    # VELVETFRUIT_EXTRACT position hard cap.
    HEDGE_LIMIT: int = 200

    # ── Delta Wall fix ────────────────────────────────────────────────────
    # Max absolute net-delta allowed across ALL voucher positions combined.
    # Must be ≤ HEDGE_LIMIT (200) or you cannot fully hedge.
    #
    # Without this: 8 strikes × 200 lots × 0.5 avg delta = 800 potential
    # delta — 4× more than the 200-lot hedge capacity.
    # With MAX_PORTFOLIO_DELTA = 175, the bot stops adding new positions on
    # any strike once the portfolio delta budget is full.
    MAX_PORTFOLIO_DELTA: float = 175.0

    # 1.0 = fully delta-neutral each tick.
    # 0.5 = hedge half the delta (less underlying churn, more residual risk).
    HEDGE_RATIO: float = 1.0

    # Skip options with fair_price below this (near-expiry OTM: unreliable IV)
    MIN_FAIR_PRICE: float = 0.5

    # ════════════════════════════════════════════════════════════════════════
    #  CONSTANTS
    # ════════════════════════════════════════════════════════════════════════

    UNDERLYING = "VELVETFRUIT_EXTRACT"

    # VEV_4000/4500 excluded: deep ITM → vega≈0 → IV solver numerically
    # unstable (observed range 1%–105%, std 37%).
    STRIKES: Dict[str, int] = {
        "VEV_5000": 5000, "VEV_5100": 5100, "VEV_5200": 5200,
        "VEV_5300": 5300, "VEV_5400": 5400, "VEV_5500": 5500,
        "VEV_6000": 6000, "VEV_6500": 6500,
    }

    EXPIRY_TS: int = 8_000_000

    # ════════════════════════════════════════════════════════════════════════
    #  MAIN LOOP
    # ════════════════════════════════════════════════════════════════════════

    def run(self, state: TradingState):
        orders: Dict[str, List[Order]] = {}
        ts = state.timestamp

        tte_days = max((self.EXPIRY_TS - ts) / 1_000_000, 1e-6)
        T        = tte_days / 252.0

        und = self.UNDERLYING
        if und not in state.order_depths:
            return {}, 0, ""
        od_und = state.order_depths[und]
        if not od_und.buy_orders or not od_und.sell_orders:
            return {}, 0, ""

        S       = (max(od_und.buy_orders) + min(od_und.sell_orders)) / 2.0
        pos_und = state.position.get(und, 0)

        # ── PASS 1: pre-compute smile values + existing portfolio delta ───────
        #
        # Done up-front so the delta budget check in Pass 2 can see the total
        # existing exposure before we start adding new orders.
        pre: Dict[str, dict] = {}
        existing_net_delta   = 0.0
        for sym, K in self.STRIKES.items():
            m      = math.log(S / K) / math.sqrt(T)
            fiv    = max(self.SMILE_A*m*m + self.SMILE_B*m + self.SMILE_C, 0.01)
            delta  = _bs_delta(S, K, T, fiv)
            fprice = _bs_price(S, K, T, fiv)
            pre[sym] = {'K': K, 'fair_iv': fiv, 'delta': delta, 'fair_price': fprice}
            existing_net_delta += state.position.get(sym, 0) * delta

        # ── PASS 2: generate option orders with delta budget tracking ─────────
        #
        # running_delta = delta from CURRENT positions + delta from new orders
        # this tick.  The hedge at the end targets -running_delta, so it
        # accounts for orders that will fill this tick.
        running_delta = existing_net_delta

        for sym, p in pre.items():
            fair_iv = p['fair_iv']
            delta   = p['delta']
            fprice  = p['fair_price']

            od = state.order_depths.get(sym)
            if od is None or not od.buy_orders or not od.sell_orders:
                continue
            if fprice < self.MIN_FAIR_PRICE:
                continue

            pos = state.position.get(sym, 0)
            bid = max(od.buy_orders)
            ask = min(od.sell_orders)
            mid = (bid + ask) / 2.0

            iv_mkt = _calc_iv(mid, S, p['K'], T)
            if math.isnan(iv_mkt):
                continue
            iv_dev = iv_mkt - fair_iv

            # ── Signal zone → target position → order quantity ────────────
            #
            # BUY  zone (iv_dev < -ENTRY): option underpriced → go long.
            #   want = max(0, target - pos) so we never reduce a long within
            #   the buy zone (avoids selling at bid when option still cheap),
            #   but we DO close a short here when signal flips.
            #
            # SELL zone (iv_dev > +ENTRY): option overpriced → go short.
            #   want = min(0, target - pos) — symmetric logic.
            #
            # EXIT zone (|iv_dev| < EXIT): edge gone → close completely at
            #   market price regardless of fair-price filter.
            #
            # HOLD zone (EXIT ≤ |iv_dev| ≤ ENTRY): keep current position.

            if iv_dev < -self.ENTRY_THRESHOLD:
                mult   = abs(iv_dev) / self.ENTRY_THRESHOLD
                target = min(round(self.BASE_QTY * mult), self.MAX_QTY, self.OPT_LIMIT)
                want   = max(0, target - pos)

            elif iv_dev > self.ENTRY_THRESHOLD:
                mult   = iv_dev / self.ENTRY_THRESHOLD
                target = -min(round(self.BASE_QTY * mult), self.MAX_QTY, self.OPT_LIMIT)
                want   = min(0, target - pos)

            elif abs(iv_dev) < self.EXIT_THRESHOLD and pos != 0:
                target = 0
                want   = -pos

            else:
                continue   # hold zone

            if want == 0:
                continue

            # ── Delta budget constraint ───────────────────────────────────
            #
            # Prevent |net_delta| from exceeding MAX_PORTFOLIO_DELTA so the
            # hedge position stays within HEDGE_LIMIT.
            # Orders that REDUCE delta (exits, position flips) are never
            # constrained — they always move toward zero.
            new_rd = running_delta + want * delta
            if new_rd > self.MAX_PORTFOLIO_DELTA:
                want = max(0, int((self.MAX_PORTFOLIO_DELTA - running_delta) / max(delta, 1e-4)))
            elif new_rd < -self.MAX_PORTFOLIO_DELTA:
                want = min(0, int((-self.MAX_PORTFOLIO_DELTA - running_delta) / max(abs(delta), 1e-4)))

            if want == 0:
                print(f"SKIP|{sym}|delta_budget_full|rd={running_delta:.1f}")
                continue

            # ── Walk the book ─────────────────────────────────────────────
            #
            # ENTRY orders: only fill at prices that preserve positive edge.
            #   BUY  → consume ask levels while price ≤ fair_price
            #   SELL → consume bid levels while price ≥ fair_price
            #
            # EXIT orders: accept market price (no edge filter — we're
            #   closing, not opening, and the edge has already collapsed).
            cap_buy  = self.OPT_LIMIT - pos
            cap_sell = self.OPT_LIMIT + pos

            if target == 0:
                # Exit at best market price
                if want > 0:
                    qty = min(want, cap_buy, abs(od.sell_orders.get(ask, 0)))
                    sym_orders = [Order(sym, ask, qty)] if qty > 0 else []
                else:
                    qty = min(-want, cap_sell, od.buy_orders.get(bid, 0))
                    sym_orders = [Order(sym, bid, -qty)] if qty > 0 else []
            else:
                # Entry/add: walk book up to fair price
                if want > 0:
                    sym_orders = _walk_buy(sym, od, min(want, cap_buy), fprice)
                else:
                    sym_orders = _walk_sell(sym, od, min(-want, cap_sell), fprice)

            if sym_orders:
                filled = sum(o.quantity for o in sym_orders)
                if want < 0:
                    filled = -filled
                running_delta += filled * delta
                orders[sym] = sym_orders

            print(f"OPT|{sym}|iv_dev={iv_dev:+.4f}|tgt={target}|want={want}|fills={len(sym_orders)}|rd={running_delta:.1f}")

        # ── Portfolio-wide delta hedge ─────────────────────────────────────────
        #
        # running_delta = existing positions + orders placed this tick.
        # Hedging against this forward-looking delta (not just historical
        # positions) prevents a one-tick lag in the hedge after large fills.
        raw_target  = -running_delta * self.HEDGE_RATIO
        target_und  = max(-self.HEDGE_LIMIT, min(self.HEDGE_LIMIT, round(raw_target)))
        hedge_delta = target_und - pos_und

        if abs(hedge_delta) >= 1:
            if hedge_delta > 0:
                ask_und = min(od_und.sell_orders)
                qty = min(hedge_delta,
                          abs(od_und.sell_orders.get(ask_und, 0)),
                          self.HEDGE_LIMIT - pos_und)
                if qty > 0:
                    orders[und] = [Order(und, ask_und, qty)]
            else:
                bid_und = max(od_und.buy_orders)
                qty = min(-hedge_delta,
                          od_und.buy_orders.get(bid_und, 0),
                          self.HEDGE_LIMIT + pos_und)
                if qty > 0:
                    orders[und] = [Order(und, bid_und, -qty)]

        print(f"HEDGE|ts={ts}|exist_Δ={existing_net_delta:.2f}|run_Δ={running_delta:.2f}|pos_und={pos_und}|tgt_und={target_und}")
        return orders, 0, ""
