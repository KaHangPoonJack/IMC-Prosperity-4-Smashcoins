"""
Round 5 — Directional Hold Trader
===================================
Takes maximum position in the direction of confirmed linear trends
at the start of the round, then holds to expiry.

Evidence (from detect_patterns.py):
  PEBBLES_XS       R²=0.900  slope=-0.00159  → short
  PEBBLES_S        R²=0.798  slope=-0.00086  → short
  MICROCHIP_OVAL   R²=0.912  slope=-0.00171  → short
  MICROCHIP_RECTANGLE R²=0.571 slope=-0.00066 → short
  MICROCHIP_TRIANGLE  R²=0.643 slope=-0.00077 → short
  UV_VISOR_AMBER   R²=0.912  slope=-0.00110  → short
  PEBBLES_XL       R²=0.687  slope=+0.00170  → long
  MICROCHIP_SQUARE R²=0.702  slope=+0.00177  → long
  UV_VISOR_MAGENTA R²=0.675  slope=+0.00058  → long
"""

from datamodel import TradingState, Order
from typing import Dict, List

POS_LIMIT = 10

# Target position per product: +POS_LIMIT = hold max long, -POS_LIMIT = hold max short
HOLD_TARGETS: Dict[str, int] = {
    # ── Shorts ────────────────────────────────────────────────
    "PEBBLES_XS":           -POS_LIMIT,  # R²=0.900  strongest downtrend
    "MICROCHIP_OVAL":       -POS_LIMIT,  # R²=0.912  strongest downtrend
    "UV_VISOR_AMBER":       -POS_LIMIT,  # R²=0.912  strongest downtrend
    # ── Longs ─────────────────────────────────────────────────
    "PEBBLES_XL":           +POS_LIMIT,  # R²=0.687  strongest uptrend
    "UV_VISOR_MAGENTA":     +POS_LIMIT,  # R²=0.675  moderate uptrend
    "MICROCHIP_SQUARE":     +POS_LIMIT,  # R²=0.702  strongest uptrend
}


class Trader:

    def run(self, state: TradingState):
        orders: Dict[str, List[Order]] = {}

        for sym, target in HOLD_TARGETS.items():
            od = state.order_depths.get(sym)
            if od is None:
                continue

            pos  = state.position.get(sym, 0)
            want = target - pos   # how many units we still need to reach target
            if want == 0:
                continue          # already at target, hold — nothing to do

            buy_cap  = POS_LIMIT - pos   # max additional buys allowed
            sell_cap = POS_LIMIT + pos   # max additional sells allowed
            sym_orders = []

            if want > 0:
                # Need to buy — walk ask levels until we fill want units
                remaining = min(want, buy_cap)
                for px in sorted(od.sell_orders):
                    if remaining <= 0:
                        break
                    qty = min(remaining, abs(od.sell_orders[px]))
                    if qty > 0:
                        sym_orders.append(Order(sym, px, qty))
                        remaining -= qty

            else:
                # Need to sell — walk bid levels until we fill -want units
                remaining = min(-want, sell_cap)
                for px in sorted(od.buy_orders, reverse=True):
                    if remaining <= 0:
                        break
                    qty = min(remaining, od.buy_orders[px])
                    if qty > 0:
                        sym_orders.append(Order(sym, px, -qty))
                        remaining -= qty

            if sym_orders:
                orders[sym] = sym_orders

        return orders, 0, ""