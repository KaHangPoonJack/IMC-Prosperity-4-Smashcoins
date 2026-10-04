from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import json
import math

class Trader:
    def __init__(self):
        self.limits = {'ASH_COATED_OSMIUM': 20, 'INTARIAN_PEPPER_ROOT': 20}
        
        # 1. Product-Specific Tuning
        self.params = {
            'ASH_COATED_OSMIUM': {
                'ema_alpha': 0.1,        # Slower alpha: ignore noise, trust the mean
                'lean_aggression': 1.5,  # Stronger skew to dump inventory
                'base_spread': 1.0
            },
            'INTARIAN_PEPPER_ROOT': {
                'ema_alpha': 0.4,        # Faster alpha: ride the trend quickly
                'lean_aggression': 2.5,  # Needs aggressive inventory management
                'base_spread': 2.0
            }
        }
        
        self.state_data = {
            'fair_ema': {},
            'price_history': {} # Replaced the empty 'volatility' dict with a history queue
        }

    def _get_wall_mid(self, od: OrderDepth):
        if not od.buy_orders or not od.sell_orders:
            return None
        best_bid, bid_vol = max(od.buy_orders.items())
        best_ask, ask_vol = min(od.sell_orders.items())
        # Volume-weighted mid-price gives edge to the side with larger resting volume
        return (best_bid * abs(ask_vol) + best_ask * bid_vol) / (bid_vol + abs(ask_vol))

    def run(self, state: TradingState):
        result = {}
        
        # Load state
        if state.traderData:
            try:
                self.state_data.update(json.loads(state.traderData))
            except Exception: 
                pass

        for product in state.order_depths:
            od = state.order_depths[product]
            orders: List[Order] = []
            pos = state.position.get(product, 0)
            limit = self.limits.get(product, 20)
            p_params = self.params.get(product)
            
            raw_mid = self._get_wall_mid(od)
            if raw_mid is None: continue

            # Maintain Rolling Price History for Volatility
            hist = self.state_data['price_history'].setdefault(product, [])
            hist.append(raw_mid)
            if len(hist) > 10:  # 10-step rolling window
                hist.pop(0)

            # Calculate Rolling Standard Deviation (Volatility)
            volatility = 0.0
            if len(hist) > 1:
                mean_price = sum(hist) / len(hist)
                variance = sum((x - mean_price) ** 2 for x in hist) / len(hist)
                volatility = math.sqrt(variance)

            # Exponential Moving Average (Fair Value)
            prev_ema = self.state_data['fair_ema'].get(product, raw_mid)
            alpha = p_params['ema_alpha']
            fair = (alpha * raw_mid) + (1 - alpha) * prev_ema
            self.state_data['fair_ema'][product] = fair

            # 1. DYNAMIC POSITION SKEW
            pos_ratio = pos / limit
            skew = - (pos_ratio * p_params['lean_aggression'])
            adjusted_fair = fair + skew

            # 2. ADAPTIVE SPREAD
            # Expand our required edge automatically when the asset is jumping around
            dynamic_spread = p_params['base_spread'] + (volatility * 0.4)

            # 3. AGGRESSIVE SNIPING (Market Taking)
            sorted_asks = sorted(od.sell_orders.items())
            for price, vol in sorted_asks:
                if price <= adjusted_fair - 0.5 and pos < limit:
                    buy_amount = min(-vol, limit - pos)
                    orders.append(Order(product, price, buy_amount))
                    pos += buy_amount

            sorted_bids = sorted(od.buy_orders.items(), reverse=True)
            for price, vol in sorted_bids:
                if price >= adjusted_fair + 0.5 and pos > -limit:
                    sell_amount = max(-vol, -limit - pos)
                    orders.append(Order(product, price, sell_amount))
                    pos += sell_amount

            # 4. PASSIVE QUOTING (Market Making)
            best_bid = max(od.buy_orders.keys()) if od.buy_orders else int(adjusted_fair - 1)
            best_ask = min(od.sell_orders.keys()) if od.sell_orders else int(adjusted_fair + 1)

            if pos < limit:
                # Floor our target to ensure we capture our required spread
                target_bid = int(math.floor(adjusted_fair - dynamic_spread))
                # Penny the best bid, but NEVER bid higher than our target or cross the best ask
                bid_price = min(target_bid, best_bid + 1, best_ask - 1)
                orders.append(Order(product, bid_price, limit - pos))

            if pos > -limit:
                # Ceil our target
                target_ask = int(math.ceil(adjusted_fair + dynamic_spread))
                # Penny the best ask, but NEVER ask lower than our target or cross the best bid
                ask_price = max(target_ask, best_ask - 1, best_bid + 1)
                orders.append(Order(product, ask_price, -limit - pos))

            result[product] = orders

        return result, 0, json.dumps(self.state_data)