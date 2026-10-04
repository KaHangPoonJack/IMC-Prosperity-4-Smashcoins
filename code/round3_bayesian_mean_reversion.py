from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import math
import json

class Trader:
    def __init__(self):
        self.LIMITS = {
            "HYDROGEL_PACK": 200,
            "VELVETFRUIT_EXTRACT": 200,
            # 期權產品列表與限制
            "VEV_5000": 200, 
            "VEV_5100": 200, "VEV_5200": 200, "VEV_5300": 200
        }
        
        self.CONFIG = {
            "HYDROGEL_PACK": {
                "BASE_THRESHOLD": 0.005,
                "Z_SCORE": 2.5,
                "COOL_DOWN": 5000,
                "UNIT_PCT": 0.2
            },
            "VELVETFRUIT_EXTRACT": {
                "BASE_THRESHOLD": 0.002,
                "Z_SCORE": 1.8,
                "COOL_DOWN": 2000,
                "UNIT_PCT": 0.2  # 依照要求設為 20%
            }
        }

        self.VEV_OPTIONS = ["VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300"]
        self.INITIAL_UNCERTAINTY = 5.0
        self.OBS_SIGMA_BASE = 2.0
        self.VOLATILITY_WINDOW = 20

    def bayesian_update(self, prior_mu, prior_sigma, obs_price, obs_sigma):
        epsilon = 1e-9
        p_prec = 1.0 / (prior_sigma ** 2 + epsilon)
        o_prec = 1.0 / (obs_sigma ** 2 + epsilon)
        post_mu = (prior_mu * p_prec + obs_price * o_prec) / (p_prec + o_prec)
        post_sigma = math.sqrt(1.0 / (p_prec + o_prec) + epsilon)
        return post_mu, post_sigma

    def calculate_volatility(self, prices: List[float]) -> float:
        if len(prices) < 2: return 0.001
        mean_price = sum(prices) / len(prices)
        variance = sum((p - mean_price) ** 2 for p in prices) / len(prices)
        return math.sqrt(variance) / mean_price if mean_price > 0 else 0.001

    def run(self, state: TradingState):
        result = {}
        conversions = 0
        current_timestamp = state.timestamp
        
        if state.traderData:
            try:
                data = json.loads(state.traderData)
                fair_values = data.get("fair_values", {})
                uncertainties = data.get("uncertainties", {})
                price_history = data.get("price_history", {})
                last_trade_time = data.get("last_trade_time", {})
            except:
                fair_values, uncertainties, price_history, last_trade_time = {}, {}, {}, {}
        else:
            fair_values, uncertainties, price_history, last_trade_time = {}, {}, {}, {}

        # 我們只對基礎資產跑策略循環，Option 採取「掛件」模式同步
        for product in ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"]:
            if product not in state.order_depths: continue
            
            cfg = self.CONFIG[product]
            order_depth: OrderDepth = state.order_depths[product]
            sell_orders = sorted(order_depth.sell_orders.items())
            buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
            if not sell_orders or not buy_orders: continue
            
            best_ask, _ = sell_orders[0]
            best_bid, _ = buy_orders[0]
            mid_price = (best_ask + best_bid) / 2.0
            
            if product not in fair_values:
                fair_values[product], uncertainties[product] = mid_price, self.INITIAL_UNCERTAINTY
                price_history[product], last_trade_time[product] = [], -cfg["COOL_DOWN"]
            
            price_history[product].append(mid_price)
            if len(price_history[product]) > self.VOLATILITY_WINDOW: price_history[product].pop(0)
            
            # 貝葉斯更新
            market_trades = state.market_trades.get(product, [])
            for trade in market_trades:
                obs_sigma = self.OBS_SIGMA_BASE / math.sqrt(max(abs(trade.quantity), 1))
                fair_values[product], uncertainties[product] = self.bayesian_update(
                    fair_values[product], uncertainties[product], trade.price, obs_sigma
                )
            
            current_fair = fair_values[product]
            current_pos = state.position.get(product, 0)
            vol = self.calculate_volatility(price_history[product])
            dynamic_threshold = cfg["BASE_THRESHOLD"] + (cfg["Z_SCORE"] * uncertainties[product] * vol)
            deviation_pct = (mid_price - current_fair) / current_fair
            
            is_cooled_down = (current_timestamp - last_trade_time[product]) >= cfg["COOL_DOWN"]
            executed_this_tick = False
            main_orders = []

            # --- 交易執行：主產品與期權同步 ---
            
            # A. 平倉/止盈邏輯
            if (current_pos > 0 and mid_price >= current_fair) or (current_pos < 0 and mid_price <= current_fair):
                # 1. 平掉主產品
                close_qty = -current_pos
                price = int(best_bid) if current_pos > 0 else int(best_ask)
                main_orders.append(Order(product, price, close_qty))
                
                # 2. 如果是 Velvetfruit，同步平掉所有 Option
                if product == "VELVETFRUIT_EXTRACT":
                    for opt in self.VEV_OPTIONS:
                        opt_pos = state.position.get(opt, 0)
                        if opt_pos != 0 and opt in state.order_depths:
                            opt_depth = state.order_depths[opt]
                            # 主動平倉價格
                            opt_price = int(max(opt_depth.buy_orders.keys())) if opt_pos > 0 else int(min(opt_depth.sell_orders.keys()))
                            result[opt] = [Order(opt, opt_price, -opt_pos)]
                
                executed_this_tick = True

            # B. 進場邏輯
            if is_cooled_down and not executed_this_tick:
                unit_size = int(self.LIMITS[product] * cfg["UNIT_PCT"])
                
                # 做多信號
                if deviation_pct < -dynamic_threshold and current_pos < self.LIMITS[product]:
                    order_qty = min(unit_size, self.LIMITS[product] - current_pos)
                    main_orders.append(Order(product, int(best_ask), order_qty))
                    
                    # Velvet 同步 Option Long
                    if product == "VELVETFRUIT_EXTRACT":
                        for opt in self.VEV_OPTIONS:
                            if opt in state.order_depths:
                                opt_limit = self.LIMITS[opt]
                                opt_pos = state.position.get(opt, 0)
                                opt_unit = int(opt_limit * cfg["UNIT_PCT"])
                                opt_qty = min(opt_unit, opt_limit - opt_pos)
                                if opt_qty > 0:
                                    opt_ask = int(min(state.order_depths[opt].sell_orders.keys()))
                                    result[opt] = [Order(opt, opt_ask, opt_qty)]
                    
                    last_trade_time[product] = current_timestamp
                
                # 做空信號
                elif deviation_pct > dynamic_threshold and current_pos > -self.LIMITS[product]:
                    order_qty = min(unit_size, self.LIMITS[product] + current_pos)
                    main_orders.append(Order(product, int(best_bid), -order_qty))
                    
                    # Velvet 同步 Option Short
                    if product == "VELVETFRUIT_EXTRACT":
                        for opt in self.VEV_OPTIONS:
                            if opt in state.order_depths:
                                opt_limit = self.LIMITS[opt]
                                opt_pos = state.position.get(opt, 0)
                                opt_unit = int(opt_limit * cfg["UNIT_PCT"])
                                opt_qty = min(opt_unit, opt_limit + opt_pos)
                                if opt_qty > 0:
                                    opt_bid = int(max(state.order_depths[opt].buy_orders.keys()))
                                    result[opt] = [Order(opt, opt_bid, -opt_qty)]
                    
                    last_trade_time[product] = current_timestamp

            if main_orders:
                result[product] = main_orders

        new_trader_data = json.dumps({
            "fair_values": fair_values,
            "uncertainties": uncertainties,
            "price_history": price_history,
            "last_trade_time": last_trade_time
        })

        return result, conversions, new_trader_data