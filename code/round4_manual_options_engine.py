import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict


# ---------- Data classes ----------

@dataclass
class Contract:
    symbol: str
    ctype: str  # 'underlying','call','put','binary_put','knockout_put','chooser'
    strike: float | None = None
    expiry_days: int = 0
    bid: float = 0.0
    ask: float = 0.0
    barrier: float | None = None
    payout: float | None = None
    decision_days: int | None = None


@dataclass
class Position:
    symbol: str         # e.g. 'AC_50_P'
    side: str           # 'BUY' or 'SELL'
    qty: int            # number of contracts
    entry: float        # your trade price


# ---------- Core GBM simulation ----------

def simulate_paths(
    S0: float,
    sigma: float,
    days_per_year: int,
    steps_per_day: int,
    total_days: int,
    n_paths: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate GBM paths with zero drift.
    Returns array shape (n_paths, n_steps+1).
    """
    dt = 1.0 / (days_per_year * steps_per_day)
    steps = int(total_days * steps_per_day)

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(size=(n_paths, steps))

    drift = (-0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * z
    log_returns = drift + diffusion

    log_price = np.cumsum(log_returns, axis=1)
    S = S0 * np.exp(np.column_stack([np.zeros(n_paths), log_price]))
    return S


# ---------- Payoff on paths ----------

def payoff_on_paths(
    contracts: List[Contract],
    paths: np.ndarray,
    days_per_year: int,
    steps_per_day: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute payoff, ITM flag, KO flag for each contract on all paths.
    """
    n_paths, n_steps_plus = paths.shape
    dt_days = 1.0 / steps_per_day

    results: Dict[str, Dict[str, np.ndarray]] = {}

    for c in contracts:
        pay = np.zeros(n_paths)
        itm = np.zeros(n_paths, dtype=bool)
        ko = np.zeros(n_paths, dtype=bool)

        if c.ctype == "underlying":
            pay = paths[:, -1]
            itm[:] = True

        else:
            expiry_idx = int(c.expiry_days * steps_per_day)
            expiry_price = paths[:, expiry_idx]

            if c.ctype == "call":
                pay = np.maximum(expiry_price - c.strike, 0.0)

            elif c.ctype == "put":
                pay = np.maximum(c.strike - expiry_price, 0.0)

            elif c.ctype == "binary_put":
                pay = np.where(expiry_price < c.strike, c.payout, 0.0)

            elif c.ctype == "knockout_put":
                barrier_hits = paths[:, : expiry_idx + 1] < c.barrier
                knocked = barrier_hits.any(axis=1)
                ko = knocked
                raw_put = np.maximum(c.strike - expiry_price, 0.0)
                pay = np.where(knocked, 0.0, raw_put)

            elif c.ctype == "chooser":
                decision_idx = int(c.decision_days * steps_per_day)
                decision_price = paths[:, decision_idx]
                is_call = decision_price >= c.strike
                call_pay = np.maximum(expiry_price - c.strike, 0.0)
                put_pay = np.maximum(c.strike - expiry_price, 0.0)
                pay = np.where(is_call, call_pay, put_pay)

            itm = pay > 0.0

        results[c.symbol] = {
            "payoff": pay,
            "itm": itm,
            "ko": ko,
        }

    return results


# ---------- Fair / Edge per contract ----------

def compute_fair_values(
    contracts: List[Contract],
    results: Dict[str, Dict[str, np.ndarray]],
) -> pd.DataFrame:
    rows = []

    for c in contracts:
        r = results[c.symbol]
        payoff = r["payoff"]
        itm = r["itm"]
        ko = r["ko"]

        mean = payoff.mean()
        var = payoff.var(ddof=1) if len(payoff) > 1 else 0.0
        se = np.sqrt(var / len(payoff)) if len(payoff) > 0 else 0.0
        itm_prob = itm.mean()
        ko_prob = ko.mean()

        buy_edge = mean - c.ask
        sell_edge = c.bid - mean

        rows.append(
            {
                "symbol": c.symbol,
                "type": c.ctype,
                "bid": c.bid,
                "ask": c.ask,
                "fair": mean,
                "se": se,
                "buy_edge": buy_edge,
                "sell_edge": sell_edge,
                "itm_prob": itm_prob,
                "ko_prob": ko_prob,
            }
        )

    df = pd.DataFrame(rows)
    return df


# ---------- Portfolio PnL aggregation ----------

def portfolio_pnl_distribution(
    contracts: List[Contract],
    positions: List[Position],
    results: Dict[str, Dict[str, np.ndarray]],
    multiplier: float = 3000.0,
) -> np.ndarray:
    """
    Given contract results and positions, compute portfolio PnL per path.
    """
    payoff_map = {sym: res["payoff"] for sym, res in results.items()}
    n_paths = len(next(iter(payoff_map.values())))
    pnls = np.zeros(n_paths)

    for pos in positions:
        pay = payoff_map[pos.symbol]
        side = pos.side.upper()
        if side == "BUY":
            pnl_each = (pay - pos.entry) * pos.qty * multiplier
        else:  # SELL
            pnl_each = (pos.entry - pay) * pos.qty * multiplier
        pnls += pnl_each

    return pnls


def summarize_portfolio_pnls(pnls: np.ndarray) -> dict:
    mean_pnl = pnls.mean()
    pcts = np.percentile(pnls, [1, 5, 50, 95, 99])
    min_pnl = pnls.min()
    max_pnl = pnls.max()
    se_pnl = pnls.std(ddof=1) / np.sqrt(len(pnls))

    return {
        "mean": mean_pnl,
        "p1": pcts[0],
        "p5": pcts[1],
        "p50": pcts[2],
        "p95": pcts[3],
        "p99": pcts[4],
        "min": min_pnl,
        "max": max_pnl,
        "se": se_pnl,
    }


# ---------- Plots ----------

def plot_price_paths(
    paths: np.ndarray,
    steps_per_day: int,
    outfile: str = "price_paths.png",
    n_show: int = 10,
):
    steps = paths.shape[1] - 1
    t_days = np.arange(steps + 1) / steps_per_day

    plt.figure(figsize=(8, 4))
    for i in range(min(n_show, paths.shape[0])):
        plt.plot(t_days, paths[i], alpha=0.6)
    plt.xlabel("Days")
    plt.ylabel("Aether Price")
    plt.title("Sample Aether Price Paths")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_portfolio_pnl_hist(pnls: np.ndarray, outfile: str = "portfolio_pnl_hist.png"):
    plt.figure(figsize=(7, 4))
    plt.hist(pnls, bins=60, alpha=0.8, color="steelblue")
    plt.axvline(pnls.mean(), color="red", linestyle="--", label="Mean")
    plt.xlabel("Portfolio PnL")
    plt.ylabel("Frequency")
    plt.title("Portfolio PnL Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_terminal_vs_pnl(
    paths: np.ndarray,
    pnls: np.ndarray,
    outfile: str = "terminal_vs_pnl.png",
):
    ST = paths[:, -1]
    plt.figure(figsize=(7, 4))
    plt.scatter(ST, pnls, s=4, alpha=0.4)
    plt.xlabel("Terminal Aether Price")
    plt.ylabel("Portfolio PnL")
    plt.title("Terminal Price vs Portfolio PnL")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


# ---------- Unified demo_run (entry point) ----------

def demo_run():
    # 1) Define contracts
    contracts = [
        Contract("AC", "underlying", expiry_days=0, bid=49.975, ask=50.025),
        Contract("AC_50_P", "put", strike=50, expiry_days=21, bid=12, ask=12.05),
        Contract("AC_50_C", "call", strike=50, expiry_days=21, bid=12, ask=12.05),
        Contract("AC_35_P", "put", strike=35, expiry_days=21, bid=4.33, ask=4.35),
        Contract("AC_40_P", "put", strike=40, expiry_days=21, bid=6.5, ask=6.55),
        Contract("AC_45_P", "put", strike=45, expiry_days=21, bid=9.05, ask=9.1),
        Contract("AC_60_C", "call", strike=60, expiry_days=21, bid=8.8, ask=8.85),
        Contract("AC_50_P_2", "put", strike=50, expiry_days=14, bid=9.7, ask=9.75),
        Contract("AC_50_C_2", "call", strike=50, expiry_days=14, bid=9.7, ask=9.75),
        Contract(
            "AC_50_CO",
            "chooser",
            strike=50,
            expiry_days=21,
            decision_days=14,
            bid=22.2,
            ask=22.3,
        ),
        Contract(
            "AC_40_BP",
            "binary_put",
            strike=40,
            expiry_days=21,
            payout=10,
            bid=5,
            ask=5.1,
        ),
        Contract(
            "AC_45_KO",
            "knockout_put",
            strike=45,
            expiry_days=21,
            barrier=35,
            bid=0.15,
            ask=0.175,
        ),
    ]

    # 2) Define your portfolio positions HERE
    positions = [
        
    Position("AC_50_C",   "BUY", 2, 12.05),  
    Position("AC_50_P",   "BUY", 1, 12.05),  
    Position("AC_60_C",   "SELL", 1, 8.85),  
    Position("AC_40_P",   "BUY", 1, 6.55),
    Position("AC_35_P",   "BUY", 1, 4.35),
    Position("AC_50_C_2", "BUY", 1, 9.75),
      
    ]

    # 3) Model parameters
    S0 = 50
    sigma = 2.51
    days_per_year = 252
    steps_per_day = 4
    total_days = max(c.expiry_days for c in contracts)
    n_paths = 50000
    seed = 42
    multiplier = 3000.0

    # 4) Simulate paths
    paths = simulate_paths(
        S0, sigma, days_per_year, steps_per_day, total_days, n_paths, seed
    )

    # 5) Per-contract payoffs
    results = payoff_on_paths(contracts, paths, days_per_year, steps_per_day)

    # 6) Fair / Edge table
    df = compute_fair_values(contracts, results)
    df.to_csv("aether_fair_values.csv", index=False)
    print("=== Contract Fair Values ===")
    print(df.to_string(index=False))

    # 7) Portfolio PnL distribution
    pnls = portfolio_pnl_distribution(
        contracts, positions, results, multiplier=multiplier
    )
    summary = summarize_portfolio_pnls(pnls)
    print("\n=== Portfolio PnL Summary ===")
    for k, v in summary.items():
        print(f"{k:>4}: {v:,.2f}")

    # 8) Plots
    plot_price_paths(paths, steps_per_day, "price_paths.png")
    plot_portfolio_pnl_hist(pnls, "portfolio_pnl_hist.png")
    plot_terminal_vs_pnl(paths, pnls, "terminal_vs_pnl.png")
    print("\nCharts saved: price_paths.png, portfolio_pnl_hist.png, terminal_vs_pnl.png")


if __name__ == "__main__":
    demo_run()