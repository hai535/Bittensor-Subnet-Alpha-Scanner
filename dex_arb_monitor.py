#!/usr/bin/env python3
"""
跨链套利路径监控
监控完整套利路径：在A链swap → 跨链桥接 → 在B链swap，计算扣费后是否有利润
"""

import requests
import time
import json
import sys
import os
import argparse
from datetime import datetime
from dataclasses import dataclass

# ============ 配置 ============

# 利润阈值（美元），超过此值触发通知
PROFIT_THRESHOLD_USD = 3.0

# 利润率阈值（%），超过此值触发通知
PROFIT_THRESHOLD_PCT = 0.3

# 冷却时间（秒），同一路径不重复通知
COOLDOWN_SECONDS = 600

# 模拟投入金额（美元）
SIM_AMOUNT_USD = 500

# DexScreener API
DEXSCREENER_BASE = "https://api.dexscreener.com/latest/dex"

# 各链 Gas 费估算（美元）— swap一次的大概Gas费
GAS_COST = {
    "ethereum": 5.0,
    "bsc": 0.15,
    "arbitrum": 0.10,
    "base": 0.05,
    "optimism": 0.10,
}

# 跨链桥接费用估算（美元）— 从A链到B链
BRIDGE_COST = {
    ("bsc", "base"): 1.5,
    ("bsc", "arbitrum"): 1.5,
    ("bsc", "optimism"): 1.5,
    ("bsc", "ethereum"): 8.0,
    ("base", "bsc"): 1.5,
    ("base", "arbitrum"): 0.5,
    ("base", "ethereum"): 5.0,
    ("base", "optimism"): 0.5,
    ("arbitrum", "bsc"): 1.5,
    ("arbitrum", "base"): 0.5,
    ("arbitrum", "ethereum"): 5.0,
    ("arbitrum", "optimism"): 0.5,
    ("optimism", "bsc"): 1.5,
    ("optimism", "base"): 0.5,
    ("optimism", "arbitrum"): 0.5,
    ("optimism", "ethereum"): 5.0,
    ("ethereum", "bsc"): 8.0,
    ("ethereum", "base"): 5.0,
    ("ethereum", "arbitrum"): 5.0,
    ("ethereum", "optimism"): 5.0,
}

# DEX swap 滑点+手续费估算（%）
SWAP_FEE_PCT = 0.3  # 0.3% per swap

# 桥接滑点估算（%）
BRIDGE_SLIPPAGE_PCT = 0.05

# ============ 交易对配置 ============
# 每条链上我们关心的主要交易对地址（DexScreener pair address）
# 只选高流动性池

PAIRS = {
    # ETH/USDT 各链
    "ETH/USDT": {
        "ethereum": "0x11b815efb8f581194ae79006d24e0d814b7697f6",  # Uniswap V3
        "bsc": "0xBe141893E4c6AD9272e8C04BAB7E6a10604501a5",       # PancakeSwap V3
        "arbitrum": "0x641c00a822e8b671738d32a431a4fb6074e5c79d",   # Uniswap V3
        "base": "0xd0b53d9277642d899df5c87a3966a349a798f224",       # Uniswap V3
    },
    # ETH/USDC 各链
    "ETH/USDC": {
        "ethereum": "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",  # Uniswap V3
        "bsc": "0x539e0EBfffd39e54A0f7E5F8FEc40ade7933A664",       # PancakeSwap
        "arbitrum": "0xc31e54c7a869b9fcbecc14363cf510d1c41fa443",   # Uniswap V3
        "base": "0xd0b53d9277642d899df5c87a3966a349a798f224",       # Uniswap V3
    },
    # BNB/ETH on BSC
    "BNB/ETH": {
        "bsc": "0xD0e226f674bBf064f54aB47F42473fF80DB98CBA",       # PancakeSwap V3
    },
    # BNB/USDT on BSC
    "BNB/USDT": {
        "bsc": "0x172fcd41e0913e95784454622d1c3724f546f849",       # PancakeSwap V3
    },
}

# ============ 套利路径定义 ============
# 每条路径: (描述, 步骤列表)
# 步骤: ("swap", chain, from_token, to_token, pair_key) 或 ("bridge", from_chain, to_chain, token)

ARB_PATHS = [
    # === BNB → ETH 跨链套利 ===
    {
        "name": "BNB(BSC)→ETH→Bridge→USDT vs BNB直接换USDT",
        "input": "BNB",
        "steps": [
            # 路径A: BNB→ETH(BSC)→桥到Base→ETH→USDT(Base)
            # 路径B: BNB→USDT(BSC) 直接换
            # 比较两条路径的USDT产出
        ],
        "path_a": [
            ("swap", "bsc", "BNB", "ETH", "BNB/ETH"),
            ("bridge", "bsc", "base", "ETH"),
            ("swap", "base", "ETH", "USDT", "ETH/USDT"),
        ],
        "path_b": [
            ("swap", "bsc", "BNB", "USDT", "BNB/USDT"),
        ],
    },
    {
        "name": "BNB(BSC)→ETH→Bridge(ARB)→USDT vs BNB直接换USDT",
        "input": "BNB",
        "path_a": [
            ("swap", "bsc", "BNB", "ETH", "BNB/ETH"),
            ("bridge", "bsc", "arbitrum", "ETH"),
            ("swap", "arbitrum", "ETH", "USDT", "ETH/USDT"),
        ],
        "path_b": [
            ("swap", "bsc", "BNB", "USDT", "BNB/USDT"),
        ],
    },
    {
        "name": "BNB(BSC)→ETH→Bridge(ETH主网)→USDT vs BNB直接换USDT",
        "input": "BNB",
        "path_a": [
            ("swap", "bsc", "BNB", "ETH", "BNB/ETH"),
            ("bridge", "bsc", "ethereum", "ETH"),
            ("swap", "ethereum", "ETH", "USDT", "ETH/USDT"),
        ],
        "path_b": [
            ("swap", "bsc", "BNB", "USDT", "BNB/USDT"),
        ],
    },

    # === ETH 跨链USDT套利 ===
    # 在A链买ETH便宜 → 桥到B链 → 卖ETH贵
    {
        "name": "USDT(BSC)→ETH→Bridge(Base)→USDT",
        "input": "USDT",
        "path_a": [
            ("swap", "bsc", "USDT", "ETH", "ETH/USDT"),
            ("bridge", "bsc", "base", "ETH"),
            ("swap", "base", "ETH", "USDT", "ETH/USDT"),
        ],
        "path_b": "direct",  # 直接持有USDT不动，产出=输入
    },
    {
        "name": "USDT(BSC)→ETH→Bridge(ARB)→USDT",
        "input": "USDT",
        "path_a": [
            ("swap", "bsc", "USDT", "ETH", "ETH/USDT"),
            ("bridge", "bsc", "arbitrum", "ETH"),
            ("swap", "arbitrum", "ETH", "USDT", "ETH/USDT"),
        ],
        "path_b": "direct",
    },
    {
        "name": "USDT(Base)→ETH→Bridge(BSC)→USDT",
        "input": "USDT",
        "path_a": [
            ("swap", "base", "USDT", "ETH", "ETH/USDT"),
            ("bridge", "base", "bsc", "ETH"),
            ("swap", "bsc", "ETH", "USDT", "ETH/USDT"),
        ],
        "path_b": "direct",
    },
    {
        "name": "USDT(Base)→ETH→Bridge(ARB)→USDT",
        "input": "USDT",
        "path_a": [
            ("swap", "base", "USDT", "ETH", "ETH/USDT"),
            ("bridge", "base", "arbitrum", "ETH"),
            ("swap", "arbitrum", "ETH", "USDT", "ETH/USDT"),
        ],
        "path_b": "direct",
    },
    {
        "name": "USDT(ARB)→ETH→Bridge(Base)→USDT",
        "input": "USDT",
        "path_a": [
            ("swap", "arbitrum", "USDT", "ETH", "ETH/USDT"),
            ("bridge", "arbitrum", "base", "ETH"),
            ("swap", "base", "ETH", "USDT", "ETH/USDT"),
        ],
        "path_b": "direct",
    },
    {
        "name": "USDT(ARB)→ETH→Bridge(BSC)→USDT",
        "input": "USDT",
        "path_a": [
            ("swap", "arbitrum", "USDT", "ETH", "ETH/USDT"),
            ("bridge", "arbitrum", "bsc", "ETH"),
            ("swap", "bsc", "ETH", "USDT", "ETH/USDT"),
        ],
        "path_b": "direct",
    },
    {
        "name": "USDT(ETH主网)→ETH→Bridge(Base)→USDT",
        "input": "USDT",
        "path_a": [
            ("swap", "ethereum", "USDT", "ETH", "ETH/USDT"),
            ("bridge", "ethereum", "base", "ETH"),
            ("swap", "base", "ETH", "USDT", "ETH/USDT"),
        ],
        "path_b": "direct",
    },
    {
        "name": "USDT(ETH主网)→ETH→Bridge(ARB)→USDT",
        "input": "USDT",
        "path_a": [
            ("swap", "ethereum", "USDT", "ETH", "ETH/USDT"),
            ("bridge", "ethereum", "arbitrum", "ETH"),
            ("swap", "arbitrum", "ETH", "USDT", "ETH/USDT"),
        ],
        "path_b": "direct",
    },

    # === USDC路径 ===
    {
        "name": "USDC(BSC)→ETH→Bridge(Base)→USDC",
        "input": "USDC",
        "path_a": [
            ("swap", "bsc", "USDC", "ETH", "ETH/USDC"),
            ("bridge", "bsc", "base", "ETH"),
            ("swap", "base", "ETH", "USDC", "ETH/USDC"),
        ],
        "path_b": "direct",
    },
    {
        "name": "USDC(Base)→ETH→Bridge(BSC)→USDC",
        "input": "USDC",
        "path_a": [
            ("swap", "base", "USDC", "ETH", "ETH/USDC"),
            ("bridge", "base", "bsc", "ETH"),
            ("swap", "bsc", "ETH", "USDC", "ETH/USDC"),
        ],
        "path_b": "direct",
    },
    {
        "name": "USDC(ARB)→ETH→Bridge(Base)→USDC",
        "input": "USDC",
        "path_a": [
            ("swap", "arbitrum", "USDC", "ETH", "ETH/USDC"),
            ("bridge", "arbitrum", "base", "ETH"),
            ("swap", "base", "ETH", "USDC", "ETH/USDC"),
        ],
        "path_b": "direct",
    },
    {
        "name": "USDC(Base)→ETH→Bridge(ARB)→USDC",
        "input": "USDC",
        "path_a": [
            ("swap", "base", "USDC", "ETH", "ETH/USDC"),
            ("bridge", "base", "arbitrum", "ETH"),
            ("swap", "arbitrum", "ETH", "USDC", "ETH/USDC"),
        ],
        "path_b": "direct",
    },
]

# ============ 价格缓存 ============
_price_cache = {}  # (pair_key, chain) -> {"priceUsd": float, "priceNative": float, "liquidity": float}
_notified = {}     # path_name -> last_notify_time


def fetch_pair_data(pair_key: str, chain: str) -> dict | None:
    """获取交易对数据，带缓存"""
    cache_key = (pair_key, chain)
    if cache_key in _price_cache:
        return _price_cache[cache_key]

    pair_addr = PAIRS.get(pair_key, {}).get(chain)
    if not pair_addr:
        return None

    url = f"{DEXSCREENER_BASE}/pairs/{chain}/{pair_addr}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        pair = data.get("pair") or (data.get("pairs") or [None])[0]
        if not pair:
            return None

        result = {
            "priceUsd": float(pair.get("priceUsd", 0) or 0),
            "priceNative": float(pair.get("priceNative", 0) or 0),
            "baseSymbol": pair.get("baseToken", {}).get("symbol", "?"),
            "quoteSymbol": pair.get("quoteToken", {}).get("symbol", "?"),
            "liquidity": float(pair.get("liquidity", {}).get("usd", 0) or 0),
            "dex": pair.get("dexId", "unknown"),
        }
        _price_cache[cache_key] = result
        return result
    except Exception as e:
        print(f"  [ERROR] 获取 {pair_key} on {chain}: {e}")
        return None


def simulate_path(steps: list, input_usd: float) -> tuple[float, list[str], float]:
    """
    模拟一条套利路径
    返回: (最终USD价值, 步骤详情列表, 总费用USD)
    """
    current_usd = input_usd
    details = []
    total_fees = 0.0

    for step in steps:
        if step[0] == "swap":
            _, chain, from_token, to_token, pair_key = step
            pair_data = fetch_pair_data(pair_key, chain)
            if not pair_data:
                return 0, [f"[FAIL] 无法获取 {pair_key} on {chain}"], 0

            # swap手续费+滑点
            swap_fee = current_usd * SWAP_FEE_PCT / 100
            gas_fee = GAS_COST.get(chain, 1.0)
            total_fees += swap_fee + gas_fee

            # 计算swap后的价值
            # DexScreener的priceUsd是base token的USD价格
            # 如果我们要 from_token→to_token:
            #   pair是 BASE/QUOTE，priceUsd是base的价格
            #   如果from=QUOTE(如USDT), to=BASE(如ETH): 我们用USDT买ETH
            #     ETH数量 = USDT金额 / ETH价格USD
            #     最终USD = ETH数量 * ETH价格USD（同链）= 输入金额（理论上1:1，但有手续费）
            #   如果from=BASE(如ETH), to=QUOTE(如USDT): 我们卖ETH换USDT
            #     同理

            # 扣掉swap手续费和gas
            after_fee_usd = current_usd - swap_fee - gas_fee

            # 实际上跨链套利的关键是: 同一个ETH在不同链的USD价格不同
            # 所以我们需要精确跟踪token数量，不能只用USD

            # 记录ETH价格以便跨链比较
            eth_price = pair_data["priceUsd"]

            if from_token in ("USDT", "USDC"):
                # 稳定币→ETH: 计算能买多少ETH
                eth_amount = after_fee_usd / eth_price if eth_price > 0 else 0
                current_usd = eth_amount * eth_price  # 此时以该链ETH价格计
                details.append(
                    f"  {chain}: {from_token} ${input_usd:.2f} → {eth_amount:.6f} ETH "
                    f"(价格${eth_price:.2f}, 手续费${swap_fee:.2f}+Gas${gas_fee:.2f})"
                )
                # 存ETH数量供后续使用
                step_data = {"eth_amount": eth_amount, "eth_price": eth_price}
            elif from_token == "BNB":
                # BNB→ETH: 用BNB/ETH的priceNative得到汇率
                # BNB/ETH pair: priceUsd是BNB的USD价格? 不对
                # DexScreener BNB/ETH pair: base=BNB, priceUsd=BNB的USD价格, priceNative=1BNB值多少ETH
                bnb_price = pair_data["priceUsd"]
                bnb_per_eth = pair_data["priceNative"]  # 1 BNB = X ETH
                bnb_amount = after_fee_usd / bnb_price if bnb_price > 0 else 0
                eth_amount = bnb_amount * bnb_per_eth
                current_usd = after_fee_usd  # 价值不变（同链）
                details.append(
                    f"  {chain}: {bnb_amount:.4f} BNB → {eth_amount:.6f} ETH "
                    f"(BNB${bnb_price:.2f}, 1BNB={bnb_per_eth:.6f}ETH, 手续费${swap_fee:.2f}+Gas${gas_fee:.2f})"
                )
                step_data = {"eth_amount": eth_amount}
            elif to_token in ("USDT", "USDC"):
                # ETH→稳定币: 卖ETH
                # 需要知道我们手上有多少ETH
                # current_usd代表当前持有资产的USD价值
                # 用目标链的ETH价格重新定价
                eth_amount_held = current_usd / eth_price if eth_price > 0 else 0
                sell_usd = eth_amount_held * eth_price
                current_usd = sell_usd - swap_fee - gas_fee
                details.append(
                    f"  {chain}: {eth_amount_held:.6f} ETH → ${current_usd:.2f} {to_token} "
                    f"(价格${eth_price:.2f}, 手续费${swap_fee:.2f}+Gas${gas_fee:.2f})"
                )
            elif from_token == "BNB" and to_token == "USDT":
                # BNB→USDT直接
                bnb_price = pair_data["priceUsd"]
                current_usd = after_fee_usd
                details.append(
                    f"  {chain}: BNB → ${current_usd:.2f} USDT (BNB${bnb_price:.2f})"
                )

            time.sleep(0.2)

        elif step[0] == "bridge":
            _, from_chain, to_chain, token = step
            bridge_fee = BRIDGE_COST.get((from_chain, to_chain), 3.0)
            bridge_slip = current_usd * BRIDGE_SLIPPAGE_PCT / 100
            total_fees += bridge_fee + bridge_slip
            current_usd -= bridge_fee + bridge_slip
            details.append(
                f"  Bridge {from_chain}→{to_chain}: 桥接费${bridge_fee:.2f} + 滑点${bridge_slip:.2f}"
            )

    return current_usd, details, total_fees


def evaluate_all_paths(input_usd: float) -> list[dict]:
    """评估所有套利路径"""
    results = []

    for path_def in ARB_PATHS:
        name = path_def["name"]
        print(f"\n--- {name} ---")

        # 模拟路径A（跨链路径）
        out_a, details_a, fees_a = simulate_path(path_def["path_a"], input_usd)
        if out_a == 0:
            for d in details_a:
                print(d)
            continue

        # 模拟路径B（基准：直接换或不动）
        if path_def["path_b"] == "direct":
            out_b = input_usd
            details_b = [f"  直接持有 ${input_usd:.2f} {path_def['input']}"]
            fees_b = 0
        else:
            out_b, details_b, fees_b = simulate_path(path_def["path_b"], input_usd)

        if out_b == 0:
            continue

        # 计算利润
        profit_usd = out_a - out_b
        profit_pct = (profit_usd / out_b * 100) if out_b > 0 else 0

        for d in details_a:
            print(d)
        print(f"  路径A产出: ${out_a:.2f} (费用${fees_a:.2f})")
        print(f"  基准产出:  ${out_b:.2f}")
        print(f"  净利润:    ${profit_usd:.2f} ({profit_pct:+.2f}%)")

        results.append({
            "name": name,
            "input_usd": input_usd,
            "output_a": out_a,
            "output_b": out_b,
            "profit_usd": profit_usd,
            "profit_pct": profit_pct,
            "fees": fees_a,
            "details": details_a,
        })

    return results


def check_alerts(results: list[dict]) -> list[dict]:
    """筛选出触发阈值的套利机会"""
    alerts = []
    now = time.time()

    for r in results:
        if r["profit_usd"] >= PROFIT_THRESHOLD_USD or r["profit_pct"] >= PROFIT_THRESHOLD_PCT:
            # 冷却检查
            last = _notified.get(r["name"], 0)
            if now - last < COOLDOWN_SECONDS:
                print(f"  [COOLDOWN] {r['name']}")
                continue
            _notified[r["name"]] = now
            alerts.append(r)

    return alerts


def format_email(alerts: list[dict], all_results: list[dict]) -> str:
    """格式化邮件内容"""
    lines = []
    lines.append(f"扫描时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"模拟投入: ${SIM_AMOUNT_USD}")
    lines.append(f"触发阈值: >${PROFIT_THRESHOLD_USD}美元 或 >{PROFIT_THRESHOLD_PCT}%")
    lines.append("")

    if alerts:
        lines.append(f"=== 发现 {len(alerts)} 个套利机会 ===\n")
        for i, a in enumerate(alerts, 1):
            lines.append(f"【机会{i}】{a['name']}")
            lines.append(f"  投入: ${a['input_usd']:.2f}")
            lines.append(f"  跨链产出: ${a['output_a']:.2f}")
            lines.append(f"  直接产出: ${a['output_b']:.2f}")
            lines.append(f"  净利润: ${a['profit_usd']:.2f} ({a['profit_pct']:+.2f}%)")
            lines.append(f"  总费用: ${a['fees']:.2f}")
            lines.append(f"  路径详情:")
            for d in a["details"]:
                lines.append(f"    {d.strip()}")
            lines.append("")

    # 附上全部路径的简要汇总
    lines.append("=== 全部路径汇总 ===\n")
    # 按利润排序
    sorted_results = sorted(all_results, key=lambda x: x["profit_pct"], reverse=True)
    for r in sorted_results:
        flag = " <<<" if r["profit_pct"] >= PROFIT_THRESHOLD_PCT else ""
        lines.append(f"  {r['profit_pct']:+.3f}% (${r['profit_usd']:+.2f}) | {r['name']}{flag}")

    lines.append("")
    lines.append("注意: 以上为估算值，实际桥接费用和滑点可能有差异。")
    lines.append("建议对持续出现的价差手动验证后再操作。")

    return "\n".join(lines)


def send_alert(alerts: list[dict], all_results: list[dict]):
    """发送邮件"""
    sys.path.insert(0, "/root/claude-chat")
    os.chdir("/root/claude-chat")
    from send_mail import send_email

    body = format_email(alerts, all_results)
    subject = f"跨链套利机会 ({len(alerts)}个, 最高{max(a['profit_pct'] for a in alerts):+.2f}%)"
    send_email(subject, body)
    print(f"\n邮件已发送! {len(alerts)} 个机会")


def run_once():
    """执行一次扫描"""
    global _price_cache
    _price_cache = {}  # 每次扫描清空缓存

    print(f"\n{'='*60}")
    print(f"跨链套利路径扫描 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模拟金额: ${SIM_AMOUNT_USD} | 利润阈值: ${PROFIT_THRESHOLD_USD} / {PROFIT_THRESHOLD_PCT}%")
    print(f"{'='*60}")

    results = evaluate_all_paths(SIM_AMOUNT_USD)

    alerts = check_alerts(results)
    if alerts:
        print(f"\n{'!'*40}")
        print(f"发现 {len(alerts)} 个套利机会! 发送邮件...")
        print(f"{'!'*40}")
        send_alert(alerts, results)
    else:
        print(f"\n未发现超过阈值的套利机会")
        # 打印最佳路径
        if results:
            best = max(results, key=lambda x: x["profit_pct"])
            print(f"当前最佳: {best['name']} ({best['profit_pct']:+.3f}%, ${best['profit_usd']:+.2f})")

    return results


def run_loop(interval: int = 60):
    """持续监控"""
    print(f"启动持续监控，每 {interval} 秒扫描一次")
    print(f"按 Ctrl+C 停止\n")

    while True:
        try:
            run_once()
            print(f"\n下次扫描: {interval}秒后...")
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\n监控已停止")
            break
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
            time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="跨链套利路径监控")
    parser.add_argument("--loop", action="store_true", help="持续监控模式")
    parser.add_argument("--interval", type=int, default=60, help="扫描间隔秒数(默认60)")
    parser.add_argument("--amount", type=float, default=500, help="模拟投入金额USD(默认500)")
    parser.add_argument("--threshold-usd", type=float, default=3.0, help="利润阈值USD(默认3)")
    parser.add_argument("--threshold-pct", type=float, default=0.3, help="利润率阈值%%(默认0.3)")
    args = parser.parse_args()

    SIM_AMOUNT_USD = args.amount
    PROFIT_THRESHOLD_USD = args.threshold_usd
    PROFIT_THRESHOLD_PCT = args.threshold_pct

    if args.loop:
        run_loop(args.interval)
    else:
        run_once()
