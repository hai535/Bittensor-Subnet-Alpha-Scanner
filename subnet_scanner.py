#!/usr/bin/env python3
"""
Bittensor Subnet Alpha Scanner + Whale Monitor
双模块：子网Alpha排放扫描 + 鲸鱼大户链上监测
"""

import asyncio
import aiohttp
import time
import math
import json
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
import threading

app = Flask(__name__)
CORS(app)

# === Config ===
TMC_BASE = "https://api.taomarketcap.com/internal/v1"
BLOCK_TIME = 12  # seconds per block
CACHE = {"scan": {"data": None, "ts": 0, "loading": False},
         "whale": {"data": None, "ts": 0, "loading": False},
         "news": {"data": None, "ts": 0, "loading": False}}
CACHE_TTL_SCAN = 120
CACHE_TTL_WHALE = 180  # 3 min for whale data
CACHE_TTL_NEWS = 300   # 5 min for news data

# === API Helpers ===
async def fetch_json(session, url, params=None):
    for attempt in range(3):
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                if resp.status == 200:
                    return await resp.json()
                elif resp.status == 429:
                    await asyncio.sleep(1 + attempt)
                else:
                    return None
        except Exception:
            await asyncio.sleep(0.5 * (attempt + 1))
    return None


# =====================================================
# MODULE 1: SUBNET ALPHA SCANNER (原有功能)
# =====================================================

async def fetch_all_subnets(session):
    all_subnets = []
    offset = 0
    limit = 50
    while True:
        data = await fetch_json(session, f"{TMC_BASE}/subnets/", {"limit": limit, "offset": offset})
        if not data or "results" not in data:
            break
        all_subnets.extend(data["results"])
        if not data.get("next"):
            break
        offset += limit
    return all_subnets


async def fetch_subnet_miners(session, netuid, subnetwork_n=0, max_validators=0):
    data = await fetch_json(session, f"{TMC_BASE}/subnets/weights/{netuid}/")
    if not data or "weights" not in data:
        total_miners = max(0, subnetwork_n - min(max_validators, subnetwork_n // 2))
        return {"total": total_miners, "active": total_miners, "validators": 0}

    weights_list = data.get("weights", [])
    validator_uids = set()
    active_miner_uids = set()

    for entry in weights_list:
        v_uid = entry.get("uid")
        if v_uid is not None:
            validator_uids.add(v_uid)
        values = entry.get("value", {})
        if isinstance(values, dict):
            for miner_uid, weight in values.items():
                if isinstance(weight, (int, float)) and weight > 0:
                    active_miner_uids.add(int(miner_uid))

    active_miner_uids -= validator_uids
    num_validators = len(validator_uids)
    total_miners = max(0, subnetwork_n - num_validators)
    active_miners = len(active_miner_uids)

    return {"total": total_miners, "active": active_miners, "validators": num_validators}


async def estimate_price_volatility(session, netuid, current_price, moving_price=None):
    if not current_price or float(current_price) <= 0:
        return None

    current_price = float(current_price)
    now = datetime.now(timezone.utc)
    ago_24h = now - timedelta(hours=24)

    data = await fetch_json(session, f"{TMC_BASE}/extrinsics/staking-activity/", {
        "subnet": netuid, "limit": 50
    })

    staking_volatility = None
    if data and "results" in data:
        prices_in_window = []
        for tx in data["results"]:
            ts = tx.get("timestamp", "")
            price = tx.get("to_alpha_price") or tx.get("from_alpha_price")
            if not price or float(price) <= 0:
                continue
            try:
                tx_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if tx_time >= ago_24h:
                    prices_in_window.append(float(price))
            except:
                continue

        if len(prices_in_window) >= 3:
            avg = sum(prices_in_window) / len(prices_in_window)
            if avg > 0:
                max_dev = max(abs(p - avg) for p in prices_in_window)
                staking_volatility = round((max_dev / avg) * 100, 2)

    ema_volatility = None
    if moving_price:
        mp = float(str(moving_price)) if not isinstance(moving_price, dict) else 0
        if mp > 0:
            ema_volatility = round(abs(current_price - mp) / mp * 100, 2)

    if staking_volatility is not None and staking_volatility < 500:
        return staking_volatility
    elif ema_volatility is not None:
        return ema_volatility
    elif staking_volatility is not None:
        return staking_volatility
    return None


def detect_registration_anomaly(snap):
    reg_this = snap.get("registrations_this_interval", 0) or 0
    target = snap.get("target_registrations_per_interval", 1) or 1
    max_per_block = snap.get("max_registrations_per_block", 1) or 1
    blocks_since = int(snap.get("blocks_since_last_step", "0") or "0")
    tempo = snap.get("tempo", 360) or 360
    burn = int(snap.get("burn", "0") or "0")
    max_burn = int(snap.get("max_burn", "100000000000") or "100000000000")
    subnetwork_n = snap.get("subnetwork_n", 0) or 0
    max_uids = snap.get("max_allowed_uids", 256) or 256

    anomaly_score = 0
    reasons = []

    fill_ratio = reg_this / target if target > 0 else 0
    if fill_ratio >= 1.0:
        anomaly_score += 30
        reasons.append(f"注册已满 {reg_this}/{target}")
    elif fill_ratio >= 0.8:
        anomaly_score += 15
        reasons.append(f"注册接近满 {reg_this}/{target}")

    progress = blocks_since / tempo if tempo > 0 else 0
    if fill_ratio >= 1.0 and progress < 0.3:
        anomaly_score += 40
        reasons.append(f"周期仅过{progress*100:.0f}%就满了")
    elif fill_ratio >= 1.0 and progress < 0.5:
        anomaly_score += 20
        reasons.append(f"周期过半前就满了")

    if burn > 0 and max_burn > 0:
        burn_ratio = burn / max_burn
        if burn_ratio > 0.1:
            anomaly_score += 20
            reasons.append(f"燃烧费高 {burn/1e9:.4f} TAO")
        elif burn_ratio > 0.01:
            anomaly_score += 10
            reasons.append(f"燃烧费中等 {burn/1e9:.4f} TAO")

    if subnetwork_n >= max_uids:
        anomaly_score += 10
        reasons.append(f"网络已满 {subnetwork_n}/{max_uids}")

    return {
        "score": min(anomaly_score, 100),
        "reasons": reasons,
        "reg_this": reg_this,
        "target": target,
        "fill_ratio": round(fill_ratio, 2),
        "blocks_since": blocks_since,
        "tempo": tempo,
        "progress_pct": round(progress * 100, 1)
    }


async def scan_subnets():
    results = []
    async with aiohttp.ClientSession() as session:
        subnets = await fetch_all_subnets(session)
        if not subnets:
            return {"error": "Failed to fetch subnets", "results": []}

        candidates = []
        for sn in subnets:
            snap = sn.get("latest_snapshot", {})
            if not snap:
                continue
            netuid = sn.get("netuid", 0)
            if netuid == 0:
                continue
            if not sn.get("is_active", False):
                continue

            alpha_in_em = int(snap.get("subnet_alpha_in_emission", "0") or "0")
            alpha_out_em = int(snap.get("subnet_alpha_out_emission", "0") or "0")
            if alpha_in_em == 0 and alpha_out_em == 0:
                continue

            tempo = snap.get("tempo", 360) or 360
            emission_interval_hours = (tempo * BLOCK_TIME) / 3600
            if emission_interval_hours > 12:
                continue

            candidates.append(sn)

        batch_size = 10
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i+batch_size]
            tasks = []

            for sn in batch:
                netuid = sn["netuid"]
                snap = sn["latest_snapshot"]
                price = snap.get("price", 0)

                async def process_subnet(sn=sn, netuid=netuid, snap=snap, price=price):
                    subnetwork_n = snap.get("subnetwork_n", 0) or 0
                    max_vals = snap.get("max_allowed_validators", 0) or 0
                    miners = await fetch_subnet_miners(session, netuid, subnetwork_n, max_vals)
                    moving_price = snap.get("subnet_moving_price")
                    if isinstance(moving_price, dict):
                        moving_price = None
                    volatility = await estimate_price_volatility(session, netuid, price, moving_price)
                    return {"netuid": netuid, "miners": miners, "volatility": volatility}

                tasks.append(process_subnet())

            batch_results = await asyncio.gather(*tasks)

            for sn, detail in zip(batch, batch_results):
                snap = sn["latest_snapshot"]
                netuid = sn["netuid"]
                tempo = snap.get("tempo", 360)
                emission_hours = round((tempo * BLOCK_TIME) / 3600, 2)
                alpha_in_em = int(snap.get("subnet_alpha_in_emission", "0") or "0")
                alpha_out_em = int(snap.get("subnet_alpha_out_emission", "0") or "0")
                total_alpha_em = alpha_in_em + alpha_out_em
                price = float(snap.get("price", 0) or 0)
                tao_in_em = int(snap.get("subnet_tao_in_emission", "0") or "0")
                reg_anomaly = detect_registration_anomaly(snap)

                identity = snap.get("subnet_identities_v3") or {}
                name = identity.get("subnetName", f"SN{netuid}")
                github = identity.get("githubRepo", "")
                desc = identity.get("description", "")
                symbol = snap.get("token_symbol", "α")

                dtao = snap.get("dtao", {}) or {}
                market_cap = dtao.get("marketCap", 0)
                fdv = dtao.get("fdv", 0)
                tao_liquidity = dtao.get("taoLiquidity", 0)

                active_miners = detail["miners"]["active"]
                total_miners = detail["miners"]["total"]
                volatility = detail["volatility"]

                pending_server = int(snap.get("pending_server_emission", "0") or "0")
                pending_validator = int(snap.get("pending_validator_emission", "0") or "0")
                pending_total = pending_server + pending_validator
                miner_emission_pct = (pending_server / pending_total * 100) if pending_total > 0 else 0

                result = {
                    "netuid": netuid, "name": name, "symbol": symbol,
                    "description": desc, "github": github,
                    "tempo": tempo, "emission_interval_hours": emission_hours,
                    "alpha_emission_per_tempo": total_alpha_em,
                    "alpha_emission_display": f"{total_alpha_em / 1e9:.4f}",
                    "tao_emission_per_tempo": tao_in_em,
                    "tao_emission_display": f"{tao_in_em / 1e9:.6f}",
                    "price_tao": price,
                    "price_display": f"{price:.6f}" if price < 1 else f"{price:.4f}",
                    "volatility_24h": volatility,
                    "active_miners": active_miners, "total_miners": total_miners,
                    "miner_emission_pct": round(miner_emission_pct, 1),
                    "pending_miner_emission": pending_server,
                    "registration": reg_anomaly,
                    "market_cap_tao": round(market_cap, 2) if market_cap else 0,
                    "fdv_tao": round(fdv, 2) if fdv else 0,
                    "tao_liquidity": round(tao_liquidity / 1e9, 2) if tao_liquidity else 0,
                    "subnetwork_n": snap.get("subnetwork_n", 0),
                    "max_uids": snap.get("max_allowed_uids", 256),
                    "burn_cost": f"{int(snap.get('burn', '0') or '0') / 1e9:.6f}",
                    "pass_miners": active_miners > 10,
                    "pass_volatility": volatility is not None and volatility <= 5.0,
                    "pass_reg_anomaly": reg_anomaly["score"] >= 30,
                    "pass_all": False
                }
                result["pass_all"] = result["pass_miners"] and result["pass_volatility"] and result["pass_reg_anomaly"]
                results.append(result)

            if i + batch_size < len(candidates):
                await asyncio.sleep(0.3)

    results.sort(key=lambda x: (-int(x["pass_all"]), -x["registration"]["score"], -x["active_miners"]))
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_subnets": len(subnets),
        "candidates_with_emission": len(candidates),
        "results": results,
        "filters_summary": {
            "pass_all": sum(1 for r in results if r["pass_all"]),
            "pass_miners_gt10": sum(1 for r in results if r["pass_miners"]),
            "pass_volatility_lt5": sum(1 for r in results if r["pass_volatility"]),
            "pass_reg_anomaly": sum(1 for r in results if r["pass_reg_anomaly"]),
        }
    }


# =====================================================
# MODULE 2: WHALE MONITOR (鲸鱼监测)
# =====================================================

async def fetch_staking_activity_pages(session, pages=20, subnet=None, func_filter=None):
    """Fetch multiple pages of staking activity for whale analysis"""
    all_txs = []
    for page in range(pages):
        offset = page * 100
        params = {"limit": 100, "offset": offset}
        if subnet:
            params["subnet"] = subnet
        data = await fetch_json(session, f"{TMC_BASE}/extrinsics/staking-activity/", params)
        if not data or "results" not in data:
            break
        results = data["results"]
        if not results:
            break
        for tx in results:
            if func_filter and tx.get("function") not in func_filter:
                continue
            all_txs.append(tx)
        if not data.get("next"):
            break
        await asyncio.sleep(0.15)
    return all_txs


async def fetch_wallet_staking_history(session, coldkey, days=7):
    """Fetch a specific wallet's staking history"""
    all_txs = []
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)

    for page in range(30):  # max 30 pages = 3000 txs
        offset = page * 100
        data = await fetch_json(session, f"{TMC_BASE}/extrinsics/staking-activity/", {
            "limit": 100, "offset": offset, "coldkey": coldkey
        })
        if not data or "results" not in data:
            break
        results = data["results"]
        if not results:
            break

        stop = False
        for tx in results:
            ts = tx.get("timestamp", "")
            try:
                tx_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if tx_time < cutoff:
                    stop = True
                    break
            except:
                pass
            all_txs.append(tx)

        if stop or not data.get("next"):
            break
        await asyncio.sleep(0.1)

    return all_txs


def analyze_whale_behavior(txs, coldkey, now):
    """Analyze a whale's buying behavior across time windows"""
    # Group by time window
    windows = {"1d": timedelta(days=1), "3d": timedelta(days=3), "7d": timedelta(days=7)}
    result = {}

    for window_name, delta in windows.items():
        cutoff = now - delta
        window_txs = []
        for tx in txs:
            ts = tx.get("timestamp", "")
            try:
                tx_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if tx_time >= cutoff:
                    window_txs.append(tx)
            except:
                continue

        # Aggregate by subnet (netuid)
        subnet_buys = defaultdict(lambda: {"count": 0, "total_tao": 0, "total_alpha": 0,
                                            "txs": [], "prices": [], "functions": []})
        total_tao_in = 0
        total_tao_out = 0

        for tx in window_txs:
            func = tx.get("function", "")
            to_netuid = tx.get("to_netuid")
            from_netuid = tx.get("from_netuid")
            to_tao = int(tx.get("to_tao_amount") or 0)
            from_tao = int(tx.get("from_tao_amount") or 0)
            to_alpha = int(tx.get("to_alpha_amount") or 0)
            from_alpha = int(tx.get("from_alpha_amount") or 0)

            if "add_stake" in func or func == "move_stake":
                # Buying into a subnet
                netuid = to_netuid
                if netuid is not None:
                    tao_amount = from_tao if from_tao > 0 else to_tao
                    subnet_buys[netuid]["count"] += 1
                    subnet_buys[netuid]["total_tao"] += tao_amount
                    subnet_buys[netuid]["total_alpha"] += to_alpha
                    subnet_buys[netuid]["functions"].append(func)
                    subnet_buys[netuid]["txs"].append({
                        "time": tx.get("timestamp"),
                        "func": func,
                        "tao": tao_amount / 1e9,
                        "alpha": to_alpha / 1e9,
                        "price": tx.get("to_alpha_price")
                    })
                    if tx.get("to_alpha_price"):
                        subnet_buys[netuid]["prices"].append(float(tx["to_alpha_price"]))
                    total_tao_in += tao_amount

            elif "remove_stake" in func:
                netuid = from_netuid
                if netuid is not None:
                    tao_amount = to_tao if to_tao > 0 else from_tao
                    total_tao_out += tao_amount

        result[window_name] = {
            "tx_count": len(window_txs),
            "total_tao_in": round(total_tao_in / 1e9, 4),
            "total_tao_out": round(total_tao_out / 1e9, 4),
            "net_flow": round((total_tao_in - total_tao_out) / 1e9, 4),
            "subnets": {}
        }

        for netuid, info in subnet_buys.items():
            avg_price = sum(info["prices"]) / len(info["prices"]) if info["prices"] else 0
            result[window_name]["subnets"][str(netuid)] = {
                "buy_count": info["count"],
                "total_tao": round(info["total_tao"] / 1e9, 4),
                "total_alpha": round(info["total_alpha"] / 1e9, 4),
                "avg_price": round(avg_price, 8),
                "recent_txs": info["txs"][-10:]  # last 10 txs
            }

    return result


def detect_anomalies(whale_data, all_whale_addresses):
    """Detect abnormal patterns in whale behavior"""
    alerts = []

    for window in ["1d", "3d", "7d"]:
        w = whale_data.get(window, {})
        if not w:
            continue

        # Alert 1: Large continuous buying (>500 TAO net inflow)
        if w["net_flow"] > 500:
            alerts.append({
                "type": "heavy_buy",
                "severity": "high" if w["net_flow"] > 2000 else "medium",
                "window": window,
                "msg": f"{window}内净流入 {w['net_flow']:.1f} TAO",
                "detail": f"买入 {w['total_tao_in']:.1f} τ, 卖出 {w['total_tao_out']:.1f} τ"
            })

        # Alert 2: Concentrated buying into single subnet
        for sn_id, sn_data in w.get("subnets", {}).items():
            if sn_data["total_tao"] > 200:
                alerts.append({
                    "type": "concentrated_buy",
                    "severity": "high" if sn_data["total_tao"] > 1000 else "medium",
                    "window": window,
                    "netuid": int(sn_id),
                    "msg": f"{window}内集中买入 SN{sn_id}: {sn_data['total_tao']:.1f} TAO ({sn_data['buy_count']}笔)",
                    "detail": f"均价 {sn_data['avg_price']:.6f}, 获得 {sn_data['total_alpha']:.2f} Alpha"
                })

            # Alert 3: Frequent small buys (DCA pattern / stealth accumulation)
            if sn_data["buy_count"] >= 5 and sn_data["total_tao"] > 50:
                avg_per_tx = sn_data["total_tao"] / sn_data["buy_count"]
                if avg_per_tx < 200:  # Small frequent buys
                    alerts.append({
                        "type": "stealth_accumulate",
                        "severity": "medium",
                        "window": window,
                        "netuid": int(sn_id),
                        "msg": f"{window}内分批买入 SN{sn_id}: {sn_data['buy_count']}笔, 均{avg_per_tx:.1f} TAO/笔",
                        "detail": "疑似隐蔽建仓，分散买入降低影响"
                    })

        # Alert 4: High frequency trading
        if w["tx_count"] > 20 and window == "1d":
            alerts.append({
                "type": "high_frequency",
                "severity": "medium",
                "window": window,
                "msg": f"24h内 {w['tx_count']} 笔交易，交易频繁",
                "detail": "高频操作，可能在进行套利或大规模建仓"
            })

    return alerts


def detect_split_wallet_patterns(whale_data_map, subnet_list):
    """Detect multiple wallets buying the same subnet at similar times"""
    alerts = []

    # Group all whale buys by subnet
    subnet_buyers = defaultdict(list)  # netuid -> [(coldkey, tao_amount, tx_count, first_tx_time)]

    for coldkey, data in whale_data_map.items():
        for window in ["1d", "3d"]:
            w = data.get("behavior", {}).get(window, {})
            for sn_id, sn_info in w.get("subnets", {}).items():
                if sn_info["total_tao"] > 50:
                    first_tx_time = None
                    if sn_info.get("recent_txs"):
                        first_tx_time = sn_info["recent_txs"][0].get("time")
                    subnet_buyers[sn_id].append({
                        "coldkey": coldkey,
                        "tao": sn_info["total_tao"],
                        "count": sn_info["buy_count"],
                        "time": first_tx_time,
                        "window": window
                    })

    # Detect: 3+ different wallets buying same subnet
    for sn_id, buyers in subnet_buyers.items():
        unique_wallets = set(b["coldkey"] for b in buyers)
        if len(unique_wallets) >= 3:
            total_tao = sum(b["tao"] for b in buyers)
            sn_name = None
            for sn in subnet_list:
                if str(sn.get("netuid")) == sn_id:
                    snap = sn.get("latest_snapshot", {})
                    identity = snap.get("subnet_identities_v3") or {}
                    sn_name = identity.get("subnetName", f"SN{sn_id}")
                    break

            alerts.append({
                "type": "split_wallet",
                "severity": "high",
                "netuid": int(sn_id),
                "subnet_name": sn_name or f"SN{sn_id}",
                "msg": f"SN{sn_id} ({sn_name}): {len(unique_wallets)} 个大户钱包同时买入, 共 {total_tao:.0f} TAO",
                "wallets": [{"addr": b["coldkey"][:8]+"..."+b["coldkey"][-6:],
                             "full": b["coldkey"],
                             "tao": b["tao"],
                             "txs": b["count"]}
                            for b in sorted(buyers, key=lambda x: -x["tao"])]
            })

    return alerts


async def scan_whales():
    """Main whale scanning function"""
    now = datetime.now(timezone.utc)

    async with aiohttp.ClientSession() as session:
        # Step 1: Fetch recent staking activity (large transactions)
        # We fetch many pages to find big players
        all_txs = await fetch_staking_activity_pages(session, pages=30,
                                                      func_filter={"add_stake", "add_stake_limit",
                                                                    "remove_stake", "remove_stake_limit",
                                                                    "move_stake"})

        if not all_txs:
            return {"error": "Failed to fetch staking activity", "whales": [], "alerts": []}

        # Step 2: Group by coldkey, calculate total volume
        coldkey_stats = defaultdict(lambda: {"total_tao_volume": 0, "tx_count": 0,
                                              "buy_tao": 0, "sell_tao": 0,
                                              "subnets_touched": set(), "txs": [],
                                              "first_seen": None, "last_seen": None})

        for tx in all_txs:
            ck = tx.get("signer_coldkey") or tx.get("from_coldkey")
            if not ck:
                continue

            func = tx.get("function", "")
            from_tao = int(tx.get("from_tao_amount") or 0)
            to_tao = int(tx.get("to_tao_amount") or 0)
            tao_amount = max(from_tao, to_tao)

            stats = coldkey_stats[ck]
            stats["total_tao_volume"] += tao_amount
            stats["tx_count"] += 1

            if "add_stake" in func or func == "move_stake":
                stats["buy_tao"] += tao_amount
                if tx.get("to_netuid") is not None:
                    stats["subnets_touched"].add(tx["to_netuid"])
            elif "remove_stake" in func:
                stats["sell_tao"] += tao_amount
                if tx.get("from_netuid") is not None:
                    stats["subnets_touched"].add(tx["from_netuid"])

            ts = tx.get("timestamp")
            if ts:
                if stats["first_seen"] is None or ts < stats["first_seen"]:
                    stats["first_seen"] = ts
                if stats["last_seen"] is None or ts > stats["last_seen"]:
                    stats["last_seen"] = ts

            stats["txs"].append(tx)

        # Step 3: Filter for whales (high volume traders)
        # Threshold: >1000 TAO volume or >10k TAO balance equivalent
        whale_threshold = 500 * 1e9  # 500 TAO minimum volume
        whale_keys = [ck for ck, stats in coldkey_stats.items()
                      if stats["total_tao_volume"] >= whale_threshold]

        # Sort by volume
        whale_keys.sort(key=lambda ck: -coldkey_stats[ck]["total_tao_volume"])
        whale_keys = whale_keys[:50]  # Top 50 whales

        # Step 4: For each whale, analyze behavior
        # Fetch subnets for name mapping
        subnets = await fetch_all_subnets(session)
        subnet_map = {}
        for sn in subnets:
            snap = sn.get("latest_snapshot", {})
            identity = snap.get("subnet_identities_v3") or {}
            subnet_map[sn["netuid"]] = {
                "name": identity.get("subnetName", f"SN{sn['netuid']}"),
                "symbol": snap.get("token_symbol", "α"),
                "price": float(snap.get("price", 0) or 0)
            }

        whales = []
        whale_data_map = {}

        for ck in whale_keys:
            stats = coldkey_stats[ck]
            txs = stats["txs"]

            # Analyze behavior across time windows
            behavior = analyze_whale_behavior(txs, ck, now)

            # Detect individual anomalies
            anomalies = detect_anomalies(behavior, whale_keys)

            # Subnet breakdown
            subnet_breakdown = []
            for window in ["7d"]:
                for sn_id, sn_data in behavior.get(window, {}).get("subnets", {}).items():
                    sn_info = subnet_map.get(int(sn_id), {})
                    subnet_breakdown.append({
                        "netuid": int(sn_id),
                        "name": sn_info.get("name", f"SN{sn_id}"),
                        "symbol": sn_info.get("symbol", "α"),
                        "current_price": sn_info.get("price", 0),
                        "total_tao": sn_data["total_tao"],
                        "total_alpha": sn_data["total_alpha"],
                        "buy_count": sn_data["buy_count"],
                        "avg_price": sn_data["avg_price"]
                    })
            subnet_breakdown.sort(key=lambda x: -x["total_tao"])

            whale = {
                "address": ck,
                "address_short": ck[:8] + "..." + ck[-6:],
                "total_volume_tao": round(stats["total_tao_volume"] / 1e9, 2),
                "buy_tao": round(stats["buy_tao"] / 1e9, 2),
                "sell_tao": round(stats["sell_tao"] / 1e9, 2),
                "net_flow": round((stats["buy_tao"] - stats["sell_tao"]) / 1e9, 2),
                "tx_count": stats["tx_count"],
                "subnets_count": len(stats["subnets_touched"]),
                "first_seen": stats["first_seen"],
                "last_seen": stats["last_seen"],
                "behavior": behavior,
                "anomalies": anomalies,
                "anomaly_count": len(anomalies),
                "high_severity_count": sum(1 for a in anomalies if a["severity"] == "high"),
                "subnet_breakdown": subnet_breakdown
            }
            whales.append(whale)
            whale_data_map[ck] = whale

        # Step 5: Detect split-wallet patterns
        split_alerts = detect_split_wallet_patterns(whale_data_map, subnets)

        # Sort whales: most anomalies first, then by volume
        whales.sort(key=lambda w: (-w["high_severity_count"], -w["anomaly_count"], -w["total_volume_tao"]))

        # Build global alerts
        all_alerts = []
        for w in whales:
            for a in w["anomalies"]:
                a["whale"] = w["address_short"]
                a["whale_full"] = w["address"]
                all_alerts.append(a)

        all_alerts.extend(split_alerts)
        all_alerts.sort(key=lambda a: -(2 if a["severity"]=="high" else 1))

        # Hot subnets: which subnets are whales buying most
        hot_subnets = defaultdict(lambda: {"total_tao": 0, "whale_count": 0, "whales": []})
        for w in whales:
            for sn in w["subnet_breakdown"]:
                key = sn["netuid"]
                hot_subnets[key]["total_tao"] += sn["total_tao"]
                hot_subnets[key]["whale_count"] += 1
                hot_subnets[key]["name"] = sn["name"]
                hot_subnets[key]["symbol"] = sn["symbol"]
                hot_subnets[key]["price"] = sn["current_price"]
                hot_subnets[key]["whales"].append({
                    "addr": w["address_short"],
                    "full": w["address"],
                    "tao": sn["total_tao"]
                })

        hot_list = [{"netuid": k, **v} for k, v in hot_subnets.items()]
        hot_list.sort(key=lambda x: -x["total_tao"])

        return {
            "timestamp": now.isoformat(),
            "total_txs_analyzed": len(all_txs),
            "whale_count": len(whales),
            "alert_count": len(all_alerts),
            "whales": whales,
            "alerts": all_alerts[:100],
            "split_wallet_alerts": split_alerts,
            "hot_subnets": hot_list[:20]
        }


# =====================================================
# MODULE 3: NEWS AGGREGATOR (最新消息)
# =====================================================

async def fetch_crypto_news(session):
    """Fetch Bittensor news from cryptocurrency.cv"""
    items = []
    for query in ["bittensor", "TAO crypto", "bittensor subnet"]:
        data = await fetch_json(session, f"https://cryptocurrency.cv/api/search",
                                {"q": query, "limit": 30})
        if not data:
            continue
        # API returns {"articles": [...]} or a plain list
        article_list = data.get("articles", data) if isinstance(data, dict) else data
        if not isinstance(article_list, list):
            continue
        for item in article_list:
            items.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "summary": item.get("description", ""),
                "image": item.get("imageUrl", ""),
                "source": item.get("source", "Unknown"),
                "source_key": item.get("sourceKey", ""),
                "category": item.get("category", "general"),
                "time_ago": item.get("timeAgo", ""),
                "pub_date": item.get("pubDate", ""),
                "type": "news"
            })
        await asyncio.sleep(0.3)
    return items


async def fetch_reddit_posts(session):
    """Fetch r/bittensor posts via RSS (JSON blocked by Reddit on server IPs)"""
    items = []
    try:
        import xml.etree.ElementTree as ET
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        async with session.get(
            "https://www.reddit.com/r/bittensor_/new.rss?limit=50",
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=15)
        ) as resp:
            if resp.status == 200:
                text = await resp.text()
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                root = ET.fromstring(text)
                for entry in root.findall("atom:entry", ns):
                    title = entry.findtext("atom:title", "", ns)
                    link_el = entry.find("atom:link", ns)
                    link = link_el.get("href", "") if link_el is not None else ""
                    updated = entry.findtext("atom:updated", "", ns)
                    author_el = entry.find("atom:author", ns)
                    author = author_el.findtext("atom:name", "", ns) if author_el is not None else ""
                    content = entry.findtext("atom:content", "", ns) or ""
                    # Clean HTML from content
                    import re
                    clean_text = re.sub(r'<[^>]+>', ' ', content).strip()[:300]

                    # Parse date
                    pub_date = ""
                    if updated:
                        try:
                            pub_date = updated.replace("Z", "+00:00")
                        except Exception:
                            pub_date = updated

                    items.append({
                        "title": title,
                        "url": link,
                        "summary": clean_text,
                        "source": "Reddit r/bittensor",
                        "author": author.replace("/u/", ""),
                        "score": 0,
                        "comments": 0,
                        "pub_date": pub_date,
                        "flair": "",
                        "type": "reddit"
                    })
    except Exception:
        pass
    return items


async def fetch_tao_app_social(session):
    """Fetch social/community data from tao.app API (public endpoints)"""
    items = []
    try:
        # Try fetching subnet events/social data from taomarketcap
        data = await fetch_json(session, f"{TMC_BASE}/subnets/", {"limit": 20, "offset": 0})
        if data and "results" in data:
            for sn in data["results"]:
                netuid = sn.get("netuid", 0)
                name = sn.get("name", "")
                # Check for notable changes - high registration, price moves
                reg_this = sn.get("registrations_this_interval", 0)
                target = sn.get("target_regs_per_interval", 1)
                if reg_this > 0 and target > 0 and (reg_this / target) > 0.3:
                    items.append({
                        "title": f"🔥 SN{netuid} {name} 注册活跃: {reg_this}/{target} ({reg_this/target*100:.0f}%)",
                        "url": f"https://www.tao.app/subnets/{netuid}",
                        "summary": f"子网 SN{netuid} 当前注册周期已完成 {reg_this/target*100:.0f}%，注册需求旺盛",
                        "source": "Bittensor Chain",
                        "pub_date": datetime.now(timezone.utc).isoformat(),
                        "type": "chain_event",
                        "netuid": netuid,
                        "severity": "high" if reg_this >= target else "medium"
                    })
    except Exception:
        pass
    return items


async def fetch_coingecko_tao(session):
    """Fetch TAO market data from CoinGecko for context"""
    try:
        data = await fetch_json(session, "https://api.coingecko.com/api/v3/coins/bittensor",
                                {"localization": "false", "tickers": "false",
                                 "market_data": "true", "community_data": "true",
                                 "developer_data": "false", "sparkline": "false"})
        if data:
            md = data.get("market_data", {})
            cd = data.get("community_data", {})
            price_usd = md.get("current_price", {}).get("usd", 0)
            change_24h = md.get("price_change_percentage_24h", 0)
            change_7d = md.get("price_change_percentage_7d", 0)
            market_cap = md.get("market_cap", {}).get("usd", 0)
            vol_24h = md.get("total_volume", {}).get("usd", 0)

            return {
                "price_usd": price_usd,
                "change_24h": round(change_24h, 2) if change_24h else 0,
                "change_7d": round(change_7d, 2) if change_7d else 0,
                "market_cap": market_cap,
                "volume_24h": vol_24h,
                "reddit_subscribers": cd.get("reddit_subscribers", 0),
                "telegram_members": cd.get("telegram_channel_user_count", 0),
                "twitter_followers": data.get("community_data", {}).get("twitter_followers", 0)
            }
    except Exception:
        pass
    return None


async def scan_news():
    """Main news aggregation function"""
    async with aiohttp.ClientSession() as session:
        # Fetch all sources concurrently
        news_task = fetch_crypto_news(session)
        reddit_task = fetch_reddit_posts(session)
        chain_task = fetch_tao_app_social(session)
        gecko_task = fetch_coingecko_tao(session)

        news_items, reddit_items, chain_items, tao_market = await asyncio.gather(
            news_task, reddit_task, chain_task, gecko_task,
            return_exceptions=True
        )

        # Handle exceptions
        if isinstance(news_items, Exception): news_items = []
        if isinstance(reddit_items, Exception): reddit_items = []
        if isinstance(chain_items, Exception): chain_items = []
        if isinstance(tao_market, Exception): tao_market = None

        all_items = []

        # Deduplicate news by title similarity
        seen_titles = set()
        for item in (news_items or []):
            title_key = item.get("title", "").lower().strip()[:60]
            if title_key and title_key not in seen_titles:
                seen_titles.add(title_key)
                all_items.append(item)

        for item in (reddit_items or []):
            all_items.append(item)

        for item in (chain_items or []):
            all_items.append(item)

        # Parse and sort by time
        now = datetime.now(timezone.utc)
        for item in all_items:
            pub = item.get("pub_date", "")
            if pub:
                try:
                    if isinstance(pub, str):
                        # Try ISO format
                        dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
                    else:
                        dt = now
                    item["timestamp"] = dt.timestamp()
                    item["age_hours"] = (now - dt).total_seconds() / 3600
                except Exception:
                    item["timestamp"] = 0
                    item["age_hours"] = 999
            else:
                item["timestamp"] = 0
                item["age_hours"] = 999

        # Sort by timestamp descending (newest first)
        all_items.sort(key=lambda x: x.get("timestamp", 0), reverse=True)

        # Categorize by time window
        hour_items = [i for i in all_items if i.get("age_hours", 999) <= 2]
        day_items = [i for i in all_items if i.get("age_hours", 999) <= 24]
        week_items = [i for i in all_items if i.get("age_hours", 999) <= 168]

        # Identify "important" items (high engagement reddit posts, major news)
        important = []
        for item in all_items:
            if item.get("type") == "reddit" and (item.get("score", 0) > 50 or item.get("comments", 0) > 20):
                important.append(item)
            elif item.get("type") == "chain_event" and item.get("severity") == "high":
                important.append(item)

        return {
            "all_items": all_items[:200],
            "hour": hour_items[:50],
            "day": day_items[:100],
            "week": week_items[:200],
            "important": important[:20],
            "tao_market": tao_market,
            "stats": {
                "total": len(all_items),
                "hour_count": len(hour_items),
                "day_count": len(day_items),
                "week_count": len(week_items),
                "news_count": len([i for i in all_items if i["type"] == "news"]),
                "reddit_count": len([i for i in all_items if i["type"] == "reddit"]),
                "chain_count": len([i for i in all_items if i["type"] == "chain_event"]),
                "important_count": len(important)
            },
            "sources": ["cryptocurrency.cv", "Reddit r/bittensor", "Bittensor Chain", "CoinGecko"],
            "timestamp": now.isoformat()
        }


# =====================================================
# ROUTES
# =====================================================

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/scan")
def api_scan():
    now = time.time()
    cache = CACHE["scan"]
    if cache["data"] and (now - cache["ts"]) < CACHE_TTL_SCAN:
        return jsonify(cache["data"])
    if cache["loading"]:
        if cache["data"]:
            return jsonify(cache["data"])
        return jsonify({"error": "Scanning in progress...", "results": []})

    cache["loading"] = True
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        data = loop.run_until_complete(scan_subnets())
        loop.close()
        cache["data"] = data
        cache["ts"] = time.time()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e), "results": []}), 500
    finally:
        cache["loading"] = False


@app.route("/api/whales")
def api_whales():
    now = time.time()
    cache = CACHE["whale"]
    if cache["data"] and (now - cache["ts"]) < CACHE_TTL_WHALE:
        return jsonify(cache["data"])
    if cache["loading"]:
        if cache["data"]:
            return jsonify(cache["data"])
        return jsonify({"error": "Scanning whales...", "whales": [], "alerts": []})

    cache["loading"] = True
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        data = loop.run_until_complete(scan_whales())
        loop.close()
        cache["data"] = data
        cache["ts"] = time.time()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e), "whales": [], "alerts": []}), 500
    finally:
        cache["loading"] = False


@app.route("/api/news")
def api_news():
    now = time.time()
    cache = CACHE["news"]
    if cache["data"] and (now - cache["ts"]) < CACHE_TTL_NEWS:
        return jsonify(cache["data"])
    if cache["loading"]:
        if cache["data"]:
            return jsonify(cache["data"])
        return jsonify({"error": "Fetching news...", "all_items": []})

    cache["loading"] = True
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        data = loop.run_until_complete(scan_news())
        loop.close()
        cache["data"] = data
        cache["ts"] = time.time()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e), "all_items": []}), 500
    finally:
        cache["loading"] = False


@app.route("/api/whale/<address>")
def api_whale_detail(address):
    """Get detailed activity for a specific whale"""
    days = int(request.args.get("days", 7))
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def get_detail():
            async with aiohttp.ClientSession() as session:
                txs = await fetch_wallet_staking_history(session, address, days=days)
                now = datetime.now(timezone.utc)
                behavior = analyze_whale_behavior(txs, address, now)
                anomalies = detect_anomalies(behavior, [])

                # Build tx list
                tx_list = []
                for tx in txs:
                    func = tx.get("function", "")
                    tx_list.append({
                        "time": tx.get("timestamp"),
                        "function": func,
                        "from_netuid": tx.get("from_netuid"),
                        "to_netuid": tx.get("to_netuid"),
                        "from_tao": round(int(tx.get("from_tao_amount") or 0) / 1e9, 4),
                        "to_tao": round(int(tx.get("to_tao_amount") or 0) / 1e9, 4),
                        "from_alpha": round(int(tx.get("from_alpha_amount") or 0) / 1e9, 4),
                        "to_alpha": round(int(tx.get("to_alpha_amount") or 0) / 1e9, 4),
                        "price": tx.get("to_alpha_price") or tx.get("from_alpha_price"),
                        "block": tx.get("block_number"),
                        "extrinsic": tx.get("extrinsic"),
                        "success": tx.get("success", True)
                    })

                return {
                    "address": address,
                    "days": days,
                    "total_txs": len(txs),
                    "behavior": behavior,
                    "anomalies": anomalies,
                    "transactions": tx_list[:200]
                }

        data = loop.run_until_complete(get_detail())
        loop.close()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =====================================================
# HTML TEMPLATE (双模块 + 左侧菜单)
# =====================================================

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Bittensor Intelligence Hub</title>
<style>
  :root {
    --bg: #0a0e17;
    --bg2: #060a12;
    --card: #111827;
    --card2: #0f172a;
    --border: #1e293b;
    --text: #e2e8f0;
    --text2: #94a3b8;
    --text3: #64748b;
    --accent: #06b6d4;
    --accent2: #8b5cf6;
    --green: #10b981;
    --red: #ef4444;
    --orange: #f59e0b;
    --blue: #3b82f6;
    --pink: #ec4899;
    --yellow: #eab308;
    --sidebar-w: 240px;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'SF Mono','Fira Code','JetBrains Mono','Cascadia Code',monospace;
    min-height: 100vh;
    display: flex;
  }

  /* === SIDEBAR === */
  .sidebar {
    width: var(--sidebar-w);
    min-height: 100vh;
    background: var(--bg2);
    border-right: 1px solid var(--border);
    position: fixed;
    left: 0;
    top: 0;
    bottom: 0;
    z-index: 100;
    display: flex;
    flex-direction: column;
    transition: transform 0.3s;
  }
  .sidebar-header {
    padding: 20px 16px;
    border-bottom: 1px solid var(--border);
  }
  .sidebar-header h1 {
    font-size: 16px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
  }
  .sidebar-header p {
    font-size: 10px;
    color: var(--text3);
  }
  .sidebar-nav {
    flex: 1;
    padding: 12px 0;
  }
  .nav-section {
    padding: 0 12px;
    margin-bottom: 8px;
  }
  .nav-section-title {
    font-size: 10px;
    color: var(--text3);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    padding: 8px 8px 6px;
  }
  .nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
    border-radius: 8px;
    cursor: pointer;
    color: var(--text2);
    font-size: 13px;
    transition: all 0.15s;
    margin-bottom: 2px;
  }
  .nav-item:hover {
    background: rgba(6,182,212,0.08);
    color: var(--text);
  }
  .nav-item.active {
    background: rgba(6,182,212,0.12);
    color: var(--accent);
    font-weight: 600;
  }
  .nav-item .icon { font-size: 18px; width: 24px; text-align: center; }
  .nav-item .badge {
    margin-left: auto;
    background: var(--red);
    color: #fff;
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 10px;
    font-weight: 700;
  }
  .sidebar-footer {
    padding: 16px;
    border-top: 1px solid var(--border);
    font-size: 10px;
    color: var(--text3);
    text-align: center;
  }

  /* === MAIN CONTENT === */
  .main {
    margin-left: var(--sidebar-w);
    flex: 1;
    min-height: 100vh;
  }
  .page { display: none; padding: 24px; max-width: 1600px; margin: 0 auto; }
  .page.active { display: block; }

  /* === SHARED COMPONENTS === */
  .page-header {
    margin-bottom: 24px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border);
  }
  .page-header h2 {
    font-size: 22px;
    margin-bottom: 6px;
  }
  .page-header p { color: var(--text2); font-size: 12px; }

  .stats-bar {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
    margin-bottom: 24px;
  }
  .stat-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px;
    text-align: center;
  }
  .stat-card .val {
    font-size: 26px;
    font-weight: bold;
    color: var(--accent);
  }
  .stat-card .label {
    font-size: 10px;
    color: var(--text2);
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  .stat-card.hot .val { color: var(--red); }
  .stat-card.warn .val { color: var(--orange); }
  .stat-card.good .val { color: var(--green); }

  .controls {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 20px;
    align-items: center;
  }
  .controls label { font-size: 11px; color: var(--text2); }
  .controls input, .controls select {
    background: var(--card);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 8px 12px;
    border-radius: 8px;
    font-family: inherit;
    font-size: 13px;
  }
  .controls input:focus, .controls select:focus {
    border-color: var(--accent);
    outline: none;
  }
  .filter-group { display: flex; flex-direction: column; gap: 4px; }

  .btn {
    padding: 10px 24px;
    border: none;
    border-radius: 8px;
    font-family: inherit;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
  }
  .btn-primary {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: #fff;
  }
  .btn-primary:hover { opacity: 0.9; transform: translateY(-1px); }
  .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn-sm { padding: 5px 12px; font-size: 12px; }
  .btn-ghost {
    background: var(--card);
    border: 1px solid var(--border);
    color: var(--text2);
  }
  .btn-ghost:hover { border-color: var(--accent); color: var(--accent); }

  .chips { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }
  .chip {
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 12px;
    cursor: pointer;
    border: 1px solid var(--border);
    background: var(--card);
    color: var(--text2);
    transition: all 0.2s;
  }
  .chip.active { border-color: var(--accent); color: var(--accent); background: rgba(6,182,212,0.1); }
  .chip:hover { border-color: var(--accent); }

  .table-wrap {
    overflow-x: auto;
    border-radius: 12px;
    border: 1px solid var(--border);
  }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  thead th {
    background: var(--card);
    padding: 12px 10px;
    text-align: left;
    font-size: 11px;
    color: var(--text2);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    position: sticky;
    top: 0;
    white-space: nowrap;
    cursor: pointer;
    user-select: none;
    border-bottom: 2px solid var(--border);
  }
  thead th:hover { color: var(--accent); }
  tbody tr { border-bottom: 1px solid var(--border); transition: background 0.15s; }
  tbody tr:hover { background: rgba(6,182,212,0.05); }
  tbody tr.highlight { background: rgba(16,185,129,0.08); }
  tbody td { padding: 10px; white-space: nowrap; }

  .tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
  }
  .tag-pass { background: rgba(16,185,129,0.15); color: var(--green); }
  .tag-fail { background: rgba(239,68,68,0.15); color: var(--red); }
  .tag-warn { background: rgba(245,158,11,0.15); color: var(--orange); }
  .tag-info { background: rgba(59,130,246,0.15); color: var(--blue); }
  .tag-hot { background: rgba(236,72,153,0.15); color: var(--pink); }
  .tag-purple { background: rgba(139,92,246,0.15); color: var(--accent2); }

  .anomaly-bar {
    width: 60px; height: 6px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
    display: inline-block;
    vertical-align: middle;
    margin-right: 6px;
  }
  .anomaly-fill { height: 100%; border-radius: 3px; transition: width 0.3s; }

  .loading { text-align: center; padding: 60px; color: var(--text2); }
  .spinner {
    width: 40px; height: 40px;
    border: 3px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin: 0 auto 16px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Modal */
  .modal-overlay {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.7);
    z-index: 1000;
    justify-content: center;
    align-items: center;
  }
  .modal-overlay.show { display: flex; }
  .modal {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px;
    max-width: 900px;
    width: 92%;
    max-height: 85vh;
    overflow-y: auto;
  }
  .modal h2 { font-size: 20px; margin-bottom: 16px; color: var(--accent); }
  .modal .close-btn {
    float: right; background: none; border: none;
    color: var(--text2); font-size: 24px; cursor: pointer;
  }
  .modal .detail-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }
  .modal .detail-item {
    padding: 10px;
    background: var(--bg);
    border-radius: 8px;
  }
  .modal .detail-item .dl { font-size: 11px; color: var(--text2); }
  .modal .detail-item .dv { font-size: 16px; font-weight: 600; margin-top: 4px; }
  .modal .reasons {
    margin-top: 12px; padding: 12px;
    background: rgba(245,158,11,0.08);
    border-radius: 8px;
    border-left: 3px solid var(--orange);
  }
  .modal .reasons li { font-size: 13px; margin: 4px 0; color: var(--orange); }

  /* === WHALE MODULE SPECIFIC === */
  .alert-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
    transition: all 0.15s;
  }
  .alert-card:hover { border-color: var(--accent); }
  .alert-card.severity-high { border-left: 3px solid var(--red); }
  .alert-card.severity-medium { border-left: 3px solid var(--orange); }
  .alert-icon { font-size: 20px; flex-shrink: 0; margin-top: 2px; }
  .alert-body { flex: 1; }
  .alert-msg { font-size: 13px; font-weight: 600; margin-bottom: 4px; }
  .alert-detail { font-size: 11px; color: var(--text2); }
  .alert-meta { font-size: 10px; color: var(--text3); margin-top: 4px; }

  .whale-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    cursor: pointer;
    transition: all 0.15s;
  }
  .whale-card:hover { border-color: var(--accent); transform: translateY(-1px); }
  .whale-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }
  .whale-addr {
    font-size: 14px;
    font-weight: 600;
    color: var(--accent);
    font-family: monospace;
  }
  .whale-stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 8px;
  }
  .whale-stat {
    background: var(--bg);
    border-radius: 6px;
    padding: 8px 10px;
  }
  .whale-stat .ws-label { font-size: 10px; color: var(--text3); }
  .whale-stat .ws-val { font-size: 14px; font-weight: 600; margin-top: 2px; }

  .hot-subnet-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    margin-bottom: 8px;
    transition: all 0.15s;
  }
  .hot-subnet-bar:hover { border-color: var(--accent); }
  .hot-sn-name { font-weight: 600; min-width: 140px; }
  .hot-sn-bar-wrap { flex: 1; height: 8px; background: var(--border); border-radius: 4px; overflow: hidden; }
  .hot-sn-fill { height: 100%; border-radius: 4px; background: linear-gradient(90deg, var(--accent), var(--pink)); }
  .hot-sn-val { min-width: 100px; text-align: right; font-size: 13px; color: var(--accent); font-weight: 600; }
  .hot-sn-whales { font-size: 11px; color: var(--text2); min-width: 80px; text-align: right; }

  /* Tabs within whale page */
  .whale-tabs {
    display: flex;
    gap: 4px;
    margin-bottom: 20px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0;
  }
  .whale-tab {
    padding: 10px 20px;
    font-size: 13px;
    color: var(--text2);
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: all 0.15s;
    font-family: inherit;
    background: none;
    border-top: none;
    border-left: none;
    border-right: none;
  }
  .whale-tab:hover { color: var(--text); }
  .whale-tab.active { color: var(--accent); border-bottom-color: var(--accent); font-weight: 600; }
  .whale-tab-content { display: none; }
  .whale-tab-content.active { display: block; }

  .update-info {
    text-align: center;
    font-size: 11px;
    color: var(--text2);
    margin-top: 16px;
    padding: 10px;
  }

  /* Mobile */
  .mobile-toggle {
    display: none;
    position: fixed;
    top: 12px; left: 12px;
    z-index: 200;
    background: var(--card);
    border: 1px solid var(--border);
    color: var(--text);
    width: 40px; height: 40px;
    border-radius: 8px;
    font-size: 18px;
    cursor: pointer;
  }

  /* === NEWS MODULE === */
  .news-ticker {
    display: flex;
    gap: 16px;
    padding: 12px 16px;
    background: linear-gradient(135deg, var(--card), var(--card2));
    border: 1px solid var(--border);
    border-radius: 10px;
    margin-bottom: 16px;
    overflow-x: auto;
    flex-wrap: wrap;
  }
  .ticker-item {
    display: flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
  }
  .ticker-label {
    font-size: 11px;
    color: var(--text3);
    font-weight: 600;
    text-transform: uppercase;
  }
  .ticker-price {
    font-size: 16px;
    font-weight: 700;
    color: var(--accent);
  }
  .ticker-change {
    font-size: 13px;
    font-weight: 600;
    padding: 2px 6px;
    border-radius: 4px;
  }
  .ticker-change.up { color: var(--green); background: rgba(16,185,129,0.1); }
  .ticker-change.down { color: var(--red); background: rgba(239,68,68,0.1); }
  .ticker-val {
    font-size: 13px;
    color: var(--text);
    font-weight: 500;
  }
  .ntab-count {
    font-size: 10px;
    background: var(--accent);
    color: var(--bg);
    padding: 1px 5px;
    border-radius: 8px;
    margin-left: 4px;
    font-weight: 700;
  }

  .news-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 12px;
    transition: border-color 0.2s, transform 0.15s;
    cursor: default;
  }
  .news-card:hover {
    border-color: var(--accent);
    transform: translateY(-1px);
  }
  .news-card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 8px;
  }
  .news-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--text);
    line-height: 1.4;
    flex: 1;
  }
  .news-title a {
    color: var(--text);
    text-decoration: none;
  }
  .news-title a:hover {
    color: var(--accent);
    text-decoration: underline;
  }
  .news-meta {
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
    margin-top: 6px;
  }
  .news-source {
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 600;
  }
  .news-source.news { background: rgba(59,130,246,0.15); color: var(--blue); }
  .news-source.reddit { background: rgba(239,68,68,0.15); color: #ff6b35; }
  .news-source.chain_event { background: rgba(16,185,129,0.15); color: var(--green); }
  .news-source.twitter { background: rgba(59,130,246,0.15); color: #1da1f2; }
  .news-time {
    font-size: 11px;
    color: var(--text3);
  }
  .news-summary {
    font-size: 12px;
    color: var(--text2);
    line-height: 1.5;
    margin-top: 6px;
    max-height: 60px;
    overflow: hidden;
  }
  .news-engagement {
    display: flex;
    gap: 12px;
    margin-top: 8px;
    font-size: 11px;
    color: var(--text3);
  }
  .news-engagement span {
    display: flex;
    align-items: center;
    gap: 3px;
  }
  .news-card.important {
    border-left: 3px solid var(--orange);
  }
  .news-card.chain {
    border-left: 3px solid var(--green);
  }
  .news-empty {
    text-align: center;
    padding: 60px 20px;
    color: var(--text3);
  }
  .news-empty-icon {
    font-size: 48px;
    margin-bottom: 12px;
  }

  /* === LANGUAGE TOGGLE === */
  .lang-toggle {
    position: fixed;
    top: 14px;
    right: 20px;
    z-index: 200;
    display: flex;
    align-items: center;
    gap: 0;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 20px;
    overflow: hidden;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
  }
  .lang-btn {
    padding: 6px 14px;
    font-size: 12px;
    font-weight: 600;
    border: none;
    background: transparent;
    color: var(--text3);
    cursor: pointer;
    transition: all 0.2s;
    font-family: inherit;
  }
  .lang-btn.active {
    background: var(--accent);
    color: var(--bg);
  }
  .lang-btn:hover:not(.active) {
    color: var(--text);
    background: var(--border);
  }

  @media (max-width: 900px) {
    .sidebar { transform: translateX(-100%); }
    .sidebar.open { transform: translateX(0); }
    .main { margin-left: 0; }
    .mobile-toggle { display: flex; align-items: center; justify-content: center; }
    .modal .detail-grid { grid-template-columns: 1fr; }
    .whale-stats { grid-template-columns: 1fr 1fr; }
    .news-ticker { gap: 8px; }
    .lang-toggle { top: 12px; right: 60px; }
  }
</style>
</head>
<body>

<button class="mobile-toggle" onclick="document.querySelector('.sidebar').classList.toggle('open')">☰</button>

<!-- LANGUAGE TOGGLE -->
<div class="lang-toggle">
  <button class="lang-btn active" id="langZh" onclick="switchLang('zh')">中文</button>
  <button class="lang-btn" id="langEn" onclick="switchLang('en')">EN</button>
</div>

<!-- SIDEBAR -->
<aside class="sidebar">
  <div class="sidebar-header">
    <h1>⛓ Bittensor Hub</h1>
    <p>Intelligence Platform</p>
  </div>
  <nav class="sidebar-nav">
    <div class="nav-section">
      <div class="nav-section-title" data-i18n="nav_analysis">分析模块</div>
      <div class="nav-item active" onclick="switchPage('scanner',this)">
        <span class="icon">⛏</span>
        <span>Alpha Scanner</span>
      </div>
      <div class="nav-item" onclick="switchPage('whale',this)">
        <span class="icon">🐋</span>
        <span data-i18n="nav_whale">鲸鱼监测</span>
        <span class="badge" id="whaleAlertBadge" style="display:none">0</span>
      </div>
      <div class="nav-item" onclick="switchPage('news',this)">
        <span class="icon">📰</span>
        <span data-i18n="nav_news">最新消息</span>
        <span class="badge" id="newsBadge" style="display:none">0</span>
      </div>
    </div>
    <div class="nav-section">
      <div class="nav-section-title" data-i18n="nav_quick">快捷操作</div>
      <div class="nav-item" onclick="startScannerScan()">
        <span class="icon">🔄</span>
        <span data-i18n="nav_refresh_subnet">刷新子网数据</span>
      </div>
      <div class="nav-item" onclick="startWhaleScan()">
        <span class="icon">🔍</span>
        <span data-i18n="nav_scan_whale">扫描鲸鱼</span>
      </div>
      <div class="nav-item" onclick="loadNews()">
        <span class="icon">📡</span>
        <span data-i18n="nav_refresh_news">刷新消息</span>
      </div>
    </div>
  </nav>
  <div class="sidebar-footer" id="sidebarFooter">
    宝宝 Bittensor Intelligence<br>
    <span style="color:var(--accent)">v3.0</span> · <span data-i18n="footer_desc">实时链上分析</span>
  </div>
</aside>

<!-- MAIN CONTENT -->
<div class="main">

  <!-- ====== PAGE 1: ALPHA SCANNER ====== -->
  <div class="page active" id="page-scanner">
    <div class="page-header">
      <h2>⛏ Subnet Alpha Scanner</h2>
      <p data-i18n="scanner_desc">实时扫描子网Alpha代币排放 · 筛选最佳挖矿机会 · 检测注册异常</p>
    </div>

    <div class="stats-bar" id="scannerStats">
      <div class="stat-card"><div class="val" id="statTotal">-</div><div class="label" data-i18n="s_total">扫描子网</div></div>
      <div class="stat-card"><div class="val" id="statEmitting">-</div><div class="label" data-i18n="s_emitting">有排放</div></div>
      <div class="stat-card good"><div class="val" id="statPassAll">-</div><div class="label" data-i18n="s_pass">全部通过</div></div>
      <div class="stat-card"><div class="val" id="statMiners">-</div><div class="label" data-i18n="s_miners">矿工>10</div></div>
      <div class="stat-card"><div class="val" id="statStable">-</div><div class="label" data-i18n="s_stable">价格稳定</div></div>
      <div class="stat-card warn"><div class="val" id="statAnomaly">-</div><div class="label" data-i18n="s_anomaly">注册异常</div></div>
    </div>

    <div class="controls">
      <div class="filter-group">
        <label data-i18n="f_min_miners">最小矿工数</label>
        <input type="number" id="minMiners" value="10" min="0" max="500" style="width:80px">
      </div>
      <div class="filter-group">
        <label data-i18n="f_max_vol">最大24h波动%</label>
        <input type="number" id="maxVolatility" value="5" min="0" max="100" step="0.5" style="width:80px">
      </div>
      <div class="filter-group">
        <label data-i18n="f_min_anomaly">最小异常分数</label>
        <input type="number" id="minAnomaly" value="30" min="0" max="100" style="width:80px">
      </div>
      <div class="filter-group">
        <label data-i18n="f_interval">排放间隔(h)</label>
        <input type="number" id="maxInterval" value="12" min="0.1" max="24" step="0.1" style="width:80px">
      </div>
      <div class="filter-group">
        <label data-i18n="f_sort">排序</label>
        <select id="sortBy">
          <option value="score" data-i18n="sort_score">综合评分</option>
          <option value="anomaly">注册异常↓</option>
          <option value="miners">活跃矿工↓</option>
          <option value="volatility">波动最低↑</option>
          <option value="emission" data-i18n="sort_emission">排放量↓</option>
        </select>
      </div>
      <button class="btn btn-primary" id="btnScan" onclick="startScannerScan()">🔍 <span data-i18n="btn_scan">开始扫描</span></button>
    </div>

    <div class="chips">
      <div class="chip active" data-filter="all" onclick="setScanFilter('all',this)" data-i18n="chip_all">全部</div>
      <div class="chip" data-filter="pass" onclick="setScanFilter('pass',this)" data-i18n="chip_pass">✅ 全部通过</div>
      <div class="chip" data-filter="miners" onclick="setScanFilter('miners',this)" data-i18n="chip_miners">⛏ 矿工充足</div>
      <div class="chip" data-filter="stable" onclick="setScanFilter('stable',this)" data-i18n="chip_stable">📊 价格稳定</div>
      <div class="chip" data-filter="anomaly" onclick="setScanFilter('anomaly',this)" data-i18n="chip_anomaly">🔥 注册异常</div>
    </div>

    <div class="loading" id="scannerLoading" style="display:none">
      <div class="spinner"></div>
      <div data-i18n="scan_loading">正在扫描 ~129 个子网，预计30-60秒...</div>
    </div>

    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th onclick="sortScanTable('netuid')">SN#</th>
            <th onclick="sortScanTable('name')" data-i18n="th_name">子网名称</th>
            <th onclick="sortScanTable('emission_interval_hours')" data-i18n="th_interval">排放间隔</th>
            <th onclick="sortScanTable('alpha_emission_per_tempo')" data-i18n="th_alpha">Alpha排放</th>
            <th onclick="sortScanTable('price_tao')" data-i18n="th_price">价格(TAO)</th>
            <th onclick="sortScanTable('volatility_24h')" data-i18n="th_vol">24h波动</th>
            <th onclick="sortScanTable('active_miners')" data-i18n="th_miners">矿工</th>
            <th onclick="sortScanTable('miner_emission_pct')" data-i18n="th_miner_pct">矿工排放%</th>
            <th onclick="sortScanTable('registration.score')" data-i18n="th_reg_anomaly">注册异常</th>
            <th onclick="sortScanTable('tao_liquidity')" data-i18n="th_liquidity">流动性</th>
            <th data-i18n="th_status">状态</th>
            <th data-i18n="th_detail">详情</th>
          </tr>
        </thead>
        <tbody id="scannerBody">
          <tr><td colspan="12" style="text-align:center;padding:40px;color:var(--text2)" data-i18n="scan_empty">点击"开始扫描"加载数据</td></tr>
        </tbody>
      </table>
    </div>

    <div class="update-info" id="scannerUpdateInfo"></div>
  </div>

  <!-- ====== PAGE 2: WHALE MONITOR ====== -->
  <div class="page" id="page-whale">
    <div class="page-header">
      <h2>🐋 <span data-i18n="whale_title">鲸鱼链上监测</span></h2>
      <p data-i18n="whale_desc">追踪大户Alpha代币买入 · 检测异常建仓 · 分钱包买入预警 · 链上操作记录</p>
    </div>

    <div class="stats-bar" id="whaleStats">
      <div class="stat-card"><div class="val" id="wStatTxs">-</div><div class="label" data-i18n="w_txs">分析交易数</div></div>
      <div class="stat-card"><div class="val" id="wStatWhales">-</div><div class="label" data-i18n="w_whales">活跃鲸鱼</div></div>
      <div class="stat-card hot"><div class="val" id="wStatAlerts">-</div><div class="label" data-i18n="w_alerts">异常预警</div></div>
      <div class="stat-card warn"><div class="val" id="wStatSplit">-</div><div class="label" data-i18n="w_split">分钱包预警</div></div>
      <div class="stat-card"><div class="val" id="wStatHotSN">-</div><div class="label" data-i18n="w_hot">热门子网</div></div>
    </div>

    <div style="display:flex;gap:12px;margin-bottom:20px;align-items:center">
      <button class="btn btn-primary" id="btnWhale" onclick="startWhaleScan()">🐋 <span data-i18n="btn_whale">扫描鲸鱼</span></button>
      <span style="font-size:11px;color:var(--text3)" id="whaleUpdateInfo" data-i18n="whale_empty">点击按钮开始扫描链上数据</span>
    </div>

    <div class="loading" id="whaleLoading" style="display:none">
      <div class="spinner"></div>
      <div data-i18n="whale_loading">正在爬取链上交易记录，分析鲸鱼行为... 预计30-90秒</div>
    </div>

    <!-- Whale Tabs -->
    <div class="whale-tabs" id="whaleTabs" style="display:none">
      <button class="whale-tab active" onclick="switchWhaleTab('alerts',this)">🚨 <span data-i18n="wtab_alerts">异常预警</span></button>
      <button class="whale-tab" onclick="switchWhaleTab('hotsubnets',this)">🔥 <span data-i18n="wtab_hot">热门子网</span></button>
      <button class="whale-tab" onclick="switchWhaleTab('whalelist',this)">🐋 <span data-i18n="wtab_list">鲸鱼列表</span></button>
      <button class="whale-tab" onclick="switchWhaleTab('split',this)">🕵 <span data-i18n="wtab_split">分钱包检测</span></button>
    </div>

    <!-- Tab: Alerts -->
    <div class="whale-tab-content active" id="wtab-alerts">
      <div id="alertsList">
        <div style="text-align:center;padding:40px;color:var(--text2)">等待扫描...</div>
      </div>
    </div>

    <!-- Tab: Hot Subnets -->
    <div class="whale-tab-content" id="wtab-hotsubnets">
      <div id="hotSubnetsList">
        <div style="text-align:center;padding:40px;color:var(--text2)">等待扫描...</div>
      </div>
    </div>

    <!-- Tab: Whale List -->
    <div class="whale-tab-content" id="wtab-whalelist">
      <div id="whaleList">
        <div style="text-align:center;padding:40px;color:var(--text2)">等待扫描...</div>
      </div>
    </div>

    <!-- Tab: Split Wallet -->
    <div class="whale-tab-content" id="wtab-split">
      <div id="splitList">
        <div style="text-align:center;padding:40px;color:var(--text2)">等待扫描...</div>
      </div>
    </div>
  </div>

  <!-- ====== PAGE 3: NEWS ====== -->
  <div class="page" id="page-news">
    <div class="page-header">
      <h2>📰 <span data-i18n="news_title">最新消息</span></h2>
      <p data-i18n="news_desc">聚合 Bittensor 生态新闻 · Twitter/Reddit 热议 · 子网重大事件 · $TAO 市场动态</p>
    </div>

    <!-- TAO Market Ticker -->
    <div class="news-ticker" id="newsTicker" style="display:none">
      <div class="ticker-item">
        <span class="ticker-label">$TAO</span>
        <span class="ticker-price" id="taoPrice">-</span>
        <span class="ticker-change" id="taoChange24h">-</span>
      </div>
      <div class="ticker-item">
        <span class="ticker-label">7d</span>
        <span class="ticker-change" id="taoChange7d">-</span>
      </div>
      <div class="ticker-item">
        <span class="ticker-label" data-i18n="t_mcap">市值</span>
        <span class="ticker-val" id="taoMcap">-</span>
      </div>
      <div class="ticker-item">
        <span class="ticker-label" data-i18n="t_vol">24h量</span>
        <span class="ticker-val" id="taoVol">-</span>
      </div>
      <div class="ticker-item">
        <span class="ticker-label">Reddit</span>
        <span class="ticker-val" id="taoReddit">-</span>
      </div>
      <div class="ticker-item">
        <span class="ticker-label">Telegram</span>
        <span class="ticker-val" id="taoTelegram">-</span>
      </div>
    </div>

    <div class="stats-bar" id="newsStats">
      <div class="stat-card"><div class="val" id="nStatTotal">-</div><div class="label" data-i18n="n_total">总消息</div></div>
      <div class="stat-card good"><div class="val" id="nStatHour">-</div><div class="label" data-i18n="n_hour">近1小时</div></div>
      <div class="stat-card"><div class="val" id="nStatDay">-</div><div class="label" data-i18n="n_day">近1天</div></div>
      <div class="stat-card"><div class="val" id="nStatWeek">-</div><div class="label" data-i18n="n_week">近1周</div></div>
      <div class="stat-card warn"><div class="val" id="nStatImportant">-</div><div class="label" data-i18n="n_important">重要事件</div></div>
      <div class="stat-card"><div class="val" id="nStatSources">-</div><div class="label" data-i18n="n_sources">数据源</div></div>
    </div>

    <div style="display:flex;gap:12px;margin-bottom:20px;align-items:center">
      <button class="btn btn-primary" id="btnNews" onclick="loadNews()">📡 <span data-i18n="btn_news">加载消息</span></button>
      <span style="font-size:11px;color:var(--text3)" id="newsUpdateInfo" data-i18n="news_empty">点击按钮加载最新消息</span>
    </div>

    <div class="loading" id="newsLoading" style="display:none">
      <div class="spinner"></div>
      <div data-i18n="news_loading">正在聚合多个数据源... (新闻/Reddit/链上事件)</div>
    </div>

    <!-- News Time Tabs -->
    <div class="whale-tabs" id="newsTabs" style="display:none">
      <button class="whale-tab active" onclick="switchNewsTab('hour',this)">⚡ <span data-i18n="ntab_hour">近1小时</span> <span class="ntab-count" id="ntabHour">0</span></button>
      <button class="whale-tab" onclick="switchNewsTab('day',this)">📅 <span data-i18n="ntab_day">近1天</span> <span class="ntab-count" id="ntabDay">0</span></button>
      <button class="whale-tab" onclick="switchNewsTab('week',this)">📆 <span data-i18n="ntab_week">近1周</span> <span class="ntab-count" id="ntabWeek">0</span></button>
      <button class="whale-tab" onclick="switchNewsTab('important',this)">🔥 <span data-i18n="ntab_important">重要事件</span> <span class="ntab-count" id="ntabImportant">0</span></button>
    </div>

    <!-- News Content -->
    <div class="whale-tab-content active" id="ntab-hour">
      <div id="newsHourList"><div style="text-align:center;padding:40px;color:var(--text2)">等待加载...</div></div>
    </div>
    <div class="whale-tab-content" id="ntab-day">
      <div id="newsDayList"><div style="text-align:center;padding:40px;color:var(--text2)">等待加载...</div></div>
    </div>
    <div class="whale-tab-content" id="ntab-week">
      <div id="newsWeekList"><div style="text-align:center;padding:40px;color:var(--text2)">等待加载...</div></div>
    </div>
    <div class="whale-tab-content" id="ntab-important">
      <div id="newsImportantList"><div style="text-align:center;padding:40px;color:var(--text2)">等待加载...</div></div>
    </div>
  </div>
</div>

<!-- Detail Modal -->
<div class="modal-overlay" id="modalOverlay" onclick="if(event.target===this)closeModal()">
  <div class="modal">
    <button class="close-btn" onclick="closeModal()">&times;</button>
    <div id="modalContent"></div>
  </div>
</div>

<script>
// ========== I18N SYSTEM ==========
let currentLang = localStorage.getItem('bh_lang') || 'zh';

const I18N = {
  zh: {
    // Sidebar
    nav_analysis: '分析模块', nav_whale: '鲸鱼监测', nav_news: '最新消息',
    nav_quick: '快捷操作', nav_refresh_subnet: '刷新子网数据', nav_scan_whale: '扫描鲸鱼',
    nav_refresh_news: '刷新消息', footer_desc: '实时链上分析',
    // Scanner page
    scanner_desc: '实时扫描子网Alpha代币排放 · 筛选最佳挖矿机会 · 检测注册异常',
    s_total: '扫描子网', s_emitting: '有排放', s_pass: '全部通过',
    s_miners: '矿工>10', s_stable: '价格稳定', s_anomaly: '注册异常',
    f_min_miners: '最小矿工数', f_max_vol: '最大24h波动%', f_min_anomaly: '最小异常分数',
    f_interval: '排放间隔(h)', f_sort: '排序',
    sort_score: '综合评分', sort_emission: '排放量↓',
    btn_scan: '开始扫描', btn_rescan: '重新扫描', btn_scanning: '扫描中...',
    chip_all: '全部', chip_pass: '✅ 全部通过', chip_miners: '⛏ 矿工充足',
    chip_stable: '📊 价格稳定', chip_anomaly: '🔥 注册异常',
    scan_loading: '正在扫描 ~129 个子网，预计30-60秒...',
    scan_empty: '点击"开始扫描"加载数据',
    th_name: '子网名称', th_interval: '排放间隔', th_alpha: 'Alpha排放',
    th_price: '价格(TAO)', th_vol: '24h波动', th_miners: '矿工',
    th_miner_pct: '矿工排放%', th_reg_anomaly: '注册异常', th_liquidity: '流动性',
    th_status: '状态', th_detail: '详情',
    scan_no_match: '没有符合条件的子网',
    tag_recommend: '⭐ 推荐', tag_instant: '🔥 秒满', tag_rush: '⚠ 抢注', tag_normal: '正常',
    scan_update: '最后更新', scan_subnets: '个子网',
    scan_fail: '扫描失败',
    // Scanner detail modal
    d_no_desc: '暂无描述', d_interval: '排放间隔', d_hour: '小时',
    d_alpha_tempo: 'Alpha排放/周期', d_tao_tempo: 'TAO排放/周期',
    d_price: '价格(TAO)', d_vol_24h: '24h波动', d_active_miners: '活跃矿工',
    d_miner_pct: '矿工排放占比', d_capacity: '网络容量', d_reg_score: '注册异常分数',
    d_reg_progress: '注册进度', d_burn: '注册燃烧费', d_mcap: '市值(TAO)',
    d_fdv: 'FDV(TAO)', d_liq: '流动性(TAO)', d_reg_reasons: '⚠ 注册异常原因：',
    // Whale page
    whale_title: '鲸鱼链上监测',
    whale_desc: '追踪大户Alpha代币买入 · 检测异常建仓 · 分钱包买入预警 · 链上操作记录',
    w_txs: '分析交易数', w_whales: '活跃鲸鱼', w_alerts: '异常预警',
    w_split: '分钱包预警', w_hot: '热门子网',
    btn_whale: '扫描鲸鱼', whale_empty: '点击按钮开始扫描链上数据',
    whale_loading: '正在爬取链上交易记录，分析鲸鱼行为... 预计30-90秒',
    wtab_alerts: '异常预警', wtab_hot: '热门子网', wtab_list: '鲸鱼列表', wtab_split: '分钱包检测',
    whale_no_alert: '暂无异常预警', whale_no_data: '暂无数据', whale_no_whale: '暂无鲸鱼数据',
    whale_no_split: '未检测到分钱包买入模式', whale_wait: '等待扫描...',
    whale_update: '最后更新', whale_analyzed: '笔交易',
    wt_heavy: '大额买入', wt_concentrated: '集中建仓', wt_stealth: '隐蔽建仓',
    wt_highfreq: '高频交易', wt_split: '分钱包',
    ws_high: '🔴 高风险', ws_medium: '🟡 中风险',
    wl_volume: '总交易量', wl_buy: '买入', wl_sell: '卖出',
    wl_net: '净流入', wl_txcount: '交易笔数', wl_subnets: '涉及子网',
    wl_high_sev: '高危', wl_alert_count: '预警', wl_normal: '正常',
    wh_flow_desc: '鲸鱼资金集中流入的子网（按TAO总额排序）',
    wh_whales: '鲸鱼',
    ws_split_desc: '检测到多个大户钱包同时买入同一子网的异常模式',
    ws_split_detail: '多个钱包协同买入同一子网，疑似关联钱包建仓',
    // Whale detail modal
    wd_title: '🐋 鲸鱼详情', wd_loading: '加载钱包详情...',
    wd_days_txs: '天内共', wd_txs_unit: '笔交易',
    wd_overview: '概览', wd_tx_count: '交易数', wd_buy_in: '买入', wd_net_flow: '净流入',
    wd_subnet_buy: '子网买入:', wd_anomaly_behavior: '⚠ 异常行为：',
    wd_recent_tx: '最近交易记录', wd_time: '时间', wd_action: '操作',
    wd_subnet: '子网', wd_error: '错误',
    // News page
    news_title: '最新消息',
    news_desc: '聚合 Bittensor 生态新闻 · Twitter/Reddit 热议 · 子网重大事件 · $TAO 市场动态',
    t_mcap: '市值', t_vol: '24h量',
    n_total: '总消息', n_hour: '近1小时', n_day: '近1天',
    n_week: '近1周', n_important: '重要事件', n_sources: '数据源',
    btn_news: '加载消息', btn_refresh_news: '刷新消息', btn_news_loading: '加载中...',
    news_empty: '点击按钮加载最新消息',
    news_loading: '正在聚合多个数据源... (新闻/Reddit/链上事件)',
    ntab_hour: '近1小时', ntab_day: '近1天', ntab_week: '近1周', ntab_important: '重要事件',
    news_empty_hour: '近1小时暂无新消息', news_empty_day: '近24小时暂无消息',
    news_empty_week: '近7天暂无消息', news_empty_important: '暂无重要事件',
    news_update: '最后更新',
    news_wait: '等待加载...',
    time_min_ago: '分钟前', time_hour_ago: '小时前', time_day_ago: '天前',
  },
  en: {
    // Sidebar
    nav_analysis: 'Analysis', nav_whale: 'Whale Monitor', nav_news: 'Latest News',
    nav_quick: 'Quick Actions', nav_refresh_subnet: 'Refresh Subnets', nav_scan_whale: 'Scan Whales',
    nav_refresh_news: 'Refresh News', footer_desc: 'Real-time On-chain Analytics',
    // Scanner page
    scanner_desc: 'Scan subnet Alpha token emissions · Find best mining opportunities · Detect registration anomalies',
    s_total: 'Subnets Scanned', s_emitting: 'Emitting', s_pass: 'All Pass',
    s_miners: 'Miners>10', s_stable: 'Price Stable', s_anomaly: 'Reg Anomaly',
    f_min_miners: 'Min Miners', f_max_vol: 'Max 24h Vol%', f_min_anomaly: 'Min Anomaly Score',
    f_interval: 'Interval(h)', f_sort: 'Sort',
    sort_score: 'Overall Score', sort_emission: 'Emission↓',
    btn_scan: 'Start Scan', btn_rescan: 'Rescan', btn_scanning: 'Scanning...',
    chip_all: 'All', chip_pass: '✅ All Pass', chip_miners: '⛏ Enough Miners',
    chip_stable: '📊 Stable Price', chip_anomaly: '🔥 Reg Anomaly',
    scan_loading: 'Scanning ~129 subnets, estimated 30-60s...',
    scan_empty: 'Click "Start Scan" to load data',
    th_name: 'Subnet Name', th_interval: 'Interval', th_alpha: 'Alpha Emission',
    th_price: 'Price(TAO)', th_vol: '24h Vol', th_miners: 'Miners',
    th_miner_pct: 'Miner Emission%', th_reg_anomaly: 'Reg Anomaly', th_liquidity: 'Liquidity',
    th_status: 'Status', th_detail: 'Detail',
    scan_no_match: 'No subnets match the criteria',
    tag_recommend: '⭐ Recommend', tag_instant: '🔥 Instant Full', tag_rush: '⚠ Rush', tag_normal: 'Normal',
    scan_update: 'Last updated', scan_subnets: 'subnets',
    scan_fail: 'Scan failed',
    // Scanner detail modal
    d_no_desc: 'No description', d_interval: 'Emission Interval', d_hour: 'hours',
    d_alpha_tempo: 'Alpha/Tempo', d_tao_tempo: 'TAO/Tempo',
    d_price: 'Price(TAO)', d_vol_24h: '24h Volatility', d_active_miners: 'Active Miners',
    d_miner_pct: 'Miner Emission%', d_capacity: 'Network Capacity', d_reg_score: 'Reg Anomaly Score',
    d_reg_progress: 'Reg Progress', d_burn: 'Reg Burn Cost', d_mcap: 'MCap(TAO)',
    d_fdv: 'FDV(TAO)', d_liq: 'Liquidity(TAO)', d_reg_reasons: '⚠ Registration Anomaly Reasons:',
    // Whale page
    whale_title: 'Whale On-chain Monitor',
    whale_desc: 'Track whale Alpha token buys · Detect abnormal accumulation · Split wallet alerts · On-chain records',
    w_txs: 'Txs Analyzed', w_whales: 'Active Whales', w_alerts: 'Anomaly Alerts',
    w_split: 'Split Wallet Alerts', w_hot: 'Hot Subnets',
    btn_whale: 'Scan Whales', whale_empty: 'Click button to start scanning on-chain data',
    whale_loading: 'Crawling on-chain transactions, analyzing whale behavior... ~30-90s',
    wtab_alerts: 'Anomaly Alerts', wtab_hot: 'Hot Subnets', wtab_list: 'Whale List', wtab_split: 'Split Wallet',
    whale_no_alert: 'No anomaly alerts', whale_no_data: 'No data yet', whale_no_whale: 'No whale data',
    whale_no_split: 'No split wallet patterns detected', whale_wait: 'Waiting for scan...',
    whale_update: 'Last updated', whale_analyzed: 'transactions',
    wt_heavy: 'Heavy Buy', wt_concentrated: 'Concentrated Buy', wt_stealth: 'Stealth Accumulation',
    wt_highfreq: 'High Frequency', wt_split: 'Split Wallet',
    ws_high: '🔴 High Risk', ws_medium: '🟡 Medium Risk',
    wl_volume: 'Total Volume', wl_buy: 'Buy', wl_sell: 'Sell',
    wl_net: 'Net Inflow', wl_txcount: 'Tx Count', wl_subnets: 'Subnets',
    wl_high_sev: 'High', wl_alert_count: 'Alerts', wl_normal: 'Normal',
    wh_flow_desc: 'Subnets with concentrated whale capital inflow (sorted by TAO total)',
    wh_whales: 'whales',
    ws_split_desc: 'Detected multiple wallets buying same subnet simultaneously',
    ws_split_detail: 'Multiple wallets coordinated buying same subnet, suspected linked wallets',
    // Whale detail modal
    wd_title: '🐋 Whale Detail', wd_loading: 'Loading wallet details...',
    wd_days_txs: 'days total', wd_txs_unit: 'transactions',
    wd_overview: 'Overview', wd_tx_count: 'Tx Count', wd_buy_in: 'Buy', wd_net_flow: 'Net Inflow',
    wd_subnet_buy: 'Subnet Buys:', wd_anomaly_behavior: '⚠ Anomalous Behavior:',
    wd_recent_tx: 'Recent Transactions', wd_time: 'Time', wd_action: 'Action',
    wd_subnet: 'Subnet', wd_error: 'Error',
    // News page
    news_title: 'Latest News',
    news_desc: 'Aggregated Bittensor ecosystem news · Twitter/Reddit discussions · Subnet events · $TAO market data',
    t_mcap: 'MCap', t_vol: '24h Vol',
    n_total: 'Total', n_hour: 'Last 1h', n_day: 'Last 1d',
    n_week: 'Last 1w', n_important: 'Important', n_sources: 'Sources',
    btn_news: 'Load News', btn_refresh_news: 'Refresh News', btn_news_loading: 'Loading...',
    news_empty: 'Click button to load latest news',
    news_loading: 'Aggregating multiple data sources... (News/Reddit/Chain events)',
    ntab_hour: 'Last 1h', ntab_day: 'Last 1d', ntab_week: 'Last 1w', ntab_important: 'Important',
    news_empty_hour: 'No news in the last hour', news_empty_day: 'No news in the last 24h',
    news_empty_week: 'No news in the last 7 days', news_empty_important: 'No important events',
    news_update: 'Last updated',
    news_wait: 'Waiting to load...',
    time_min_ago: 'min ago', time_hour_ago: 'h ago', time_day_ago: 'd ago',
  }
};

function t(key) {
  return I18N[currentLang]?.[key] || I18N['zh'][key] || key;
}

function switchLang(lang) {
  currentLang = lang;
  localStorage.setItem('bh_lang', lang);
  document.getElementById('langZh').classList.toggle('active', lang === 'zh');
  document.getElementById('langEn').classList.toggle('active', lang === 'en');
  document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
  // Update all static elements with data-i18n
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    if (el.tagName === 'OPTION') el.textContent = t(key);
    else el.textContent = t(key);
  });
  // Re-render dynamic content if data is loaded
  if (scanData) { renderScannerTable(); updateScannerStats(); }
  if (whaleData) { renderAlerts(); renderHotSubnets(); renderWhaleList(); renderSplitAlerts(); updateWhaleStats(); }
  if (newsData) {
    renderNewsList('hour', newsData.hour || [], 'newsHourList');
    renderNewsList('day', newsData.day || [], 'newsDayList');
    renderNewsList('week', newsData.week || [], 'newsWeekList');
    renderNewsList('important', newsData.important || [], 'newsImportantList');
  }
}

// ========== GLOBAL STATE ==========
let scanData = null;
let whaleData = null;
let newsData = null;
let currentScanFilter = 'all';

// ========== PAGE NAVIGATION ==========
function switchPage(page, el) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('page-' + page).classList.add('active');
  if (el) el.classList.add('active');
  // Close mobile sidebar
  document.querySelector('.sidebar').classList.remove('open');
}

// ========== SCANNER MODULE ==========
async function startScannerScan() {
  switchPage('scanner', document.querySelector('.nav-item'));
  const btn = document.getElementById('btnScan');
  const loading = document.getElementById('scannerLoading');
  const tbody = document.getElementById('scannerBody');

  btn.disabled = true;
  btn.textContent = '⏳ ' + t('btn_scanning');
  loading.style.display = 'block';
  tbody.innerHTML = '';

  try {
    const resp = await fetch('/api/scan');
    scanData = await resp.json();

    if (scanData.error && !scanData.results?.length) {
      tbody.innerHTML = `<tr><td colspan="12" style="text-align:center;padding:40px;color:var(--red)">${scanData.error}</td></tr>`;
      return;
    }
    updateScannerStats();
    renderScannerTable();
    const locale = currentLang === 'zh' ? 'zh-CN' : 'en-US';
    document.getElementById('scannerUpdateInfo').textContent =
      `${t('scan_update')}: ${new Date(scanData.timestamp).toLocaleString(locale)} · ${scanData.total_subnets} ${t('scan_subnets')}`;
  } catch(e) {
    tbody.innerHTML = `<tr><td colspan="12" style="text-align:center;padding:40px;color:var(--red)">${t('scan_fail')}: ${e.message}</td></tr>`;
  } finally {
    btn.disabled = false;
    btn.textContent = '🔍 ' + t('btn_rescan');
    loading.style.display = 'none';
  }
}

function updateScannerStats() {
  if (!scanData) return;
  const s = scanData.filters_summary;
  document.getElementById('statTotal').textContent = scanData.total_subnets;
  document.getElementById('statEmitting').textContent = scanData.candidates_with_emission;
  document.getElementById('statPassAll').textContent = s.pass_all;
  document.getElementById('statMiners').textContent = s.pass_miners_gt10;
  document.getElementById('statStable').textContent = s.pass_volatility_lt5;
  document.getElementById('statAnomaly').textContent = s.pass_reg_anomaly;
}

function getScanFiltered() {
  if (!scanData?.results) return [];
  const minMiners = parseInt(document.getElementById('minMiners').value) || 0;
  const maxVol = parseFloat(document.getElementById('maxVolatility').value) || 100;
  const minAnomaly = parseInt(document.getElementById('minAnomaly').value) || 0;
  const maxInterval = parseFloat(document.getElementById('maxInterval').value) || 24;

  let filtered = scanData.results.filter(r => {
    if (r.emission_interval_hours > maxInterval) return false;
    switch(currentScanFilter) {
      case 'pass':
        return r.active_miners > minMiners && r.volatility_24h !== null && r.volatility_24h <= maxVol && r.registration.score >= minAnomaly;
      case 'miners': return r.active_miners > minMiners;
      case 'stable': return r.volatility_24h !== null && r.volatility_24h <= maxVol;
      case 'anomaly': return r.registration.score >= minAnomaly;
      default: return true;
    }
  });

  const sortBy = document.getElementById('sortBy').value;
  filtered.sort((a, b) => {
    switch(sortBy) {
      case 'anomaly': return b.registration.score - a.registration.score;
      case 'miners': return b.active_miners - a.active_miners;
      case 'volatility': return (a.volatility_24h ?? 999) - (b.volatility_24h ?? 999);
      case 'emission': return b.alpha_emission_per_tempo - a.alpha_emission_per_tempo;
      default:
        let sa = (a.pass_miners?25:0) + (a.pass_volatility?25:0) + (a.pass_reg_anomaly?25:0) + Math.min(a.registration.score, 25);
        let sb = (b.pass_miners?25:0) + (b.pass_volatility?25:0) + (b.pass_reg_anomaly?25:0) + Math.min(b.registration.score, 25);
        return sb - sa;
    }
  });
  return filtered;
}

function renderScannerTable() {
  const tbody = document.getElementById('scannerBody');
  const items = getScanFiltered();

  if (!items.length) {
    tbody.innerHTML = `<tr><td colspan="12" style="text-align:center;padding:40px;color:var(--text2)">${t('scan_no_match')}</td></tr>`;
    return;
  }

  tbody.innerHTML = items.map(r => {
    const mm = parseInt(document.getElementById('minMiners').value)||10;
    const mv = parseFloat(document.getElementById('maxVolatility').value)||5;
    const ma = parseInt(document.getElementById('minAnomaly').value)||30;
    const isAllPass = r.active_miners > mm && r.volatility_24h !== null && r.volatility_24h <= mv && r.registration.score >= ma;

    const anomalyColor = r.registration.score >= 70 ? 'var(--red)' : r.registration.score >= 40 ? 'var(--orange)' : r.registration.score >= 20 ? 'var(--blue)' : 'var(--text2)';
    const volTag = r.volatility_24h === null ? '<span class="tag tag-info">N/A</span>' :
                   r.volatility_24h <= 5 ? `<span class="tag tag-pass">${r.volatility_24h}%</span>` :
                   r.volatility_24h <= 15 ? `<span class="tag tag-warn">${r.volatility_24h}%</span>` :
                   `<span class="tag tag-fail">${r.volatility_24h}%</span>`;
    const minerTag = r.active_miners > 10 ? `<span class="tag tag-pass">${r.active_miners}</span>` : `<span class="tag tag-fail">${r.active_miners}</span>`;

    let statusTags = '';
    if (isAllPass) statusTags += `<span class="tag tag-hot">${t('tag_recommend')}</span> `;
    if (r.registration.score >= 70) statusTags += `<span class="tag tag-fail">${t('tag_instant')}</span>`;
    else if (r.registration.score >= 40) statusTags += `<span class="tag tag-warn">${t('tag_rush')}</span>`;

    return `<tr class="${isAllPass ? 'highlight' : ''}">
      <td><strong style="color:var(--accent)">SN${r.netuid}</strong></td>
      <td><div style="max-width:150px;overflow:hidden;text-overflow:ellipsis">${r.name}</div><div style="font-size:10px;color:var(--text2)">${r.symbol}</div></td>
      <td>${r.emission_interval_hours}h</td>
      <td>${r.alpha_emission_display} α</td>
      <td>${r.price_display}</td>
      <td>${volTag}</td>
      <td>${minerTag} <span style="font-size:10px;color:var(--text2)">/${r.total_miners}</span></td>
      <td>${r.miner_emission_pct}%</td>
      <td><div class="anomaly-bar"><div class="anomaly-fill" style="width:${r.registration.score}%;background:${anomalyColor}"></div></div><span style="color:${anomalyColor}">${r.registration.score}</span></td>
      <td>${r.tao_liquidity} τ</td>
      <td>${statusTags || `<span class="tag tag-info">${t('tag_normal')}</span>`}</td>
      <td><button class="btn btn-sm btn-ghost" onclick='showScanDetail(${JSON.stringify(r).replace(/'/g,"&#39;")})'>📋</button></td>
    </tr>`;
  }).join('');
}

function setScanFilter(f, el) {
  currentScanFilter = f;
  document.querySelectorAll('#page-scanner .chip').forEach(c => c.classList.remove('active'));
  el.classList.add('active');
  renderScannerTable();
}

function sortScanTable(key) {
  if (!scanData?.results) return;
  scanData.results.sort((a, b) => {
    let va = key.includes('.') ? key.split('.').reduce((o,k) => o?.[k], a) : a[key];
    let vb = key.includes('.') ? key.split('.').reduce((o,k) => o?.[k], b) : b[key];
    if (va == null) va = -Infinity;
    if (vb == null) vb = -Infinity;
    if (typeof va === 'string') return va.localeCompare(vb);
    return vb - va;
  });
  renderScannerTable();
}

function showScanDetail(r) {
  const content = document.getElementById('modalContent');
  const reasons = r.registration.reasons.length ?
    `<div class="reasons"><strong>${t('d_reg_reasons')}</strong><ul>${r.registration.reasons.map(x=>`<li>${x}</li>`).join('')}</ul></div>` : '';
  content.innerHTML = `
    <h2>SN${r.netuid} - ${r.name}</h2>
    <p style="color:var(--text2);font-size:13px;margin-bottom:16px">${r.description || t('d_no_desc')}</p>
    ${r.github ? `<p style="margin-bottom:16px"><a href="${r.github}" target="_blank" style="color:var(--accent)">📂 GitHub</a></p>` : ''}
    <div class="detail-grid">
      <div class="detail-item"><div class="dl">${t('d_interval')}</div><div class="dv">${r.emission_interval_hours} ${t('d_hour')}</div></div>
      <div class="detail-item"><div class="dl">Tempo</div><div class="dv">${r.tempo} blocks</div></div>
      <div class="detail-item"><div class="dl">${t('d_alpha_tempo')}</div><div class="dv">${r.alpha_emission_display} α</div></div>
      <div class="detail-item"><div class="dl">${t('d_tao_tempo')}</div><div class="dv">${r.tao_emission_display} τ</div></div>
      <div class="detail-item"><div class="dl">${t('d_price')}</div><div class="dv">${r.price_display}</div></div>
      <div class="detail-item"><div class="dl">${t('d_vol_24h')}</div><div class="dv">${r.volatility_24h !== null ? r.volatility_24h + '%' : 'N/A'}</div></div>
      <div class="detail-item"><div class="dl">${t('d_active_miners')}</div><div class="dv">${r.active_miners} / ${r.total_miners}</div></div>
      <div class="detail-item"><div class="dl">${t('d_miner_pct')}</div><div class="dv">${r.miner_emission_pct}%</div></div>
      <div class="detail-item"><div class="dl">${t('d_capacity')}</div><div class="dv">${r.subnetwork_n} / ${r.max_uids}</div></div>
      <div class="detail-item"><div class="dl">${t('d_reg_score')}</div><div class="dv" style="color:${r.registration.score>=50?'var(--red)':'var(--orange)'}">${r.registration.score}/100</div></div>
      <div class="detail-item"><div class="dl">${t('d_reg_progress')}</div><div class="dv">${r.registration.reg_this}/${r.registration.target} (${r.registration.progress_pct}%)</div></div>
      <div class="detail-item"><div class="dl">${t('d_burn')}</div><div class="dv">${r.burn_cost} τ</div></div>
      <div class="detail-item"><div class="dl">${t('d_mcap')}</div><div class="dv">${r.market_cap_tao?.toLocaleString()}</div></div>
      <div class="detail-item"><div class="dl">${t('d_fdv')}</div><div class="dv">${r.fdv_tao?.toLocaleString()}</div></div>
      <div class="detail-item"><div class="dl">${t('d_liq')}</div><div class="dv">${r.tao_liquidity}</div></div>
    </div>
    ${reasons}`;
  document.getElementById('modalOverlay').classList.add('show');
}

// ========== WHALE MODULE ==========
async function startWhaleScan() {
  switchPage('whale', document.querySelectorAll('.nav-item')[1]);
  const btn = document.getElementById('btnWhale');
  const loading = document.getElementById('whaleLoading');

  btn.disabled = true;
  btn.textContent = '⏳ ' + t('btn_scanning');
  loading.style.display = 'block';
  document.getElementById('whaleTabs').style.display = 'none';

  try {
    const resp = await fetch('/api/whales');
    whaleData = await resp.json();

    if (whaleData.error && !whaleData.whales?.length) {
      document.getElementById('alertsList').innerHTML = `<div style="text-align:center;padding:40px;color:var(--red)">${whaleData.error}</div>`;
      return;
    }

    document.getElementById('whaleTabs').style.display = 'flex';
    document.getElementById('whaleUpdateInfo').removeAttribute('data-i18n');
    updateWhaleStats();
    renderAlerts();
    renderHotSubnets();
    renderWhaleList();
    renderSplitAlerts();

    const locale = currentLang === 'zh' ? 'zh-CN' : 'en-US';
    document.getElementById('whaleUpdateInfo').textContent =
      `${t('whale_update')}: ${new Date(whaleData.timestamp).toLocaleString(locale)} · ${whaleData.total_txs_analyzed} ${t('whale_analyzed')}`;

    // Update sidebar badge
    const badge = document.getElementById('whaleAlertBadge');
    if (whaleData.alert_count > 0) {
      badge.textContent = whaleData.alert_count;
      badge.style.display = 'inline';
    }
  } catch(e) {
    document.getElementById('alertsList').innerHTML = `<div style="text-align:center;padding:40px;color:var(--red)">扫描失败: ${e.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = '🐋 ' + t('btn_whale');
    loading.style.display = 'none';
  }
}

function updateWhaleStats() {
  if (!whaleData) return;
  document.getElementById('wStatTxs').textContent = whaleData.total_txs_analyzed?.toLocaleString() || '-';
  document.getElementById('wStatWhales').textContent = whaleData.whale_count || '-';
  document.getElementById('wStatAlerts').textContent = whaleData.alert_count || '0';
  document.getElementById('wStatSplit').textContent = whaleData.split_wallet_alerts?.length || '0';
  document.getElementById('wStatHotSN').textContent = whaleData.hot_subnets?.length || '0';
}

function switchWhaleTab(tab, el) {
  document.querySelectorAll('.whale-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.whale-tab-content').forEach(t => t.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('wtab-' + tab).classList.add('active');
}

function renderAlerts() {
  const container = document.getElementById('alertsList');
  const alerts = whaleData?.alerts || [];

  if (!alerts.length) {
    container.innerHTML = `<div style="text-align:center;padding:40px;color:var(--text2)">${t('whale_no_alert')}</div>`;
    return;
  }

  container.innerHTML = alerts.map(a => {
    const icons = {
      heavy_buy: '💰', concentrated_buy: '🎯', stealth_accumulate: '🕵',
      high_frequency: '⚡', split_wallet: '🔗'
    };
    const typeLabels = {
      heavy_buy: t('wt_heavy'), concentrated_buy: t('wt_concentrated'), stealth_accumulate: t('wt_stealth'),
      high_frequency: t('wt_highfreq'), split_wallet: t('wt_split')
    };
    return `<div class="alert-card severity-${a.severity}">
      <span class="alert-icon">${icons[a.type] || '⚠'}</span>
      <div class="alert-body">
        <div class="alert-msg">${a.msg}</div>
        <div class="alert-detail">${a.detail || ''}</div>
        <div class="alert-meta">
          <span class="tag ${a.severity==='high'?'tag-fail':'tag-warn'}">${a.severity==='high'?t('ws_high'):t('ws_medium')}</span>
          <span class="tag tag-info">${typeLabels[a.type] || a.type}</span>
          ${a.whale ? `<span class="tag tag-purple" style="cursor:pointer" onclick="showWhaleDetail('${a.whale_full}')">${a.whale}</span>` : ''}
          ${a.netuid ? `<span class="tag tag-info">SN${a.netuid}</span>` : ''}
        </div>
      </div>
    </div>`;
  }).join('');
}

function renderHotSubnets() {
  const container = document.getElementById('hotSubnetsList');
  const hot = whaleData?.hot_subnets || [];

  if (!hot.length) {
    container.innerHTML = `<div style="text-align:center;padding:40px;color:var(--text2)">${t('whale_no_data')}</div>`;
    return;
  }

  const maxTao = Math.max(...hot.map(h => h.total_tao));

  container.innerHTML = `<div style="margin-bottom:12px;font-size:12px;color:var(--text2)">${t('wh_flow_desc')}</div>` +
    hot.map(h => {
      const pct = (h.total_tao / maxTao * 100).toFixed(1);
      return `<div class="hot-subnet-bar">
        <div class="hot-sn-name"><span style="color:var(--accent)">SN${h.netuid}</span> ${h.name || ''}</div>
        <div class="hot-sn-bar-wrap"><div class="hot-sn-fill" style="width:${pct}%"></div></div>
        <div class="hot-sn-val">${h.total_tao.toFixed(1)} τ</div>
        <div class="hot-sn-whales">${h.whale_count} ${t('wh_whales')}</div>
      </div>`;
    }).join('');
}

function renderWhaleList() {
  const container = document.getElementById('whaleList');
  const whales = whaleData?.whales || [];

  if (!whales.length) {
    container.innerHTML = `<div style="text-align:center;padding:40px;color:var(--text2)">${t('whale_no_whale')}</div>`;
    return;
  }

  container.innerHTML = whales.map(w => {
    const flowColor = w.net_flow > 0 ? 'var(--green)' : w.net_flow < 0 ? 'var(--red)' : 'var(--text2)';
    const flowIcon = w.net_flow > 0 ? '📈' : w.net_flow < 0 ? '📉' : '➖';
    const alertTags = w.anomaly_count > 0 ?
      `<span class="tag tag-fail">${w.high_severity_count} ${t('wl_high_sev')}</span> <span class="tag tag-warn">${w.anomaly_count} ${t('wl_alert_count')}</span>` :
      `<span class="tag tag-pass">${t('wl_normal')}</span>`;

    const snTags = (w.subnet_breakdown || []).slice(0, 5).map(sn =>
      `<span class="tag tag-info" style="margin:2px">SN${sn.netuid}: ${sn.total_tao.toFixed(1)}τ</span>`
    ).join('');

    return `<div class="whale-card" onclick="showWhaleDetail('${w.address}')">
      <div class="whale-card-header">
        <span class="whale-addr">${w.address_short}</span>
        <div>${alertTags}</div>
      </div>
      <div class="whale-stats">
        <div class="whale-stat">
          <div class="ws-label">${t('wl_volume')}</div>
          <div class="ws-val" style="color:var(--accent)">${w.total_volume_tao.toLocaleString()} τ</div>
        </div>
        <div class="whale-stat">
          <div class="ws-label">${t('wl_buy')}</div>
          <div class="ws-val" style="color:var(--green)">${w.buy_tao.toLocaleString()} τ</div>
        </div>
        <div class="whale-stat">
          <div class="ws-label">${t('wl_sell')}</div>
          <div class="ws-val" style="color:var(--red)">${w.sell_tao.toLocaleString()} τ</div>
        </div>
        <div class="whale-stat">
          <div class="ws-label">${t('wl_net')}</div>
          <div class="ws-val" style="color:${flowColor}">${flowIcon} ${w.net_flow.toLocaleString()} τ</div>
        </div>
        <div class="whale-stat">
          <div class="ws-label">${t('wl_txcount')}</div>
          <div class="ws-val">${w.tx_count}</div>
        </div>
        <div class="whale-stat">
          <div class="ws-label">${t('wl_subnets')}</div>
          <div class="ws-val">${w.subnets_count}</div>
        </div>
      </div>
      ${snTags ? `<div style="margin-top:10px">${snTags}</div>` : ''}
    </div>`;
  }).join('');
}

function renderSplitAlerts() {
  const container = document.getElementById('splitList');
  const splits = whaleData?.split_wallet_alerts || [];

  if (!splits.length) {
    container.innerHTML = `<div style="text-align:center;padding:40px;color:var(--text2)">${t('whale_no_split')}</div>`;
    return;
  }

  container.innerHTML = `<div style="margin-bottom:12px;font-size:12px;color:var(--text2)">${t('ws_split_desc')}</div>` +
    splits.map(s => {
      const walletList = (s.wallets || []).map(w =>
        `<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--border)">
          <span class="whale-addr" style="font-size:12px;cursor:pointer" onclick="showWhaleDetail('${w.full}')">${w.addr}</span>
          <span style="color:var(--accent);font-weight:600">${w.tao.toFixed(1)} τ (${w.txs}笔)</span>
        </div>`
      ).join('');

      return `<div class="alert-card severity-high" style="flex-direction:column">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">
          <span style="font-size:24px">🔗</span>
          <div>
            <div class="alert-msg">${s.msg}</div>
            <div class="alert-detail">${t('ws_split_detail')}</div>
          </div>
        </div>
        <div style="background:var(--bg);border-radius:8px;padding:12px">${walletList}</div>
      </div>`;
    }).join('');
}

async function showWhaleDetail(address) {
  const content = document.getElementById('modalContent');
  content.innerHTML = `<div class="loading"><div class="spinner"></div><div>${t('wd_loading')}</div></div>`;
  document.getElementById('modalOverlay').classList.add('show');

  try {
    const resp = await fetch(`/api/whale/${address}?days=7`);
    const data = await resp.json();

    if (data.error) {
      content.innerHTML = `<h2>${t('wd_error')}</h2><p style="color:var(--red)">${data.error}</p>`;
      return;
    }

    // Build behavior summary
    let behaviorHtml = '';
    for (const [window, wData] of Object.entries(data.behavior || {})) {
      const flowColor = wData.net_flow > 0 ? 'var(--green)' : 'var(--red)';
      behaviorHtml += `
        <div style="background:var(--bg);border-radius:8px;padding:12px;margin-bottom:8px">
          <div style="font-weight:600;margin-bottom:8px;color:var(--accent)">${window} ${t('wd_overview')}</div>
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px">
            <div><span style="color:var(--text2);font-size:11px">${t('wd_tx_count')}</span><div>${wData.tx_count}</div></div>
            <div><span style="color:var(--text2);font-size:11px">${t('wd_buy_in')}</span><div style="color:var(--green)">${wData.total_tao_in} τ</div></div>
            <div><span style="color:var(--text2);font-size:11px">${t('wd_net_flow')}</span><div style="color:${flowColor}">${wData.net_flow} τ</div></div>
          </div>
          ${Object.keys(wData.subnets || {}).length ? `<div style="margin-top:8px;font-size:12px;color:var(--text2)">${t('wd_subnet_buy')}</div>` +
            Object.entries(wData.subnets).map(([sn, si]) =>
              `<div style="font-size:12px;padding:2px 0">SN${sn}: ${si.total_tao} τ (${si.buy_count}笔, 均价${si.avg_price?.toFixed(6)})</div>`
            ).join('') : ''}
        </div>`;
    }

    // Build anomalies
    let anomalyHtml = '';
    if (data.anomalies?.length) {
      anomalyHtml = `<div class="reasons" style="margin-top:12px"><strong>${t('wd_anomaly_behavior')}</strong><ul>` +
        data.anomalies.map(a => `<li>${a.msg} - ${a.detail || ''}</li>`).join('') + `</ul></div>`;
    }

    // Build transaction list
    let txHtml = '';
    const txs = (data.transactions || []).slice(0, 50);
    if (txs.length) {
      txHtml = `<div style="margin-top:16px">
        <div style="font-weight:600;margin-bottom:8px">${t('wd_recent_tx')} (${data.total_txs})</div>
        <div style="max-height:300px;overflow-y:auto;border:1px solid var(--border);border-radius:8px">
          <table style="font-size:11px">
            <thead><tr><th>${t('wd_time')}</th><th>${t('wd_action')}</th><th>${t('wd_subnet')}</th><th>TAO</th><th>Alpha</th><th>${t('d_price')}</th></tr></thead>
            <tbody>${txs.map(tx => {
              const t = new Date(tx.time).toLocaleString('zh-CN',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'});
              const func = tx.function?.replace('_limit','') || '';
              const isAdd = func.includes('add');
              const netuid = tx.to_netuid ?? tx.from_netuid ?? '-';
              const tao = tx.from_tao || tx.to_tao || 0;
              const alpha = tx.to_alpha || tx.from_alpha || 0;
              return `<tr>
                <td>${t}</td>
                <td><span class="tag ${isAdd?'tag-pass':'tag-fail'}">${func}</span></td>
                <td>SN${netuid}</td>
                <td>${tao.toFixed(2)} τ</td>
                <td>${alpha.toFixed(2)} α</td>
                <td>${tx.price ? parseFloat(tx.price).toFixed(6) : '-'}</td>
              </tr>`;
            }).join('')}</tbody>
          </table>
        </div>
      </div>`;
    }

    content.innerHTML = `
      <h2>${t('wd_title')}</h2>
      <div style="margin-bottom:16px">
        <span class="whale-addr" style="font-size:13px;word-break:break-all">${data.address}</span>
        <a href="https://www.tao.app/portfolio/${data.address}/transfers" target="_blank" style="color:var(--accent);font-size:12px;margin-left:8px">↗ tao.app</a>
        <a href="https://taostats.io/account/${data.address}" target="_blank" style="color:var(--accent);font-size:12px;margin-left:8px">↗ taostats</a>
      </div>
      <div style="font-size:13px;color:var(--text2);margin-bottom:16px">${data.days} ${t('wd_days_txs')} ${data.total_txs} ${t('wd_txs_unit')}</div>
      ${behaviorHtml}
      ${anomalyHtml}
      ${txHtml}`;
  } catch(e) {
    content.innerHTML = `<h2>${t('wd_error')}</h2><p style="color:var(--red)">${e.message}</p>`;
  }
}

// ========== NEWS MODULE ==========
async function loadNews() {
  switchPage('news', document.querySelectorAll('.nav-item')[2]);
  const btn = document.getElementById('btnNews');
  const loading = document.getElementById('newsLoading');

  btn.disabled = true;
  btn.textContent = '⏳ ' + t('btn_news_loading');
  loading.style.display = 'block';
  document.getElementById('newsTabs').style.display = 'none';

  try {
    const resp = await fetch('/api/news');
    newsData = await resp.json();

    if (newsData.error && !newsData.all_items?.length) {
      document.getElementById('newsHourList').innerHTML =
        `<div style="text-align:center;padding:40px;color:var(--red)">${newsData.error}</div>`;
      return;
    }

    document.getElementById('newsTabs').style.display = 'flex';
    updateNewsStats();
    updateTicker();
    renderNewsList('hour', newsData.hour || [], 'newsHourList');
    renderNewsList('day', newsData.day || [], 'newsDayList');
    renderNewsList('week', newsData.week || [], 'newsWeekList');
    renderNewsList('important', newsData.important || [], 'newsImportantList');

    // Update tab counts
    document.getElementById('ntabHour').textContent = newsData.stats?.hour_count || 0;
    document.getElementById('ntabDay').textContent = newsData.stats?.day_count || 0;
    document.getElementById('ntabWeek').textContent = newsData.stats?.week_count || 0;
    document.getElementById('ntabImportant').textContent = newsData.stats?.important_count || 0;

    const locale = currentLang === 'zh' ? 'zh-CN' : 'en-US';
    document.getElementById('newsUpdateInfo').textContent =
      `${t('news_update')}: ${new Date(newsData.timestamp).toLocaleString(locale)} · ${newsData.sources?.join(' / ')}`;
    document.getElementById('newsUpdateInfo').removeAttribute('data-i18n');

    // Update sidebar badge
    const badge = document.getElementById('newsBadge');
    const hourCount = newsData.stats?.hour_count || 0;
    if (hourCount > 0) {
      badge.textContent = hourCount;
      badge.style.display = 'inline';
    }
  } catch(e) {
    document.getElementById('newsHourList').innerHTML =
      `<div style="text-align:center;padding:40px;color:var(--red)">加载失败: ${e.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = '📡 ' + t('btn_refresh_news');
    loading.style.display = 'none';
  }
}

function updateNewsStats() {
  if (!newsData?.stats) return;
  const s = newsData.stats;
  document.getElementById('nStatTotal').textContent = s.total;
  document.getElementById('nStatHour').textContent = s.hour_count;
  document.getElementById('nStatDay').textContent = s.day_count;
  document.getElementById('nStatWeek').textContent = s.week_count;
  document.getElementById('nStatImportant').textContent = s.important_count;
  document.getElementById('nStatSources').textContent = newsData.sources?.length || 0;
}

function updateTicker() {
  const m = newsData?.tao_market;
  if (!m) return;

  document.getElementById('newsTicker').style.display = 'flex';
  document.getElementById('taoPrice').textContent = '$' + (m.price_usd?.toLocaleString(undefined, {maximumFractionDigits: 2}) || '-');

  const ch24 = document.getElementById('taoChange24h');
  ch24.textContent = (m.change_24h >= 0 ? '+' : '') + m.change_24h + '%';
  ch24.className = 'ticker-change ' + (m.change_24h >= 0 ? 'up' : 'down');

  const ch7d = document.getElementById('taoChange7d');
  ch7d.textContent = (m.change_7d >= 0 ? '+' : '') + m.change_7d + '%';
  ch7d.className = 'ticker-change ' + (m.change_7d >= 0 ? 'up' : 'down');

  document.getElementById('taoMcap').textContent = m.market_cap ?
    '$' + (m.market_cap / 1e9).toFixed(2) + 'B' : '-';
  document.getElementById('taoVol').textContent = m.volume_24h ?
    '$' + (m.volume_24h / 1e6).toFixed(1) + 'M' : '-';
  document.getElementById('taoReddit').textContent = m.reddit_subscribers?.toLocaleString() || '-';
  document.getElementById('taoTelegram').textContent = m.telegram_members?.toLocaleString() || '-';
}

function switchNewsTab(tab, el) {
  document.querySelectorAll('#page-news .whale-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('#page-news .whale-tab-content').forEach(t => t.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('ntab-' + tab).classList.add('active');
}

function renderNewsList(timeWindow, items, containerId) {
  const container = document.getElementById(containerId);

  if (!items.length) {
    const emptyMsg = {
      'hour': t('news_empty_hour'),
      'day': t('news_empty_day'),
      'week': t('news_empty_week'),
      'important': t('news_empty_important')
    };
    container.innerHTML = `<div class="news-empty">
      <div class="news-empty-icon">${timeWindow === 'important' ? '🔥' : '📭'}</div>
      <div>${emptyMsg[timeWindow] || t('whale_no_data')}</div>
    </div>`;
    return;
  }

  container.innerHTML = items.map(item => {
    const isImportant = item.score > 50 || item.comments > 20 || item.severity === 'high';
    const isChain = item.type === 'chain_event';
    const cardClass = isImportant ? 'news-card important' : isChain ? 'news-card chain' : 'news-card';

    // Source badge
    const sourceClass = item.type || 'news';
    const sourceIcon = {
      'news': '📰', 'reddit': '🔴', 'chain_event': '⛓', 'twitter': '🐦'
    }[item.type] || '📰';

    // Time display
    let timeStr = item.time_ago || '';
    if (!timeStr && item.pub_date) {
      try {
        const dt = new Date(item.pub_date);
        const hours = item.age_hours || 0;
        if (hours < 1) timeStr = Math.round(hours * 60) + ' ' + t('time_min_ago');
        else if (hours < 24) timeStr = Math.round(hours) + ' ' + t('time_hour_ago');
        else timeStr = Math.round(hours / 24) + ' ' + t('time_day_ago');
      } catch(e) { timeStr = ''; }
    }

    // Engagement (for Reddit)
    let engHtml = '';
    if (item.type === 'reddit') {
      engHtml = `<div class="news-engagement">
        <span>⬆ ${item.score || 0}</span>
        <span>💬 ${item.comments || 0}</span>
        ${item.author ? `<span>👤 u/${item.author}</span>` : ''}
        ${item.flair ? `<span class="tag tag-info" style="font-size:10px">${item.flair}</span>` : ''}
      </div>`;
    }

    // Summary
    const summary = item.summary ?
      `<div class="news-summary">${item.summary.replace(/</g,'&lt;').replace(/>/g,'&gt;')}</div>` : '';

    return `<div class="${cardClass}">
      <div class="news-card-header">
        <div class="news-title">
          ${item.url ? `<a href="${item.url}" target="_blank">${item.title}</a>` : item.title}
        </div>
      </div>
      <div class="news-meta">
        <span class="news-source ${sourceClass}">${sourceIcon} ${item.source || item.type}</span>
        ${item.category ? `<span class="tag tag-info" style="font-size:10px">${item.category}</span>` : ''}
        ${item.netuid ? `<span class="tag tag-info" style="font-size:10px">SN${item.netuid}</span>` : ''}
        <span class="news-time">🕐 ${timeStr}</span>
      </div>
      ${summary}
      ${engHtml}
    </div>`;
  }).join('');
}

function closeModal() {
  document.getElementById('modalOverlay').classList.remove('show');
}

// Listen to filter changes
['minMiners', 'maxVolatility', 'minAnomaly', 'maxInterval', 'sortBy'].forEach(id => {
  document.getElementById(id).addEventListener('change', renderScannerTable);
  document.getElementById(id).addEventListener('input', renderScannerTable);
});

// Init language on load
if (currentLang !== 'zh') switchLang(currentLang);

// Auto-scan on load
window.addEventListener('load', () => setTimeout(startScannerScan, 500));

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeModal();
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("=" * 50)
    print("Bittensor Intelligence Hub v3.0")
    print("Modules: Alpha Scanner + Whale Monitor + News Aggregator")
    print("http://0.0.0.0:7683")
    print("=" * 50)
    app.run(host="0.0.0.0", port=7683, debug=False)
