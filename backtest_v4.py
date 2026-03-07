#!/usr/bin/env python3
"""
V4 模型历史回测脚本
====================
用滚动窗口方法：每次用前N天数据训练，预测下一个7天窗口，对比实际结果。

用法:
    python3 backtest_v4.py
    python3 backtest_v4.py --min-train-days 28   # 最少训练天数
    python3 backtest_v4.py --json                 # JSON输出
"""

import json
import math
import os
import sys
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from copy import deepcopy

# 导入V4的核心组件
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elon_tweet_predictor_v4 import (
    RealDataCalibrator, EnsemblePredictorV4, HawkesProcess
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "xtracker_daily_data.json")

# Polymarket 区间定义
POLYMARKET_BINS = [
    (0, 179), (180, 199), (200, 219), (220, 239), (240, 259),
    (260, 279), (280, 299), (300, 319), (320, 339), (340, 359),
    (360, 379), (380, 399), (400, 449), (450, 499), (500, float('inf'))
]


def load_all_data():
    with open(DATA_FILE) as f:
        raw = json.load(f)
    return raw


def build_weeks(daily_counts, start_weekday=0):
    """
    将日数据按7天窗口分组。
    start_weekday=0 表示从周一开始（与Polymarket大致对齐）。
    返回 [(start_date, end_date, total_count, [daily_list]), ...]
    """
    sorted_dates = sorted(daily_counts.keys())
    if not sorted_dates:
        return []

    # 找到第一个周一（或指定weekday）
    first = datetime.strptime(sorted_dates[0], "%Y-%m-%d")
    while first.weekday() != start_weekday:
        first += timedelta(days=1)

    weeks = []
    current = first
    last_date = datetime.strptime(sorted_dates[-1], "%Y-%m-%d")

    while current + timedelta(days=6) <= last_date:
        week_start = current.strftime("%Y-%m-%d")
        week_end = (current + timedelta(days=6)).strftime("%Y-%m-%d")

        daily_list = []
        total = 0
        for d in range(7):
            ds = (current + timedelta(days=d)).strftime("%Y-%m-%d")
            c = daily_counts.get(ds, 0)
            daily_list.append((ds, c))
            total += c

        weeks.append({
            "start": week_start,
            "end": week_end,
            "total": total,
            "daily": daily_list,
        })
        current += timedelta(days=7)

    return weeks


def make_subset_data(raw, cutoff_date):
    """创建截止到 cutoff_date（不含）的数据子集"""
    subset = {
        "daily_counts": {},
        "hourly_counts": {},
    }
    for ds, count in raw["daily_counts"].items():
        if ds < cutoff_date:
            subset["daily_counts"][ds] = count
    for ds, hours in raw.get("hourly_counts", {}).items():
        if ds < cutoff_date:
            subset["hourly_counts"][ds] = hours
    return subset


def predict_week_with_subset(subset_data, week_start_date, n_sim=30000):
    """
    用subset数据训练V4模型，预测从week_start_date开始的7天总量。
    返回 (simulations_array, meta_dict)
    """
    # 写临时文件给calibrator用
    tmp_file = os.path.join(SCRIPT_DIR, "_backtest_tmp_data.json")
    with open(tmp_file, 'w') as f:
        json.dump(subset_data, f)

    try:
        cal = RealDataCalibrator(tmp_file)
        predictor = EnsemblePredictorV4(calibrator=cal)

        start_dt = datetime.strptime(week_start_date, "%Y-%m-%d")
        sims, meta = predictor.predict_week(
            current_count=0,
            days_elapsed=0.0,
            start_date=start_dt,
            total_days=7,
            n_sim=n_sim,
        )
        return sims, meta
    finally:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)


def find_bin(total):
    """找到total落在哪个Polymarket区间"""
    for lo, hi in POLYMARKET_BINS:
        if lo <= total <= hi:
            return (lo, hi)
    return None


def run_backtest(min_train_days=28, n_sim=30000, verbose=True):
    raw = load_all_data()
    weeks = build_weeks(raw["daily_counts"])

    if verbose:
        print(f"总数据: {len(raw['daily_counts'])} 天")
        print(f"可划分为 {len(weeks)} 个完整7天窗口")
        print(f"最少训练天数: {min_train_days}")
        print("=" * 80)

    results = []
    first_date = min(raw["daily_counts"].keys())
    first_dt = datetime.strptime(first_date, "%Y-%m-%d")

    for i, week in enumerate(weeks):
        week_start_dt = datetime.strptime(week["start"], "%Y-%m-%d")
        train_days = (week_start_dt - first_dt).days

        if train_days < min_train_days:
            if verbose:
                print(f"  跳过 Week {i+1} ({week['start']}~{week['end']}): 训练数据仅{train_days}天 < {min_train_days}")
            continue

        if verbose:
            print(f"\n--- Week {i+1}: {week['start']} ~ {week['end']} ---")
            print(f"  训练数据: {train_days} 天")
            print(f"  实际总量: {week['total']}")

        # 用截止到week_start的数据训练
        subset = make_subset_data(raw, week["start"])
        np.random.seed(42 + i)  # 可复现

        try:
            sims, meta = predict_week_with_subset(subset, week["start"], n_sim)
        except Exception as e:
            if verbose:
                print(f"  预测失败: {e}")
            continue

        pred_mean = float(np.mean(sims))
        pred_median = float(np.median(sims))
        pred_std = float(np.std(sims))
        ci50 = (float(np.percentile(sims, 25)), float(np.percentile(sims, 75)))
        ci90 = (float(np.percentile(sims, 5)), float(np.percentile(sims, 95)))

        actual = week["total"]
        error = pred_mean - actual
        abs_error = abs(error)
        pct_error = abs_error / max(1, actual) * 100

        # 实际值落在哪个区间
        actual_bin = find_bin(actual)
        # 模型给这个区间的概率
        if actual_bin:
            lo, hi = actual_bin
            if hi == float('inf'):
                bin_prob = float(np.mean(sims >= lo))
            else:
                bin_prob = float(np.mean((sims >= lo) & (sims <= hi)))
        else:
            bin_prob = 0

        # 模型预测最可能的区间
        bin_probs = {}
        for lo, hi in POLYMARKET_BINS:
            if hi == float('inf'):
                bp = float(np.mean(sims >= lo))
            else:
                bp = float(np.mean((sims >= lo) & (sims <= hi)))
            bin_probs[(lo, hi)] = bp
        pred_bin = max(bin_probs, key=bin_probs.get)
        pred_bin_prob = bin_probs[pred_bin]

        # 50% CI 和 90% CI 是否覆盖实际值
        in_ci50 = ci50[0] <= actual <= ci50[1]
        in_ci90 = ci90[0] <= actual <= ci90[1]

        result = {
            "week_idx": i + 1,
            "start": week["start"],
            "end": week["end"],
            "train_days": train_days,
            "actual": actual,
            "pred_mean": round(pred_mean, 1),
            "pred_median": round(pred_median, 1),
            "pred_std": round(pred_std, 1),
            "ci50": (round(ci50[0]), round(ci50[1])),
            "ci90": (round(ci90[0]), round(ci90[1])),
            "error": round(error, 1),
            "abs_error": round(abs_error, 1),
            "pct_error": round(pct_error, 1),
            "actual_bin": f"{actual_bin[0]}-{actual_bin[1]}" if actual_bin else "?",
            "actual_bin_prob": round(bin_prob * 100, 1),
            "pred_bin": f"{pred_bin[0]}-{pred_bin[1]}",
            "pred_bin_prob": round(pred_bin_prob * 100, 1),
            "bin_correct": actual_bin == pred_bin,
            "in_ci50": in_ci50,
            "in_ci90": in_ci90,
            "models": meta.get("models", {}),
        }
        results.append(result)

        if verbose:
            print(f"  预测均值: {result['pred_mean']}  中位数: {result['pred_median']}  标准差: {result['pred_std']}")
            print(f"  50% CI: [{result['ci50'][0]}, {result['ci50'][1]}]  {'OK' if in_ci50 else 'MISS'}")
            print(f"  90% CI: [{result['ci90'][0]}, {result['ci90'][1]}]  {'OK' if in_ci90 else 'MISS'}")
            print(f"  误差: {result['error']} ({result['pct_error']}%)")
            print(f"  实际区间: {result['actual_bin']} (模型概率: {result['actual_bin_prob']}%)")
            print(f"  预测最可能区间: {result['pred_bin']} ({result['pred_bin_prob']}%)"
                  f"  {'HIT' if result['bin_correct'] else 'MISS'}")

            # 各子模型预测
            if meta.get("models"):
                parts = []
                for name in ["negbin", "hawkes", "regime", "hist", "trend"]:
                    if name in meta["models"]:
                        parts.append(f"{name}={meta['models'][name]['mean']:.0f}")
                print(f"  子模型: {', '.join(parts)}")

    return results


def print_summary(results):
    if not results:
        print("\n没有足够的数据进行回测。")
        return

    n = len(results)
    print("\n" + "=" * 80)
    print(f"回测总结 ({n} 周)")
    print("=" * 80)

    # 误差指标
    errors = [r["error"] for r in results]
    abs_errors = [r["abs_error"] for r in results]
    pct_errors = [r["pct_error"] for r in results]

    mae = np.mean(abs_errors)
    rmse = math.sqrt(np.mean([e**2 for e in errors]))
    mean_bias = np.mean(errors)
    median_abs_error = np.median(abs_errors)

    print(f"\n误差指标:")
    print(f"  MAE (平均绝对误差):     {mae:.1f}")
    print(f"  RMSE (均方根误差):      {rmse:.1f}")
    print(f"  Mean Bias (平均偏差):   {mean_bias:+.1f} ({'偏高' if mean_bias > 0 else '偏低'})")
    print(f"  Median AE (中位绝对误差): {median_abs_error:.1f}")
    print(f"  平均百分比误差:          {np.mean(pct_errors):.1f}%")

    # 区间命中
    bin_hits = sum(1 for r in results if r["bin_correct"])
    print(f"\n区间预测:")
    print(f"  区间命中率: {bin_hits}/{n} = {bin_hits/n*100:.1f}%")
    print(f"  实际区间平均模型概率: {np.mean([r['actual_bin_prob'] for r in results]):.1f}%")

    # CI覆盖率
    ci50_hits = sum(1 for r in results if r["in_ci50"])
    ci90_hits = sum(1 for r in results if r["in_ci90"])
    print(f"\n置信区间覆盖:")
    print(f"  50% CI 覆盖率: {ci50_hits}/{n} = {ci50_hits/n*100:.1f}% (理想: 50%)")
    print(f"  90% CI 覆盖率: {ci90_hits}/{n} = {ci90_hits/n*100:.1f}% (理想: 90%)")

    if ci50_hits / n > 0.7:
        print(f"  -> 50% CI 过宽 (覆盖率 > 70%), 模型不确定性偏大")
    elif ci50_hits / n < 0.3:
        print(f"  -> 50% CI 过窄 (覆盖率 < 30%), 模型过度自信")

    if ci90_hits / n < 0.7:
        print(f"  -> 90% CI 覆盖不足, 存在系统性偏差或尾部风险低估")

    # 各子模型对比
    print(f"\n各子模型平均预测 vs 实际:")
    model_names = ["negbin", "hawkes", "regime", "hist", "trend", "ensemble"]
    for name in model_names:
        model_errors = []
        for r in results:
            if name in r.get("models", {}):
                pred = r["models"][name]["mean"]
                model_errors.append(pred - r["actual"])
        if model_errors:
            m_mae = np.mean([abs(e) for e in model_errors])
            m_bias = np.mean(model_errors)
            m_rmse = math.sqrt(np.mean([e**2 for e in model_errors]))
            print(f"  {name:10s}: MAE={m_mae:6.1f}  Bias={m_bias:+7.1f}  RMSE={m_rmse:6.1f}")

    # 逐周对比表
    print(f"\n逐周明细:")
    print(f"{'周':<6} {'日期区间':<25} {'实际':>6} {'预测均值':>8} {'误差':>7} {'区间':>12} {'命中':>4} {'CI90':>5}")
    print("-" * 80)
    for r in results:
        hit = "Y" if r["bin_correct"] else "N"
        ci = "Y" if r["in_ci90"] else "N"
        print(f"W{r['week_idx']:<5} {r['start']}~{r['end']}  {r['actual']:>6} {r['pred_mean']:>8.0f} {r['error']:>+7.0f} "
              f"{r['actual_bin']:>12} {hit:>4} {ci:>5}")

    # 朴素基线对比
    print(f"\n基线对比 (上周值作为预测):")
    naive_errors = []
    for i in range(1, len(results)):
        naive_pred = results[i-1]["actual"]
        actual = results[i]["actual"]
        naive_errors.append(abs(naive_pred - actual))
    if naive_errors:
        naive_mae = np.mean(naive_errors)
        model_mae_same = np.mean([r["abs_error"] for r in results[1:]])
        print(f"  朴素基线 MAE: {naive_mae:.1f}")
        print(f"  V4模型 MAE:   {model_mae_same:.1f} (同期)")
        if model_mae_same < naive_mae:
            improvement = (1 - model_mae_same / naive_mae) * 100
            print(f"  -> V4 比朴素基线好 {improvement:.1f}%")
        else:
            degradation = (model_mae_same / naive_mae - 1) * 100
            print(f"  -> V4 比朴素基线差 {degradation:.1f}%")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="V4模型回测")
    parser.add_argument("--min-train-days", type=int, default=28, help="最少训练天数")
    parser.add_argument("--n-sim", type=int, default=30000, help="蒙特卡洛模拟次数")
    parser.add_argument("--json", action="store_true", help="JSON输出")
    args = parser.parse_args()

    results = run_backtest(
        min_train_days=args.min_train_days,
        n_sim=args.n_sim,
        verbose=not args.json,
    )

    if args.json:
        # 清理不可JSON序列化的内容
        for r in results:
            if "models" in r:
                del r["models"]
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print_summary(results)


if __name__ == "__main__":
    main()
