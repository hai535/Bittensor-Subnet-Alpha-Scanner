#!/usr/bin/env python3
"""
Elon Musk Tweet Count Predictor for Polymarket v2
===================================================
多参数非线性预测模型：
1. 负二项分布（过度离散计数数据）
2. 星期效应（工作日 vs 周末，逐日权重）
3. 振荡模式检测（高低交替周期）
4. 政权切换（高/低活跃期 + 转换概率矩阵）
5. 贝叶斯周中实时更新（已知计数作为强先验）
6. 自相关 + 均值回归
7. 趋势分量（长期下降/上升）
8. 具体价格变动预测（每选项3天后预估价）

用法:
    python3 elon_tweet_predictor.py
    python3 elon_tweet_predictor.py --current-count 150 --days-elapsed 3.5
    python3 elon_tweet_predictor.py --json
"""

import argparse
import json
import math
import sys
from datetime import datetime, timedelta

try:
    import numpy as np
    from scipy import stats
except ImportError:
    print("需要安装: pip install numpy scipy")
    sys.exit(1)

np.random.seed(42)

# ============================================================
# 历史数据 (Polymarket已结算市场 - 非重叠周)
# ============================================================
# (label, start_date, days, resolved_midpoint, daily_rate)
WEEKLY_DATA = [
    ("W1-Jan20", "2026-01-20", 7, 389.5, 55.6),
    ("W2-Jan27", "2026-01-27", 7, 349.5, 49.9),
    ("W3-Feb03", "2026-02-03", 7, 369.5, 52.8),
    ("W4-Feb10", "2026-02-10", 7, 249.5, 35.6),
    ("W5-Feb17", "2026-02-17", 7, 369.5, 52.8),
    ("W6-Feb24", "2026-02-24", 7, 209.5, 29.9),
]

# 短期验证数据
SHORT_DATA = [
    ("Mar2-4", "2026-03-02", 2, 30, 15.0),    # <40 resolved
    ("Mar5-7", "2026-03-05", 2, 77, 38.5),     # 65-89 projected
]

# 星期权重 (基于Musk发帖模式研究)
# 工作日活跃度高，周末显著降低
DOW_WEIGHTS = {
    0: 1.20,  # Mon - DOGE/政务高峰
    1: 1.12,  # Tue
    2: 1.08,  # Wed
    3: 1.05,  # Thu
    4: 0.95,  # Fri - 略降
    5: 0.72,  # Sat - 显著降低
    6: 0.68,  # Sun - 最低
}
# 归一化使均值=1.0
_mean_w = sum(DOW_WEIGHTS.values()) / 7
DOW_WEIGHTS = {k: v / _mean_w for k, v in DOW_WEIGHTS.items()}


class ElonPredictor:
    def __init__(self):
        self.rates = np.array([w[4] for w in WEEKLY_DATA])
        self.totals = np.array([w[3] for w in WEEKLY_DATA])
        self.n_weeks = len(self.rates)

        # 基本统计
        self.global_mean = np.mean(self.rates)
        self.global_std = np.std(self.rates, ddof=1)

        # 趋势分析（线性回归）
        x = np.arange(self.n_weeks)
        self.trend_slope = np.polyfit(x, self.rates, 1)[0]  # 每周变化

        # 振荡模式检测
        self._detect_oscillation()

        # 政权转换矩阵
        self._build_regime_model()

        # 最近状态
        self.last_rate = self.rates[-1]
        self.last_regime = self.regimes[-1]

    def _detect_oscillation(self):
        """检测高低交替振荡"""
        rates = self.rates
        median = np.median(rates)
        self.regimes = []
        for r in rates:
            if r > median * 1.05:
                self.regimes.append("H")
            elif r < median * 0.95:
                self.regimes.append("L")
            else:
                self.regimes.append("M")

        # 计算交替频率
        alternations = 0
        for i in range(1, len(self.regimes)):
            if self.regimes[i] != self.regimes[i-1]:
                alternations += 1
        self.alt_ratio = alternations / (len(self.regimes) - 1)

        # 振荡检测: H,H,H,L,H,L → 最后4周是 H,L,H,L = 完美交替
        recent_4 = self.regimes[-4:]
        self.recent_alternating = all(
            recent_4[i] != recent_4[i+1] for i in range(len(recent_4)-1)
        )

    def _build_regime_model(self):
        """构建政权转换模型"""
        # 各政权的参数
        high_rates = self.rates[np.array(self.regimes) == "H"]
        low_rates = self.rates[np.array(self.regimes) == "L"]

        self.regime_params = {
            "H": {
                "mean": np.mean(high_rates) if len(high_rates) > 0 else self.global_mean * 1.2,
                "std": np.std(high_rates, ddof=1) if len(high_rates) > 1 else self.global_std * 0.5,
            },
            "L": {
                "mean": np.mean(low_rates) if len(low_rates) > 0 else self.global_mean * 0.7,
                "std": np.std(low_rates, ddof=1) if len(low_rates) > 1 else self.global_std * 0.5,
            },
            "M": {
                "mean": self.global_mean,
                "std": self.global_std * 0.7,
            }
        }

        # 转换概率 (基于历史观察)
        # 最近模式: H,L,H,L → 高交替概率
        if self.recent_alternating:
            self.transition = {
                "H": {"H": 0.25, "M": 0.15, "L": 0.60},
                "L": {"H": 0.55, "M": 0.20, "L": 0.25},
                "M": {"H": 0.35, "M": 0.30, "L": 0.35},
            }
        else:
            self.transition = {
                "H": {"H": 0.40, "M": 0.25, "L": 0.35},
                "L": {"H": 0.40, "M": 0.25, "L": 0.35},
                "M": {"H": 0.33, "M": 0.34, "L": 0.33},
            }

    def predict_week(self, current_count=0, days_elapsed=0.0,
                     start_date=None, total_days=7, n_sim=100000):
        """
        蒙特卡洛模拟一周的推特总量

        Returns: (simulated_totals, metadata)
        """
        if start_date is None:
            start_date = datetime(2026, 3, 3, 12, 0)

        days_remaining = total_days - days_elapsed

        # 确定本周政权
        if days_elapsed > 0 and current_count > 0:
            observed_rate = current_count / days_elapsed
        else:
            observed_rate = None

        # 政权概率（先验 from 转换矩阵）
        prior_probs = self.transition[self.last_regime]

        if observed_rate is not None:
            # 贝叶斯更新: P(regime|data) ∝ P(data|regime) × P(regime)
            posterior = {}
            for regime in ["H", "M", "L"]:
                rp = self.regime_params[regime]
                likelihood = stats.norm.pdf(observed_rate, rp["mean"], rp["std"] + 5)
                posterior[regime] = likelihood * prior_probs[regime]
            total_p = sum(posterior.values())
            posterior = {k: v/total_p for k, v in posterior.items()}
        else:
            posterior = prior_probs

        # 为每次模拟采样政权
        regime_choices = np.random.choice(
            ["H", "M", "L"],
            size=n_sim,
            p=[posterior["H"], posterior["M"], posterior["L"]]
        )

        totals = np.full(n_sim, float(current_count))

        # 为每个政权生成日均率
        base_rates = np.zeros(n_sim)
        for regime in ["H", "M", "L"]:
            mask = regime_choices == regime
            rp = self.regime_params[regime]
            n_r = np.sum(mask)
            if n_r > 0:
                if observed_rate is not None and days_elapsed >= 1:
                    # 混合先验和观测
                    weight = min(0.7, days_elapsed / (days_elapsed + 3))
                    blended_mean = weight * observed_rate + (1 - weight) * rp["mean"]
                    blended_std = rp["std"] * (1 - weight * 0.5)
                else:
                    blended_mean = rp["mean"]
                    blended_std = rp["std"]

                # 加入趋势分量
                weeks_from_last = 1  # 当前周 vs 上周
                trend_adj = self.trend_slope * weeks_from_last
                blended_mean += trend_adj

                base_rates[mask] = np.maximum(
                    5, np.random.normal(blended_mean, blended_std, n_r)
                )

        # 逐日模拟剩余天数
        for d_idx in range(int(math.ceil(days_remaining))):
            day_date = start_date + timedelta(days=days_elapsed + d_idx)
            dow = day_date.weekday()
            dow_w = DOW_WEIGHTS[dow]

            # 天数比例（处理不完整天）
            if d_idx == 0 and days_remaining != int(days_remaining):
                frac = days_remaining - int(days_remaining)
                if frac < 0.01:
                    frac = 1.0
            elif d_idx == int(math.ceil(days_remaining)) - 1:
                frac = days_remaining - int(days_remaining)
                if frac < 0.01:
                    frac = 1.0
            else:
                frac = 1.0

            day_means = base_rates * dow_w * frac

            # 负二项分布模拟（比泊松更适合过度离散数据）
            dispersion = 3.0
            for i in range(n_sim):
                dm = max(0.5, day_means[i])
                p = dispersion / (dispersion + dm)
                totals[i] += np.random.negative_binomial(dispersion, p)

        return totals.astype(int), {
            "regime_posterior": posterior,
            "observed_rate": observed_rate,
            "base_rate_mean": float(np.mean(base_rates)),
            "trend_slope": self.trend_slope,
        }

    def analyze_options(self, options, simulated_totals, meta,
                        current_count, days_elapsed, total_days,
                        start_date):
        """逐选项分析定价偏差"""
        results = []
        days_remaining = total_days - days_elapsed
        daily_rate = current_count / days_elapsed if days_elapsed > 0 else meta["base_rate_mean"]

        for (lo, hi), market_price in sorted(options.items()):
            # 模型概率
            if hi == float('inf'):
                model_prob = float(np.mean(simulated_totals >= lo))
            else:
                model_prob = float(np.mean((simulated_totals >= lo) & (simulated_totals <= hi)))

            edge = model_prob - market_price

            # EV计算
            if market_price > 0.001:
                ev_yes = model_prob / market_price - 1
                ev_no = (1 - model_prob) / (1 - market_price) - 1
            else:
                ev_yes = 99.0 if model_prob > 0.005 else 0
                ev_no = 0

            # Kelly
            if 0.005 < market_price < 0.995:
                odds = 1 / market_price - 1
                kelly_yes = max(0, (model_prob * odds - (1 - model_prob)) / odds)
                odds_no = 1 / (1 - market_price) - 1
                kelly_no = max(0, ((1 - model_prob) * odds_no - model_prob) / odds_no)
            else:
                kelly_yes = kelly_no = 0

            # 3天价格变动预测
            price_3d = self._predict_3day_price(
                lo, hi, market_price, model_prob,
                current_count, daily_rate, days_elapsed, days_remaining,
                simulated_totals, start_date
            )

            # 交易信号
            signal = self._signal(edge, model_prob, market_price, ev_yes, ev_no)

            results.append({
                "range": f"{lo}-{hi}" if hi != float('inf') else f"{lo}+",
                "lo": lo, "hi": hi,
                "market_price": market_price,
                "model_prob": model_prob,
                "edge": edge,
                "ev_yes": ev_yes,
                "ev_no": ev_no,
                "kelly_yes": kelly_yes,
                "kelly_no": kelly_no,
                "signal": signal,
                "price_3d": price_3d,
            })

        return results

    def _predict_3day_price(self, lo, hi, mp, model_prob,
                            current_count, daily_rate, days_elapsed,
                            days_remaining, sim_totals, start_date):
        """
        预测该选项未来逐天价格变动 (Day+1, Day+2, Day+3)
        核心: 用Monte Carlo模拟每天新增推文后重新计算条件概率
        """
        if days_remaining <= 0:
            return {"dir": "settled", "delta": 0, "new_price": mp, "reason": "已结算",
                    "daily_forecasts": [], "count_after_3d": current_count,
                    "projected_total": current_count, "delta_pct": 0}

        forecast_days = min(3, int(days_remaining))
        daily_forecasts = []
        cumulative_count = current_count

        for d in range(1, forecast_days + 1):
            day_date = start_date + timedelta(days=days_elapsed + d - 1)
            dow = day_date.weekday()
            dow_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][dow]

            # 该天预期推文数 (考虑星期效应)
            day_expected = daily_rate * DOW_WEIGHTS[dow]
            cumulative_count += day_expected

            new_elapsed = days_elapsed + d
            new_remaining = 7 - new_elapsed

            # 重新计算条件概率
            if new_remaining <= 0.5:
                # 接近结算 - 概率趋向二值化
                if lo <= cumulative_count <= (hi if hi != float('inf') else 9999):
                    day_prob = min(0.90, model_prob * 3.0)
                elif abs(cumulative_count - lo) < 15 or (hi != float('inf') and abs(cumulative_count - hi) < 15):
                    day_prob = model_prob * 0.6
                else:
                    day_prob = max(0.005, model_prob * 0.1)
            else:
                # 还有时间 - 计算剩余天需要的日均来命中区间
                need_lo = lo - cumulative_count  # 还需要多少才到下限
                need_hi = (hi if hi != float('inf') else lo + 19) - cumulative_count

                # 剩余天的预期总推文
                remaining_expected = 0
                remaining_sd = 0
                for rd in range(int(new_remaining)):
                    rd_date = start_date + timedelta(days=new_elapsed + rd)
                    rd_dow = rd_date.weekday()
                    remaining_expected += daily_rate * DOW_WEIGHTS[rd_dow]
                    remaining_sd += (daily_rate * DOW_WEIGHTS[rd_dow] * 0.35) ** 2
                remaining_sd = math.sqrt(remaining_sd) if remaining_sd > 0 else daily_rate * 0.5

                # 用正态CDF估算概率
                if remaining_sd > 0:
                    p_above_lo = 1 - stats.norm.cdf(need_lo, remaining_expected, remaining_sd)
                    p_below_hi = stats.norm.cdf(need_hi, remaining_expected, remaining_sd)
                    day_prob = max(0.005, min(0.95, p_above_lo * p_below_hi))
                else:
                    day_prob = model_prob

            day_delta = day_prob - mp
            day_pct = day_delta / mp * 100 if mp > 0.005 else 0

            daily_forecasts.append({
                "day": d,
                "weekday": dow_name,
                "count_at_day": cumulative_count,
                "predicted_price": day_prob,
                "price_change_pct": day_pct,
                "day_tweets_expected": day_expected,
            })

        # 最终3天后数据
        final = daily_forecasts[-1] if daily_forecasts else None
        if final:
            new_prob = final["predicted_price"]
            delta = new_prob - mp
            pct = final["price_change_pct"]
        else:
            new_prob = mp
            delta = 0
            pct = 0

        if delta > 0.005:
            direction = "up"
        elif delta < -0.005:
            direction = "down"
        else:
            direction = "flat"

        # 投影最终总计
        total_proj = cumulative_count
        if forecast_days < days_remaining:
            for rd in range(forecast_days, int(days_remaining)):
                rd_date = start_date + timedelta(days=days_elapsed + rd)
                total_proj += daily_rate * DOW_WEIGHTS[rd_date.weekday()]

        # 生成解释
        weekend_days = sum(1 for f in daily_forecasts if f["weekday"] in ("Sat", "Sun"))
        reason_parts = [f"日均{daily_rate:.0f}"]
        if weekend_days:
            reason_parts.append(f"含{weekend_days}天周末(发帖-30%)")
        reason_parts.append(f"预计{forecast_days}天后累计{cumulative_count:.0f}")
        if lo <= total_proj <= (hi if hi != float('inf') else 9999):
            reason_parts.append(f"趋势指向{lo}-{hi if hi != float('inf') else ''}内")
        elif total_proj < lo:
            reason_parts.append(f"趋势偏低({total_proj:.0f}<{lo})")
        else:
            reason_parts.append(f"趋势偏高({total_proj:.0f}>{hi if hi != float('inf') else lo+19})")

        return {
            "dir": direction,
            "delta": delta,
            "delta_pct": pct,
            "new_price": new_prob,
            "current_price": mp,
            "reason": "，".join(reason_parts),
            "count_after_3d": cumulative_count,
            "projected_total": total_proj,
            "daily_forecasts": daily_forecasts,
        }

    def _explain_movement(self, direction, delta, pct, daily_rate, new_count,
                          lo, hi, proj_total, days_remaining, forecast_days,
                          start_date, days_elapsed):
        """生成价格变动解释"""
        # 检查未来3天是否有周末
        weekends = 0
        for d in range(int(math.ceil(forecast_days))):
            day_date = start_date + timedelta(days=days_elapsed + d)
            if day_date.weekday() >= 5:
                weekends += 1

        hi_str = str(hi) if hi != float('inf') else "∞"
        range_str = f"{lo}-{hi_str}"

        if direction == "up":
            parts = [f"当前日均{daily_rate:.0f}条"]
            if lo <= proj_total <= (hi if hi != float('inf') else 9999):
                parts.append(f"预期总计{proj_total:.0f}落在{range_str}内")
            parts.append(f"3天后预计涨{abs(pct):.0f}%至{(delta+self._placeholder_mp)*100:.1f}c")
            if weekends:
                parts.append(f"(含{weekends}天周末，发帖减少)")
            return "，".join(parts)
        elif direction == "down":
            parts = [f"当前日均{daily_rate:.0f}条"]
            if proj_total < lo:
                parts.append(f"预期总计{proj_total:.0f}低于{lo}")
            elif hi != float('inf') and proj_total > hi:
                parts.append(f"预期总计{proj_total:.0f}高于{hi}")
            parts.append(f"3天后预计跌{abs(pct):.0f}%")
            if weekends:
                parts.append(f"(含{weekends}天周末)")
            return "，".join(parts)
        else:
            return f"价格基本持平，日均{daily_rate:.0f}条"

    _placeholder_mp = 0  # will be set per-option

    def _signal(self, edge, model_prob, mp, ev_yes, ev_no):
        """交易信号"""
        if edge > 0.06 and model_prob > 0.04 and ev_yes > 0.3:
            return "STRONG_BUY_YES"
        elif edge > 0.03 and model_prob > 0.02 and ev_yes > 0.15:
            return "BUY_YES"
        elif edge < -0.06 and mp > 0.04 and ev_no > 0.3:
            return "STRONG_BUY_NO"
        elif edge < -0.03 and mp > 0.02 and ev_no > 0.15:
            return "BUY_NO"
        elif abs(edge) < 0.015:
            return "FAIR"
        else:
            return "WEAK"

    def format_report(self, market_name, sim_totals, meta, options_analysis,
                      current_count, days_elapsed, total_days):
        """格式化Markdown报告"""
        days_remaining = total_days - days_elapsed
        daily_rate = current_count / days_elapsed if days_elapsed > 0 else 0

        lines = []
        lines.append(f"\n## 市场: {market_name}")
        lines.append(f"\n**实时数据:**")
        lines.append(f"- 当前计数: **{current_count}** tweets")
        lines.append(f"- 已过/总天数: {days_elapsed:.1f} / {total_days}")
        lines.append(f"- 当前日均率: **{daily_rate:.1f}** tweets/day")
        lines.append(f"- 政权后验: H={meta['regime_posterior']['H']:.0%} M={meta['regime_posterior']['M']:.0%} L={meta['regime_posterior']['L']:.0%}")
        lines.append(f"- 趋势斜率: {self.trend_slope:+.1f}/week")
        lines.append(f"- 振荡模式: {'高低交替(近4周)' if self.recent_alternating else '非交替'}")

        # 统计
        mean = np.mean(sim_totals)
        median = np.median(sim_totals)
        std = np.std(sim_totals)
        lines.append(f"\n**模型预测分布:**")
        lines.append(f"- 均值: **{mean:.0f}** | 中位数: **{median:.0f}** | SD: {std:.0f}")
        lines.append(f"- 90% CI: [{np.percentile(sim_totals,5):.0f}, {np.percentile(sim_totals,95):.0f}]")
        lines.append(f"- 50% CI: [{np.percentile(sim_totals,25):.0f}, {np.percentile(sim_totals,75):.0f}]")

        # 市场隐含均值 vs 模型均值
        market_implied = sum(
            (o['lo'] + min(o['hi'], o['lo']+19)) / 2 * o['market_price']
            for o in options_analysis if o['market_price'] > 0.005
        )
        lines.append(f"- 市场隐含均值: ~{market_implied:.0f} | 模型均值: ~{mean:.0f} | 差异: {mean - market_implied:+.0f}")

        # 逐选项表格
        lines.append(f"\n### 逐选项深度分析")
        lines.append("")
        lines.append("| 区间 | 市场价 | 模型概率 | Edge | EV(YES) | EV(NO) | 3天后价格 | 涨跌 | 信号 |")
        lines.append("|------|--------|----------|------|---------|--------|-----------|------|------|")

        for o in options_analysis:
            if o['market_price'] < 0.003 and abs(o['edge']) < 0.01:
                continue

            mp = f"{o['market_price']*100:.1f}%"
            mdl = f"{o['model_prob']*100:.1f}%"
            edg = f"{o['edge']*100:+.1f}%"
            evy = f"{o['ev_yes']*100:+.0f}%" if abs(o['ev_yes']) < 50 else ("极高" if o['ev_yes'] > 0 else "极低")
            evn = f"{o['ev_no']*100:+.0f}%" if abs(o['ev_no']) < 50 else ("极高" if o['ev_no'] > 0 else "极低")

            p3 = o['price_3d']
            np_str = f"{p3['new_price']*100:.1f}%"
            if p3['dir'] == 'up':
                chg = f"+{p3['delta_pct']:.0f}%"
            elif p3['dir'] == 'down':
                chg = f"{p3['delta_pct']:.0f}%"
            else:
                chg = "~0%"

            sig_map = {
                "STRONG_BUY_YES": "**BUY YES**",
                "BUY_YES": "买YES",
                "STRONG_BUY_NO": "**BUY NO**",
                "BUY_NO": "买NO",
                "FAIR": "公平",
                "WEAK": "弱edge",
            }
            sig = sig_map.get(o['signal'], o['signal'])

            lines.append(f"| {o['range']} | {mp} | {mdl} | {edg} | {evy} | {evn} | {np_str} | {chg} | {sig} |")

        # 重点机会详解
        lines.append("\n### 具体交易机会")
        actionable = [o for o in options_analysis
                     if o['signal'] in ('STRONG_BUY_YES', 'BUY_YES', 'STRONG_BUY_NO', 'BUY_NO')]

        if not actionable:
            # 找edge最大的几个
            by_edge = sorted(options_analysis, key=lambda x: abs(x['edge']), reverse=True)
            actionable = by_edge[:5]

        for o in sorted(actionable, key=lambda x: abs(x['edge']), reverse=True)[:6]:
            p3 = o['price_3d']
            lines.append(f"\n#### {o['range']} tweets")
            lines.append(f"- 市场定价: **{o['market_price']*100:.1f}c** | 模型: **{o['model_prob']*100:.1f}%** | Edge: **{o['edge']*100:+.1f}%**")

            if o['edge'] > 0:
                lines.append(f"- 操作: **Buy YES @ {o['market_price']*100:.1f}c**")
                lines.append(f"- 期望回报: {o['ev_yes']*100:+.0f}%")
                if o['kelly_yes'] > 0:
                    lines.append(f"- Kelly仓位: {o['kelly_yes']*100:.1f}% (建议1/4 Kelly = **{o['kelly_yes']*25:.1f}%**)")
            else:
                lines.append(f"- 操作: **Buy NO @ {(1-o['market_price'])*100:.1f}c**")
                lines.append(f"- 期望回报: {o['ev_no']*100:+.0f}%")
                if o['kelly_no'] > 0:
                    lines.append(f"- Kelly仓位: {o['kelly_no']*100:.1f}% (建议1/4 Kelly = **{o['kelly_no']*25:.1f}%**)")

            lines.append(f"- 分析: {p3['reason']}")

            # 逐天价格走势
            if p3.get('daily_forecasts'):
                lines.append(f"- 逐天价格预测:")
                lines.append(f"  | 天 | 星期 | 预期累计 | 预测价格 | 涨跌 |")
                lines.append(f"  |----|------|---------|---------|------|")
                lines.append(f"  | 今天 | - | {current_count} | {o['market_price']*100:.1f}c | - |")
                for df in p3['daily_forecasts']:
                    chg = f"{df['price_change_pct']:+.0f}%" if abs(df['price_change_pct']) > 1 else "~0%"
                    lines.append(f"  | +{df['day']} | {df['weekday']} | {df['count_at_day']:.0f} | {df['predicted_price']*100:.1f}c | {chg} |")

            # 场景分析
            if o['market_price'] > 0.01:
                lines.append(f"- 场景分析:")
                hi_val = o['hi'] if o['hi'] != float('inf') else o['lo'] + 20
                for scenario, factor, label in [
                    ("爆发", 1.25, "日均+25%"),
                    ("偏高", 1.10, "日均+10%"),
                    ("持平", 1.0, "日均不变"),
                    ("偏低", 0.85, "日均-15%"),
                    ("骤降", 0.65, "日均-35%"),
                ]:
                    proj = current_count + daily_rate * factor * days_remaining
                    hit = "命中" if o['lo'] <= proj <= hi_val else "未中"
                    lines.append(f"  - {label} → 总计~{proj:.0f}: **{hit}** {'(目标{}-{})'.format(o['lo'], hi_val)}")

        return "\n".join(lines)


def run_analysis(current_count=122, days_elapsed=3.0):
    """运行完整分析"""
    predictor = ElonPredictor()

    start_date = datetime(2026, 3, 3, 12, 0)
    total_days = 7

    # 市场定价 (Polymarket Mar 3-10 实时数据 2026-03-06 更新)
    options = {
        (100, 119): 0.0005, (120, 139): 0.0005, (140, 159): 0.0015,
        (160, 179): 0.0055, (180, 199): 0.026, (200, 219): 0.0805,
        (220, 239): 0.139, (240, 259): 0.185, (260, 279): 0.185,
        (280, 299): 0.145, (300, 319): 0.0995, (320, 339): 0.065,
        (340, 359): 0.0325, (360, 379): 0.0175, (380, 399): 0.0095,
        (400, 419): 0.0065, (420, 439): 0.0055, (440, 459): 0.0025,
        (460, 479): 0.0025, (480, 499): 0.0025,
        (500, float('inf')): 0.0025,
    }

    # 模拟
    sim_totals, meta = predictor.predict_week(
        current_count=current_count,
        days_elapsed=days_elapsed,
        start_date=start_date,
        total_days=total_days,
    )

    # Set placeholder for explain function
    daily_rate = current_count / days_elapsed if days_elapsed > 0 else meta["base_rate_mean"]

    # 分析每个选项
    for (lo, hi), mp in options.items():
        predictor._placeholder_mp = mp

    opt_analysis = predictor.analyze_options(
        options, sim_totals, meta,
        current_count, days_elapsed, total_days, start_date
    )

    # 修正explain中的placeholder问题
    for o in opt_analysis:
        p3 = o['price_3d']
        if p3['dir'] == 'up':
            p3['reason'] = p3['reason'].replace(
                f"{(p3['delta']+0)*100:.1f}c",
                f"{p3['new_price']*100:.1f}c"
            )

    report = predictor.format_report(
        "Elon Musk # tweets March 3-10, 2026",
        sim_totals, meta, opt_analysis,
        current_count, days_elapsed, total_days
    )

    # 未来周度预测 (考虑振荡交替)
    report += "\n\n## 未来周度预测\n"
    future_weeks = [
        ("Mar 10-17 (W13)", datetime(2026, 3, 10, 12, 0), 1),
        ("Mar 17-24 (W14)", datetime(2026, 3, 17, 12, 0), 2),
        ("Mar 24-31 (W15)", datetime(2026, 3, 24, 12, 0), 3),
    ]

    # 当前周(W12)的政权估算用于推断后续
    current_regime_seq = list(predictor.regimes)  # [..., H, L]

    for name, start, weeks_ahead in future_weeks:
        # 根据振荡模式预测政权
        if predictor.recent_alternating:
            # 交替模式: L->H->L->H...
            # 上周是L(W11), 本周W12不确定, 下周取决于本周
            expected_regimes = []
            last = current_regime_seq[-1]  # "L"
            for w in range(weeks_ahead + 1):
                if last == "L":
                    next_r = "H"
                elif last == "H":
                    next_r = "L"
                else:
                    next_r = "H" if w % 2 == 0 else "L"
                expected_regimes.append(next_r)
                last = next_r
            expected_regime = expected_regimes[-1]
        else:
            expected_regime = "M"

        # 模拟该周
        sims_h, _ = predictor.predict_week(0, 0, start, 7, n_sim=20000)
        # 用政权特定参数调整
        rp = predictor.regime_params.get(expected_regime, predictor.regime_params["M"])
        regime_sims = np.random.normal(rp["mean"] * 7, rp["std"] * 7 * 0.6, 20000)
        regime_sims = np.maximum(regime_sims, 80)
        # 混合: 70%政权模型 + 30%通用模型
        mixed = 0.7 * regime_sims + 0.3 * sims_h

        mean = np.mean(mixed)
        p5, p25, p75, p95 = [np.percentile(mixed, p) for p in [5, 25, 75, 95]]

        report += f"\n### {name}\n"
        report += f"- 预测政权: **{expected_regime}** ({'高活跃' if expected_regime == 'H' else '低活跃' if expected_regime == 'L' else '中等'})\n"
        report += f"- 均值: **{mean:.0f}** | 中位数: {np.median(mixed):.0f}\n"
        report += f"- 90% CI: [{p5:.0f}, {p95:.0f}] | 50% CI: [{p25:.0f}, {p75:.0f}]\n"

        # Top3区间
        bins = list(range(0, 600, 20))
        top_bins = []
        for i in range(len(bins) - 1):
            prob = np.mean((mixed >= bins[i]) & (mixed < bins[i+1]))
            if prob > 0.03:
                top_bins.append((f"{bins[i]}-{bins[i+1]-1}", prob))
        top_bins.sort(key=lambda x: x[1], reverse=True)
        report += f"- Top区间: {', '.join(f'{k}({v:.0%})' for k, v in top_bins[:4])}\n"

        # 策略建议
        if expected_regime == "H":
            report += f"- 策略: 市场开盘时如果定价偏低(均值<{mean:.0f})，买高端区间YES\n"
        elif expected_regime == "L":
            report += f"- 策略: 市场开盘时如果定价偏高(均值>{mean:.0f})，买低端区间YES\n"
        else:
            report += f"- 策略: 等周中数据出来再决定方向\n"

    return report, opt_analysis, sim_totals, meta


def main():
    parser = argparse.ArgumentParser(description='Elon Tweet Predictor v2')
    parser.add_argument('--current-count', type=int, default=122)
    parser.add_argument('--days-elapsed', type=float, default=3.0)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    report, opt_analysis, sim_totals, meta = run_analysis(
        args.current_count, args.days_elapsed
    )

    if args.json:
        output = {
            "stats": {
                "mean": float(np.mean(sim_totals)),
                "std": float(np.std(sim_totals)),
                "p5": float(np.percentile(sim_totals, 5)),
                "p95": float(np.percentile(sim_totals, 95)),
            },
            "regime": meta["regime_posterior"],
            "options": [{k: v for k, v in o.items()} for o in opt_analysis],
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        print(report)


if __name__ == "__main__":
    main()
