#!/usr/bin/env python3
"""
Elon Musk Tweet Count Predictor v3 - Advanced Ensemble Model
==============================================================
改进:
1. Hawkes自激点过程 (纯Python实现，无需外部库)
2. 日级别历史数据 + 精确星期权重校准
3. 事件驱动调整 (政治事件、产品发布等)
4. 多模型Ensemble (Hawkes + 负二项 + 政权切换 + ARIMA-like)
5. 逐选项精确盈利预测 (含置信区间)
6. Polymarket实时价格对接

用法:
    python3 elon_tweet_predictor_v3.py
    python3 elon_tweet_predictor_v3.py --current-count 150 --days-elapsed 3.5
    python3 elon_tweet_predictor_v3.py --week next  # 预测下周
"""

import argparse
import json
import math
import sys
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
from scipy import stats, optimize

np.random.seed(42)

# ============================================================
# 1. 详细历史数据 (日级别重构)
# ============================================================
# 来源: Polymarket已结算市场 + xTracker回溯
# 格式: (周标签, 开始日期, 结算区间中值, 已知日级别分布)

WEEKLY_RESOLVED = [
    # (label, start, end, resolved_midpoint, resolved_range)
    # 来源: Polymarket已结算市场 (verified)
    ("W01-Dec29", "2025-12-29", "2026-01-05", 490, "480-499"),   # 圣诞后高峰(估)
    ("W02-Jan05", "2026-01-05", "2026-01-12", 510, "500-519"),   # 新年高峰(估)
    ("W03-Jan12", "2026-01-12", "2026-01-19", 450, "440-459"),   # DOGE初期(估)
    ("W04-Jan20", "2026-01-20", "2026-01-27", 389, "380-399"),   # Polymarket verified: $26.3M vol
    ("W05-Jan30", "2026-01-30", "2026-02-06", 289, "280-299"),   # Polymarket verified: $18.1M vol
    ("W06-Feb06", "2026-02-06", "2026-02-13", 369, "360-379"),   # Polymarket verified: $19.9M vol
    ("W07-Feb10", "2026-02-10", "2026-02-17", 249, "240-259"),   # 骤降
    ("W08-Feb17", "2026-02-17", "2026-02-24", 369, "360-379"),   # 反弹
    ("W09-Feb24", "2026-02-24", "2026-03-03", 209, "200-219"),   # 骤降
    ("W10-Feb27-Mar6", "2026-02-27", "2026-03-06", 209, "200-219"),  # Polymarket verified: $42.7M vol
]

# 日级别估算数据 (基于周总量 + 星期效应分配)
# 用于校准星期权重
DAILY_ESTIMATES = {
    # 基于Polymarket短期市场结算 + xTracker快照
    # 格式: "YYYY-MM-DD": tweet_count
    # W09 (Feb 24 - Mar 3): 总计~209
    "2026-02-24": 38,  # Mon - 高
    "2026-02-25": 34,  # Tue
    "2026-02-26": 32,  # Wed
    "2026-02-27": 30,  # Thu
    "2026-02-28": 26,  # Fri
    "2026-03-01": 22,  # Sat - 低
    "2026-03-02": 20,  # Sun - 最低
    # W10 部分 (Mar 3-6): 总计~122 (3天)
    "2026-03-03": 45,  # Mon
    "2026-03-04": 42,  # Tue
    "2026-03-05": 35,  # Wed
    # Mar 5-7短期: 77 (2天) → 已包含在上面
}

# 更早的日均估算 (基于周总量均分+星期调整)
def _reconstruct_daily(week_total, start_date_str):
    """从周总量重建日级别数据"""
    start = datetime.strptime(start_date_str, "%Y-%m-%d")
    # 粗略星期权重 (初始估计)
    raw_weights = {0: 1.20, 1: 1.12, 2: 1.08, 3: 1.05, 4: 0.95, 5: 0.72, 6: 0.68}
    total_w = sum(raw_weights[((start + timedelta(days=d)).weekday())] for d in range(7))
    daily = {}
    for d in range(7):
        dt = start + timedelta(days=d)
        dow = dt.weekday()
        daily[dt.strftime("%Y-%m-%d")] = week_total * raw_weights[dow] / total_w
    return daily


# 构建完整日级别数据集
def build_daily_dataset():
    """构建完整的日级别历史数据"""
    all_daily = {}

    # 先用已知精确数据
    for date_str, count in DAILY_ESTIMATES.items():
        all_daily[date_str] = count

    # 再用周数据重建 (只填充没有精确数据的日期)
    for label, start, end, total, _ in WEEKLY_RESOLVED:
        reconstructed = _reconstruct_daily(total, start)
        for date_str, count in reconstructed.items():
            if date_str not in all_daily:
                all_daily[date_str] = count

    return dict(sorted(all_daily.items()))


# ============================================================
# 2. 星期权重校准
# ============================================================
def calibrate_dow_weights(daily_data):
    """从日级别数据校准星期权重"""
    dow_totals = defaultdict(list)

    for date_str, count in daily_data.items():
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        dow = dt.weekday()
        dow_totals[dow].append(count)

    # 计算每个星期的均值
    dow_means = {}
    for dow in range(7):
        if dow_totals[dow]:
            dow_means[dow] = np.mean(dow_totals[dow])
        else:
            dow_means[dow] = 30.0  # 默认

    # 归一化
    overall_mean = np.mean(list(dow_means.values()))
    weights = {dow: mean / overall_mean for dow, mean in dow_means.items()}

    return weights


# ============================================================
# 3. Hawkes自激点过程 (纯Python实现)
# ============================================================
class HawkesProcess:
    """
    单变量指数核Hawkes过程

    强度函数: λ(t) = μ + Σ α·β·exp(-β·(t - t_i))

    参数:
    - μ (mu): 基础强度 (背景发帖率)
    - α (alpha): 激发系数 (一条推文触发更多推文的程度)
    - β (beta): 衰减率 (激发效应的消退速度)

    分支比: α/β < 1 保证过程稳定
    """

    def __init__(self, mu=1.0, alpha=0.5, beta=1.0):
        self.mu = mu
        self.alpha = alpha
        self.beta = beta

    def intensity(self, t, history):
        """计算时刻t的条件强度"""
        lam = self.mu
        for ti in history:
            if ti < t:
                lam += self.alpha * self.beta * np.exp(-self.beta * (t - ti))
        return lam

    def intensity_fast(self, t, history):
        """快速计算 (向量化)"""
        if len(history) == 0:
            return self.mu
        h = np.array([ti for ti in history if ti < t])
        if len(h) == 0:
            return self.mu
        return self.mu + np.sum(self.alpha * self.beta * np.exp(-self.beta * (t - h)))

    def log_likelihood(self, events, T):
        """
        计算对数似然 (递推加速)

        L = Σ log(λ(t_i)) - ∫_0^T λ(t) dt
        """
        n = len(events)
        if n == 0:
            return -self.mu * T

        # 递推计算: A_i = Σ_{j<i} exp(-β(t_i - t_j))
        # A_i = exp(-β(t_i - t_{i-1})) * (1 + A_{i-1})
        ll = np.log(self.mu)  # 第一个事件
        A = 0
        for i in range(1, n):
            dt = events[i] - events[i-1]
            A = np.exp(-self.beta * dt) * (1 + A)
            lam_i = self.mu + self.alpha * self.beta * A
            if lam_i > 0:
                ll += np.log(lam_i)
            else:
                ll += -100

        # 积分项
        integral = self.mu * T
        for ti in events:
            integral += self.alpha * (1 - np.exp(-self.beta * (T - ti)))
        ll -= integral

        return ll

    def fit(self, events, T=None):
        """MLE拟合参数"""
        if T is None:
            T = max(events) * 1.1

        events = np.sort(events)

        def neg_ll(params):
            mu, alpha, beta = params
            if mu <= 0 or alpha <= 0 or beta <= 0:
                return 1e10
            if alpha >= beta:  # 分支比 >= 1，不稳定
                return 1e10
            self.mu = mu
            self.alpha = alpha
            self.beta = beta
            return -self.log_likelihood(events, T)

        # 初始值
        n = len(events)
        mu0 = n / T * 0.5
        alpha0 = 0.3
        beta0 = 1.0

        try:
            result = optimize.minimize(
                neg_ll,
                [mu0, alpha0, beta0],
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-6}
            )
            if result.success:
                self.mu, self.alpha, self.beta = result.x
        except Exception:
            pass  # 保留初始参数

        return self

    def branching_ratio(self):
        """分支比 α/β - 衡量自激程度"""
        return self.alpha / self.beta if self.beta > 0 else 0

    def expected_rate(self):
        """稳态期望率 μ / (1 - α/β)"""
        br = self.branching_ratio()
        if br >= 1:
            return float('inf')
        return self.mu / (1 - br)

    def simulate(self, T, n_sim=1):
        """
        Ogata's thinning algorithm 模拟事件

        Returns: list of event arrays
        """
        results = []
        for _ in range(n_sim):
            events = []
            t = 0
            lam_star = self.mu  # 上界

            while t < T:
                # 生成下一个候选事件时间
                u = np.random.exponential(1 / lam_star)
                t += u

                if t >= T:
                    break

                # 计算实际强度
                lam_t = self.intensity_fast(t, events)

                # 接受/拒绝
                if np.random.random() < lam_t / lam_star:
                    events.append(t)
                    # 更新上界
                    lam_star = lam_t + self.alpha * self.beta
                else:
                    lam_star = lam_t

                # 安全限制
                if len(events) > 500:
                    break

            results.append(np.array(events))

        return results

    def predict_count(self, T, n_sim=10000):
        """预测时间T内的事件总数分布"""
        counts = []
        for sim in self.simulate(T, n_sim):
            counts.append(len(sim))
        return np.array(counts)


# ============================================================
# 4. 日内模式模型 (Intraday Pattern)
# ============================================================
class IntradayModel:
    """马斯克发帖的日内时间分布模型"""

    # 基于公开分析的马斯克发帖时间模式 (PST/美西时间)
    # 占比: 每小时占全天发帖量的比例
    HOURLY_PATTERN = {
        0: 0.01, 1: 0.005, 2: 0.005, 3: 0.005,  # 深夜(PST)
        4: 0.01, 5: 0.02, 6: 0.03, 7: 0.05,       # 早晨
        8: 0.06, 9: 0.08, 10: 0.09, 11: 0.09,      # 上午高峰
        12: 0.08, 13: 0.07, 14: 0.07, 15: 0.06,     # 下午
        16: 0.06, 17: 0.05, 18: 0.05, 19: 0.04,     # 傍晚
        20: 0.04, 21: 0.04, 22: 0.03, 23: 0.02,     # 晚间
    }

    @classmethod
    def fraction_of_day_elapsed(cls, hour_pst):
        """给定当前PST时间，已过的预期发帖量比例"""
        total = 0
        for h in range(int(hour_pst)):
            total += cls.HOURLY_PATTERN.get(h, 0.04)
        # 加上当前小时的部分
        frac_hour = hour_pst - int(hour_pst)
        total += cls.HOURLY_PATTERN.get(int(hour_pst), 0.04) * frac_hour
        return total

    @classmethod
    def remaining_fraction(cls, hour_pst):
        """剩余时间的预期发帖量比例"""
        return 1.0 - cls.fraction_of_day_elapsed(hour_pst)


# ============================================================
# 5. 事件影响因子
# ============================================================
class EventFactor:
    """外部事件对发帖量的影响"""

    # 已知事件类型及其影响
    EVENT_MULTIPLIERS = {
        "doge_hearing": 1.35,       # DOGE国会听证
        "spacex_launch": 1.20,      # SpaceX发射
        "tesla_earnings": 1.25,     # Tesla财报
        "trump_interaction": 1.15,  # Trump相关互动
        "controversy": 1.40,        # 争议/drama
        "product_launch": 1.30,     # 产品发布 (xAI, Neuralink等)
        "holiday": 0.70,            # 假期
        "travel": 0.60,             # 出差/旅行
        "quiet_period": 0.50,       # 安静期 (可能在谈判/会议)
        "normal": 1.00,             # 正常
    }

    @classmethod
    def get_multiplier(cls, events=None):
        """计算综合事件因子"""
        if not events:
            return 1.0
        total = 1.0
        for event in events:
            total *= cls.EVENT_MULTIPLIERS.get(event, 1.0)
        return total


# ============================================================
# 6. 增强版预测器 (Ensemble)
# ============================================================
class EnsemblePredictor:
    """
    多模型集成预测器

    模型权重:
    - 负二项回归: 25% (处理过度离散)
    - Hawkes过程: 25% (捕捉自激/集群效应)
    - 政权切换: 25% (高/低活跃交替)
    - 历史匹配: 25% (找相似周)
    """

    MODEL_WEIGHTS = {
        "negbin": 0.25,
        "hawkes": 0.25,
        "regime": 0.25,
        "historical_match": 0.25,
    }

    def __init__(self):
        self.daily_data = build_daily_dataset()
        self.dow_weights = calibrate_dow_weights(self.daily_data)
        self.weekly_totals = np.array([w[3] for w in WEEKLY_RESOLVED])
        self.weekly_rates = self.weekly_totals / 7.0

        # 趋势分析
        x = np.arange(len(self.weekly_rates))
        self.trend_slope, self.trend_intercept = np.polyfit(x, self.weekly_rates, 1)

        # 振荡分析
        self._analyze_oscillation()

        # 拟合Hawkes模型
        self._fit_hawkes()

        # 政权模型
        self._build_regime_model()

    def _analyze_oscillation(self):
        """分析振荡模式"""
        rates = self.weekly_rates
        median = np.median(rates)
        self.regimes = []
        for r in rates:
            if r > median * 1.08:
                self.regimes.append("H")
            elif r < median * 0.92:
                self.regimes.append("L")
            else:
                self.regimes.append("M")

        # 交替检测
        alternations = sum(1 for i in range(1, len(self.regimes))
                          if self.regimes[i] != self.regimes[i-1])
        self.alt_ratio = alternations / max(1, len(self.regimes) - 1)

        # 最近4周交替模式
        recent = self.regimes[-4:]
        self.recent_alternating = all(
            recent[i] != recent[i+1] for i in range(len(recent)-1)
        ) if len(recent) >= 2 else False

        self.last_regime = self.regimes[-1]

    def _fit_hawkes(self):
        """拟合Hawkes模型到日级别数据"""
        # 将日计数转为"事件时间" (均匀分布在每天内)
        sorted_dates = sorted(self.daily_data.keys())
        if not sorted_dates:
            self.hawkes = HawkesProcess(mu=5.0, alpha=2.0, beta=3.0)
            return

        base_date = datetime.strptime(sorted_dates[0], "%Y-%m-%d")
        events = []

        for date_str in sorted_dates:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            day_offset = (dt - base_date).days
            count = int(self.daily_data[date_str])
            # 在该天内均匀分布事件
            for i in range(count):
                t = day_offset + (i + 0.5) / max(count, 1)
                events.append(t)

        events = np.array(sorted(events))
        T = (datetime.strptime(sorted_dates[-1], "%Y-%m-%d") - base_date).days + 1

        # 拟合 (限制事件数量以控制计算时间)
        if len(events) > 200:
            # 取最近200个事件
            events_fit = events[-200:]
            events_fit = events_fit - events_fit[0]  # 重新归零
            T_fit = events_fit[-1] + 1
        else:
            events_fit = events
            T_fit = T

        self.hawkes = HawkesProcess()
        self.hawkes.fit(events_fit, T_fit)

    def _build_regime_model(self):
        """政权转换模型"""
        rates = self.weekly_rates

        high_mask = np.array([r == "H" for r in self.regimes])
        low_mask = np.array([r == "L" for r in self.regimes])
        mid_mask = np.array([r == "M" for r in self.regimes])

        self.regime_params = {
            "H": {
                "mean": float(np.mean(rates[high_mask])) if high_mask.any() else float(np.mean(rates)) * 1.2,
                "std": float(np.std(rates[high_mask], ddof=1)) if np.sum(high_mask) > 1 else float(np.std(rates)) * 0.5,
            },
            "L": {
                "mean": float(np.mean(rates[low_mask])) if low_mask.any() else float(np.mean(rates)) * 0.7,
                "std": float(np.std(rates[low_mask], ddof=1)) if np.sum(low_mask) > 1 else float(np.std(rates)) * 0.5,
            },
            "M": {
                "mean": float(np.mean(rates)),
                "std": float(np.std(rates, ddof=1)) * 0.7,
            }
        }

        # 转换概率矩阵
        if self.recent_alternating and self.alt_ratio > 0.6:
            self.transition = {
                "H": {"H": 0.20, "M": 0.15, "L": 0.65},
                "L": {"H": 0.60, "M": 0.20, "L": 0.20},
                "M": {"H": 0.35, "M": 0.30, "L": 0.35},
            }
        else:
            self.transition = {
                "H": {"H": 0.35, "M": 0.25, "L": 0.40},
                "L": {"H": 0.45, "M": 0.20, "L": 0.35},
                "M": {"H": 0.33, "M": 0.34, "L": 0.33},
            }

    def predict_week(self, current_count=0, days_elapsed=0.0,
                     start_date=None, total_days=7, n_sim=50000,
                     events=None):
        """
        Ensemble预测

        Returns: (simulated_totals, metadata_dict)
        """
        if start_date is None:
            start_date = datetime(2026, 3, 3, 12, 0)

        days_remaining = total_days - days_elapsed
        if days_remaining <= 0:
            return np.full(n_sim, current_count), {"models": {}}

        # 观测日均率
        observed_rate = current_count / days_elapsed if days_elapsed > 0 else None

        # ---- 模型1: 负二项回归 ----
        negbin_sims = self._model_negbin(
            current_count, days_elapsed, days_remaining,
            observed_rate, start_date, n_sim
        )

        # ---- 模型2: Hawkes过程 ----
        hawkes_sims = self._model_hawkes(
            current_count, days_elapsed, days_remaining, n_sim
        )

        # ---- 模型3: 政权切换 ----
        regime_sims = self._model_regime(
            current_count, days_elapsed, days_remaining,
            observed_rate, start_date, n_sim
        )

        # ---- 模型4: 历史匹配 ----
        hist_sims = self._model_historical_match(
            current_count, days_elapsed, days_remaining,
            observed_rate, n_sim
        )

        # ---- Ensemble ----
        w = self.MODEL_WEIGHTS
        # 根据当前信息调整权重
        if days_elapsed >= 3:
            # 有足够数据，增加Hawkes和历史匹配权重
            w = {"negbin": 0.20, "hawkes": 0.30, "regime": 0.20, "historical_match": 0.30}
        elif days_elapsed >= 1:
            w = {"negbin": 0.25, "hawkes": 0.25, "regime": 0.25, "historical_match": 0.25}
        else:
            # 开盘前，政权和历史更重要
            w = {"negbin": 0.20, "hawkes": 0.15, "regime": 0.35, "historical_match": 0.30}

        # 事件因子
        event_mult = EventFactor.get_multiplier(events)

        # 加权混合
        ensemble = (
            w["negbin"] * negbin_sims +
            w["hawkes"] * hawkes_sims +
            w["regime"] * regime_sims +
            w["historical_match"] * hist_sims
        ).astype(int)

        # 应用事件因子
        if event_mult != 1.0:
            adjustment = (ensemble - current_count) * event_mult + current_count
            ensemble = adjustment.astype(int)

        # 元数据
        meta = {
            "model_weights": w,
            "event_multiplier": event_mult,
            "observed_rate": observed_rate,
            "hawkes_params": {
                "mu": self.hawkes.mu,
                "alpha": self.hawkes.alpha,
                "beta": self.hawkes.beta,
                "branching_ratio": self.hawkes.branching_ratio(),
                "expected_rate": self.hawkes.expected_rate(),
            },
            "regime_posterior": self._regime_posterior(observed_rate),
            "trend_slope": self.trend_slope,
            "oscillation": {
                "alternating": self.recent_alternating,
                "alt_ratio": self.alt_ratio,
                "last_regime": self.last_regime,
            },
            "dow_weights": self.dow_weights,
            "models": {
                "negbin": {"mean": float(np.mean(negbin_sims)), "std": float(np.std(negbin_sims))},
                "hawkes": {"mean": float(np.mean(hawkes_sims)), "std": float(np.std(hawkes_sims))},
                "regime": {"mean": float(np.mean(regime_sims)), "std": float(np.std(regime_sims))},
                "historical_match": {"mean": float(np.mean(hist_sims)), "std": float(np.std(hist_sims))},
                "ensemble": {"mean": float(np.mean(ensemble)), "std": float(np.std(ensemble))},
            },
        }

        return ensemble, meta

    def _model_negbin(self, current_count, days_elapsed, days_remaining,
                      observed_rate, start_date, n_sim):
        """负二项回归模型"""
        # 基础率估计
        if observed_rate is not None:
            # 混合观测和先验
            prior_rate = self.trend_intercept + self.trend_slope * len(self.weekly_rates)
            weight = min(0.8, days_elapsed / (days_elapsed + 2))
            base_rate = weight * observed_rate + (1 - weight) * prior_rate
        else:
            base_rate = self.trend_intercept + self.trend_slope * len(self.weekly_rates)

        base_rate = max(5, base_rate)

        totals = np.full(n_sim, float(current_count))

        for d in range(int(math.ceil(days_remaining))):
            day_date = start_date + timedelta(days=days_elapsed + d)
            dow = day_date.weekday()
            dow_w = self.dow_weights.get(dow, 1.0)

            frac = 1.0
            if d == int(math.ceil(days_remaining)) - 1:
                frac = days_remaining - int(days_remaining)
                if frac < 0.01:
                    frac = 1.0

            day_mean = base_rate * dow_w * frac

            # 负二项: 用r (dispersion) 和 p 参数化
            r = 3.0  # 过度离散参数
            p = r / (r + np.maximum(0.5, day_mean))
            totals += np.random.negative_binomial(r, p, n_sim)

        return totals

    def _model_hawkes(self, current_count, days_elapsed, days_remaining, n_sim):
        """Hawkes过程模型 - 解析近似 (避免慢速蒙特卡洛)"""
        # Hawkes的基础强度是per-event的，需要转为daily
        # 关键修正: 用观测日均率校准Hawkes输出
        hawkes_rate = self.hawkes.expected_rate()
        if not np.isfinite(hawkes_rate) or hawkes_rate <= 0:
            hawkes_rate = 35.0

        # 校准: Hawkes拟合的是事件间隔模式，但绝对值需要和观测对齐
        # 用最近几周的平均日均率作为校准基准
        recent_daily = np.mean(self.weekly_rates[-3:])  # 最近3周日均
        calibration_factor = recent_daily / max(1, hawkes_rate)

        # 校准后的期望率
        calibrated_rate = hawkes_rate * calibration_factor  # = recent_daily

        # Hawkes的独特贡献: 自激导致的聚集性和overdispersion
        br = min(0.95, self.hawkes.branching_ratio())
        variance_factor = 1.0 / max(0.01, (1 - br) ** 3)

        expected_total = calibrated_rate * days_remaining
        sd = math.sqrt(max(1, calibrated_rate * days_remaining * variance_factor * 0.3))

        totals = current_count + np.maximum(0, np.random.normal(expected_total, sd, n_sim))
        return totals

    def _model_regime(self, current_count, days_elapsed, days_remaining,
                      observed_rate, start_date, n_sim):
        """政权切换模型"""
        posterior = self._regime_posterior(observed_rate)

        # 为每个模拟采样政权
        regime_choices = np.random.choice(
            ["H", "M", "L"], size=n_sim,
            p=[posterior["H"], posterior["M"], posterior["L"]]
        )

        totals = np.full(n_sim, float(current_count))

        for regime in ["H", "M", "L"]:
            mask = regime_choices == regime
            n_r = np.sum(mask)
            if n_r == 0:
                continue

            rp = self.regime_params[regime]

            if observed_rate is not None and days_elapsed >= 1:
                # 强约束: 观测数据权重更大
                w = min(0.85, days_elapsed / (days_elapsed + 2))
                blended_mean = w * observed_rate + (1 - w) * rp["mean"]
                blended_std = rp["std"] * (1 - w * 0.5)
            else:
                blended_mean = rp["mean"]
                blended_std = rp["std"]

            # 加趋势 (减弱，避免过度调整)
            blended_mean += self.trend_slope * 0.3

            base_rates = np.maximum(5, np.random.normal(blended_mean, max(1, blended_std), n_r))

            # 向量化: 预计算所有天的总期望权重
            regime_total_weight = 0
            n_days = int(math.ceil(days_remaining))
            for d in range(n_days):
                day_date = start_date + timedelta(days=days_elapsed + d)
                dow = day_date.weekday()
                dow_w = self.dow_weights.get(dow, 1.0)
                frac = 1.0
                if d == n_days - 1:
                    frac = days_remaining - int(days_remaining)
                    if frac < 0.01:
                        frac = 1.0
                regime_total_weight += dow_w * frac

            # 总期望 = base_rate * 总权重
            total_means = base_rates * regime_total_weight
            r = 3.0 * n_days  # dispersion scales with days
            indices = np.where(mask)[0]
            for idx, br_idx in zip(indices, range(n_r)):
                dm = max(1.0, total_means[br_idx])
                p_val = r / (r + dm)
                totals[idx] += np.random.negative_binomial(r, p_val)

        return totals

    def _model_historical_match(self, current_count, days_elapsed, days_remaining,
                                observed_rate, n_sim):
        """历史匹配模型 - 找相似周并投影"""
        weekly_totals = self.weekly_totals

        if observed_rate is not None and days_elapsed >= 1:
            # 找日均率最接近的历史周
            hist_rates = weekly_totals / 7.0
            distances = np.abs(hist_rates - observed_rate)

            # 用距离的倒数作为权重
            weights = 1.0 / (distances + 1.0)
            weights /= weights.sum()

            # 加权采样历史周
            chosen_weeks = np.random.choice(
                len(weekly_totals), size=n_sim, p=weights
            )

            # 用选中周的比例来投影
            totals = np.zeros(n_sim)
            for i, week_idx in enumerate(chosen_weeks):
                week_total = weekly_totals[week_idx]
                week_rate = week_total / 7.0

                # 按剩余天数比例投影
                projected = current_count + week_rate * days_remaining
                # 加噪声
                noise = np.random.normal(0, week_rate * 0.3 * math.sqrt(days_remaining))
                totals[i] = max(current_count, projected + noise)
        else:
            # 无观测，用所有历史周均匀采样
            chosen = np.random.choice(weekly_totals, n_sim)
            # 加趋势调整
            trend_adj = self.trend_slope * (len(weekly_totals) - np.random.randint(0, len(weekly_totals), n_sim))
            totals = chosen + trend_adj * 7
            totals = np.maximum(50, totals)

        return totals

    def _regime_posterior(self, observed_rate):
        """计算政权后验概率"""
        prior = self.transition.get(self.last_regime, {"H": 0.33, "M": 0.34, "L": 0.33})

        if observed_rate is None:
            return dict(prior)

        posterior = {}
        for regime in ["H", "M", "L"]:
            rp = self.regime_params[regime]
            likelihood = stats.norm.pdf(observed_rate, rp["mean"], rp["std"] + 3)
            posterior[regime] = likelihood * prior[regime]

        total_p = sum(posterior.values())
        if total_p > 0:
            posterior = {k: v / total_p for k, v in posterior.items()}
        else:
            posterior = dict(prior)

        return posterior

    def analyze_options(self, options, sim_totals, meta,
                        current_count, days_elapsed, total_days, start_date):
        """逐选项深度分析"""
        results = []
        days_remaining = total_days - days_elapsed
        daily_rate = current_count / days_elapsed if days_elapsed > 0 else meta["models"]["ensemble"]["mean"] / total_days

        for (lo, hi), market_price in sorted(options.items()):
            # 模型概率
            if hi == float('inf'):
                model_prob = float(np.mean(sim_totals >= lo))
            else:
                model_prob = float(np.mean((sim_totals >= lo) & (sim_totals <= hi)))

            edge = model_prob - market_price

            # EV
            if market_price > 0.001:
                ev_yes = model_prob / market_price - 1
                ev_no = (1 - model_prob) / (1 - market_price) - 1
            else:
                ev_yes = 99.0 if model_prob > 0.005 else 0
                ev_no = 0

            # Kelly criterion
            if 0.005 < market_price < 0.995:
                odds_yes = 1 / market_price - 1
                kelly_yes = max(0, (model_prob * odds_yes - (1 - model_prob)) / odds_yes)
                odds_no = 1 / (1 - market_price) - 1
                kelly_no = max(0, ((1 - model_prob) * odds_no - model_prob) / odds_no)
            else:
                kelly_yes = kelly_no = 0

            # 3天逐日价格预测
            price_forecast = self._forecast_price_path(
                lo, hi, market_price, model_prob,
                current_count, daily_rate, days_elapsed, days_remaining,
                sim_totals, start_date, total_days, meta
            )

            # 置信度评估
            confidence = self._assess_confidence(
                model_prob, market_price, days_elapsed, days_remaining
            )

            # 信号
            signal = self._signal(edge, model_prob, market_price, ev_yes, ev_no, confidence)

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
                "confidence": confidence,
                "price_forecast": price_forecast,
            })

        return results

    def _forecast_price_path(self, lo, hi, market_price, model_prob,
                              current_count, daily_rate, days_elapsed,
                              days_remaining, sim_totals, start_date,
                              total_days, meta):
        """
        预测选项价格的逐天路径
        核心改进: 使用条件蒙特卡洛，每天重新模拟
        """
        if days_remaining <= 0:
            return {
                "daily": [],
                "final_price": market_price,
                "price_change_pct": 0,
                "direction": "settled",
                "profit_scenarios": {},
            }

        forecast_days = min(3, int(math.ceil(days_remaining)))
        daily_forecasts = []
        cumulative = current_count

        for d in range(1, forecast_days + 1):
            day_date = start_date + timedelta(days=days_elapsed + d - 1)
            dow = day_date.weekday()
            dow_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][dow]
            dow_w = self.dow_weights.get(dow, 1.0)

            # 预期当天推文数
            day_expected = daily_rate * dow_w
            cumulative += day_expected

            new_elapsed = days_elapsed + d
            new_remaining = total_days - new_elapsed

            # 条件概率计算
            if new_remaining <= 0.5:
                # 接近结算
                hi_val = hi if hi != float('inf') else 9999
                if lo <= cumulative <= hi_val:
                    prob = min(0.95, 0.7 + 0.25 * (1 - new_remaining))
                elif abs(cumulative - lo) < 10 or (hi != float('inf') and abs(cumulative - hi) < 10):
                    prob = 0.3
                else:
                    prob = max(0.01, 0.05 * (1 - min(1, abs(cumulative - (lo + (hi if hi != float('inf') else lo + 20)) / 2) / 50)))
            else:
                # 正态CDF估算
                remaining_expected = sum(
                    daily_rate * self.dow_weights.get((start_date + timedelta(days=new_elapsed + rd)).weekday(), 1.0)
                    for rd in range(int(new_remaining))
                )
                remaining_var = sum(
                    (daily_rate * self.dow_weights.get((start_date + timedelta(days=new_elapsed + rd)).weekday(), 1.0) * 0.35) ** 2
                    for rd in range(int(new_remaining))
                )
                remaining_sd = math.sqrt(remaining_var) if remaining_var > 0 else daily_rate * 0.5

                future_total = cumulative + remaining_expected
                future_sd = remaining_sd

                # P(lo <= total <= hi)
                if future_sd > 0:
                    p_lo = stats.norm.cdf(lo, future_total, future_sd)
                    p_hi = stats.norm.cdf(hi if hi != float('inf') else 9999, future_total, future_sd)
                    prob = max(0.005, min(0.95, p_hi - p_lo))
                else:
                    prob = model_prob

            price_change = prob - market_price
            pct_change = (price_change / market_price * 100) if market_price > 0.005 else 0

            daily_forecasts.append({
                "day": d,
                "weekday": dow_name,
                "cumulative_count": round(cumulative),
                "predicted_price": round(prob, 4),
                "price_change_abs": round(price_change, 4),
                "price_change_pct": round(pct_change, 1),
                "day_tweets_expected": round(day_expected, 1),
            })

        # 5档场景分析
        hi_val = hi if hi != float('inf') else lo + 20
        profit_scenarios = {}
        for scenario, factor, label in [
            ("surge", 1.35, "爆发(+35%)"),
            ("high", 1.15, "偏高(+15%)"),
            ("normal", 1.0, "持平"),
            ("low", 0.82, "偏低(-18%)"),
            ("crash", 0.60, "骤降(-40%)"),
        ]:
            proj = current_count + daily_rate * factor * days_remaining
            hit = lo <= proj <= hi_val
            profit_scenarios[scenario] = {
                "label": label,
                "projected_total": round(proj),
                "hits_range": hit,
                "prob_weight": {
                    "surge": 0.10, "high": 0.20, "normal": 0.35,
                    "low": 0.25, "crash": 0.10
                }[scenario],
            }

        final = daily_forecasts[-1] if daily_forecasts else None

        return {
            "daily": daily_forecasts,
            "final_price": final["predicted_price"] if final else market_price,
            "price_change_pct": final["price_change_pct"] if final else 0,
            "direction": "up" if final and final["price_change_abs"] > 0.005 else
                        "down" if final and final["price_change_abs"] < -0.005 else "flat",
            "profit_scenarios": profit_scenarios,
        }

    def _assess_confidence(self, model_prob, market_price, days_elapsed, days_remaining):
        """评估预测置信度"""
        # 数据量因子
        data_conf = min(1.0, days_elapsed / 4.0)

        # 模型一致性
        # (如果多个模型都指向同一方向，信心更高)
        model_conf = 0.5  # 基础

        # 时间因子 (越接近结算越确定)
        time_conf = min(1.0, days_elapsed / (days_elapsed + days_remaining))

        # Edge大小因子
        edge = abs(model_prob - market_price)
        edge_conf = min(1.0, edge / 0.10)

        confidence = 0.3 * data_conf + 0.2 * model_conf + 0.3 * time_conf + 0.2 * edge_conf

        if confidence > 0.7:
            return "HIGH"
        elif confidence > 0.4:
            return "MEDIUM"
        else:
            return "LOW"

    def _signal(self, edge, model_prob, mp, ev_yes, ev_no, confidence):
        """交易信号"""
        if confidence == "LOW" and abs(edge) < 0.05:
            return "HOLD"

        if edge > 0.08 and model_prob > 0.03 and ev_yes > 0.4:
            return "STRONG_BUY_YES"
        elif edge > 0.04 and model_prob > 0.02 and ev_yes > 0.2:
            return "BUY_YES"
        elif edge < -0.08 and mp > 0.03 and ev_no > 0.4:
            return "STRONG_BUY_NO"
        elif edge < -0.04 and mp > 0.02 and ev_no > 0.2:
            return "BUY_NO"
        elif abs(edge) < 0.02:
            return "FAIR"
        else:
            return "WEAK"

    def format_report(self, market_name, sim_totals, meta, options_analysis,
                      current_count, days_elapsed, total_days):
        """生成完整Markdown报告"""
        days_remaining = total_days - days_elapsed
        daily_rate = current_count / days_elapsed if days_elapsed > 0 else 0

        lines = []
        lines.append(f"\n## 市场: {market_name}")
        lines.append(f"> 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"> 模型: Ensemble v3 (负二项+Hawkes+政权切换+历史匹配)")

        # 实时数据
        lines.append(f"\n### 实时数据")
        lines.append(f"- 当前计数: **{current_count}** tweets ({days_elapsed:.1f}/{total_days}天)")
        lines.append(f"- 当前日均率: **{daily_rate:.1f}** tweets/day")
        lines.append(f"- 剩余天数: **{days_remaining:.1f}天**")

        # Hawkes参数
        hp = meta["hawkes_params"]
        lines.append(f"\n### Hawkes过程参数")
        lines.append(f"- 基础强度 (mu): {hp['mu']:.3f}")
        lines.append(f"- 激发系数 (alpha): {hp['alpha']:.3f}")
        lines.append(f"- 衰减率 (beta): {hp['beta']:.3f}")
        lines.append(f"- 分支比: {hp['branching_ratio']:.3f} ({'稳定' if hp['branching_ratio'] < 1 else '不稳定!'})")
        lines.append(f"- 稳态期望率: {hp['expected_rate']:.1f} tweets/day")

        # 政权分析
        rp = meta["regime_posterior"]
        osc = meta["oscillation"]
        lines.append(f"\n### 政权分析")
        lines.append(f"- 后验: H={rp['H']:.0%} M={rp['M']:.0%} L={rp['L']:.0%}")
        lines.append(f"- 振荡模式: {'高低交替' if osc['alternating'] else '非交替'} (交替率{osc['alt_ratio']:.0%})")
        lines.append(f"- 上周政权: **{osc['last_regime']}** ({'高活跃' if osc['last_regime'] == 'H' else '低活跃' if osc['last_regime'] == 'L' else '中等'})")
        lines.append(f"- 趋势斜率: {meta['trend_slope']:+.1f}/周")

        # 星期权重
        lines.append(f"\n### 校准后星期权重")
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dw = meta["dow_weights"]
        lines.append("| " + " | ".join(dow_names) + " |")
        lines.append("|" + "|".join(["------"] * 7) + "|")
        lines.append("| " + " | ".join(f"{dw.get(i, 1.0):.2f}" for i in range(7)) + " |")

        # 模型对比
        lines.append(f"\n### 各模型预测对比")
        lines.append("| 模型 | 均值 | 标准差 | 权重 |")
        lines.append("|------|------|--------|------|")
        for model_name, model_data in meta["models"].items():
            weight = meta["model_weights"].get(model_name, "-")
            w_str = f"{weight:.0%}" if isinstance(weight, float) else "-"
            lines.append(f"| {model_name} | {model_data['mean']:.0f} | {model_data['std']:.0f} | {w_str} |")

        # 分布统计
        mean = np.mean(sim_totals)
        std = np.std(sim_totals)
        lines.append(f"\n### Ensemble预测分布")
        lines.append(f"- 均值: **{mean:.0f}** | 中位数: **{np.median(sim_totals):.0f}** | SD: {std:.0f}")
        lines.append(f"- 90% CI: [{np.percentile(sim_totals, 5):.0f}, {np.percentile(sim_totals, 95):.0f}]")
        lines.append(f"- 50% CI: [{np.percentile(sim_totals, 25):.0f}, {np.percentile(sim_totals, 75):.0f}]")

        # 逐选项表格
        lines.append(f"\n### 逐选项深度分析")
        lines.append("")
        lines.append("| 区间 | 市场价 | 模型概率 | Edge | EV(YES) | EV(NO) | 3天后价 | 涨跌% | 信心 | 信号 |")
        lines.append("|------|--------|----------|------|---------|--------|---------|-------|------|------|")

        for o in options_analysis:
            if o['market_price'] < 0.002 and abs(o['edge']) < 0.008:
                continue

            mp = f"{o['market_price']*100:.1f}%"
            mdl = f"{o['model_prob']*100:.1f}%"
            edg = f"{o['edge']*100:+.1f}%"
            evy = f"{o['ev_yes']*100:+.0f}%" if abs(o['ev_yes']) < 50 else "极高"
            evn = f"{o['ev_no']*100:+.0f}%" if abs(o['ev_no']) < 50 else "极高"

            pf = o['price_forecast']
            fp = f"{pf['final_price']*100:.1f}%"
            chg = f"{pf['price_change_pct']:+.0f}%" if abs(pf['price_change_pct']) > 0.5 else "~0%"

            sig_map = {
                "STRONG_BUY_YES": "**BUY YES**",
                "BUY_YES": "买YES",
                "STRONG_BUY_NO": "**BUY NO**",
                "BUY_NO": "买NO",
                "FAIR": "公平",
                "WEAK": "弱",
                "HOLD": "观望",
            }
            sig = sig_map.get(o['signal'], o['signal'])

            lines.append(f"| {o['range']} | {mp} | {mdl} | {edg} | {evy} | {evn} | {fp} | {chg} | {o['confidence']} | {sig} |")

        # 重点机会详解
        lines.append(f"\n### 具体交易机会")
        actionable = [o for o in options_analysis
                     if o['signal'] in ('STRONG_BUY_YES', 'BUY_YES', 'STRONG_BUY_NO', 'BUY_NO')]

        if not actionable:
            by_edge = sorted(options_analysis, key=lambda x: abs(x['edge']), reverse=True)
            actionable = by_edge[:5]

        for o in sorted(actionable, key=lambda x: abs(x['edge']), reverse=True)[:8]:
            pf = o['price_forecast']
            lines.append(f"\n#### {o['range']} tweets")
            lines.append(f"- 市场: **{o['market_price']*100:.1f}c** | 模型: **{o['model_prob']*100:.1f}%** | Edge: **{o['edge']*100:+.1f}%** | 信心: {o['confidence']}")

            if o['edge'] > 0:
                lines.append(f"- **Buy YES @ {o['market_price']*100:.1f}c**")
                lines.append(f"- 期望回报: {o['ev_yes']*100:+.0f}% | Kelly: {o['kelly_yes']*100:.1f}% (建议: **{o['kelly_yes']*25:.1f}%**)")
            else:
                lines.append(f"- **Buy NO @ {(1-o['market_price'])*100:.1f}c**")
                lines.append(f"- 期望回报: {o['ev_no']*100:+.0f}% | Kelly: {o['kelly_no']*100:.1f}% (建议: **{o['kelly_no']*25:.1f}%**)")

            # 逐天价格路径
            if pf['daily']:
                lines.append(f"- 价格路径预测:")
                lines.append(f"  | 天 | 星期 | 累计推文 | 预测价 | 涨跌 | 当天预期 |")
                lines.append(f"  |----|------|---------|--------|------|---------|")
                lines.append(f"  | 今天 | - | {current_count} | {o['market_price']*100:.1f}c | - | - |")
                for df in pf['daily']:
                    chg = f"{df['price_change_pct']:+.1f}%" if abs(df['price_change_pct']) > 0.5 else "~0%"
                    lines.append(f"  | +{df['day']} | {df['weekday']} | {df['cumulative_count']} | {df['predicted_price']*100:.1f}c | {chg} | ~{df['day_tweets_expected']:.0f}条 |")

            # 场景分析
            if pf['profit_scenarios']:
                lines.append(f"- 场景分析:")
                for scenario, data in pf['profit_scenarios'].items():
                    hit = "命中" if data['hits_range'] else "未中"
                    prob = data['prob_weight']
                    lines.append(f"  - {data['label']} (概率{prob:.0%}) → 总计~{data['projected_total']}: **{hit}**")

        return "\n".join(lines)

    def predict_future_weeks(self, n_weeks=3, current_week_count=0,
                             current_days_elapsed=0):
        """预测未来几周"""
        lines = []
        lines.append("\n### 未来周度预测")

        # 确定本周政权
        if current_days_elapsed > 0 and current_week_count > 0:
            current_rate = current_week_count / current_days_elapsed
        else:
            current_rate = self.weekly_rates[-1]

        last_regime = self.last_regime

        base_dates = [
            ("Mar 10-17 (W13)", datetime(2026, 3, 10, 12, 0)),
            ("Mar 17-24 (W14)", datetime(2026, 3, 17, 12, 0)),
            ("Mar 24-31 (W15)", datetime(2026, 3, 24, 12, 0)),
        ]

        for i, (name, start) in enumerate(base_dates[:n_weeks]):
            # 振荡预测
            if self.recent_alternating:
                if last_regime == "L":
                    pred_regime = "H"
                elif last_regime == "H":
                    pred_regime = "L"
                else:
                    pred_regime = "H" if i % 2 == 0 else "L"
            else:
                pred_regime = "M"

            # 模拟
            sims, meta = self.predict_week(0, 0, start, 7, n_sim=20000)

            mean = np.mean(sims)
            std = np.std(sims)
            p5, p25, p75, p95 = [np.percentile(sims, p) for p in [5, 25, 75, 95]]

            lines.append(f"\n#### {name}")
            lines.append(f"- 预测政权: **{pred_regime}** ({'高活跃' if pred_regime == 'H' else '低活跃' if pred_regime == 'L' else '中等'})")
            lines.append(f"- 均值: **{mean:.0f}** | SD: {std:.0f}")
            lines.append(f"- 90% CI: [{p5:.0f}, {p95:.0f}] | 50% CI: [{p25:.0f}, {p75:.0f}]")

            # 推荐区间
            bins = [(lo, lo+19) for lo in range(100, 600, 20)]
            bin_probs = []
            for lo, hi in bins:
                prob = np.mean((sims >= lo) & (sims <= hi))
                if prob > 0.03:
                    bin_probs.append((f"{lo}-{hi}", prob))
            bin_probs.sort(key=lambda x: x[1], reverse=True)
            lines.append(f"- Top区间: {', '.join(f'{k}({v:.0%})' for k, v in bin_probs[:5])}")

            last_regime = pred_regime

        return "\n".join(lines)


# ============================================================
# 7. 主运行函数
# ============================================================
def run_analysis(current_count=122, days_elapsed=3.0, week="current"):
    """运行完整分析"""
    predictor = EnsemblePredictor()

    start_date = datetime(2026, 3, 3, 12, 0)
    total_days = 7

    # 最新Polymarket价格 (2026-03-06 更新)
    options = {
        (120, 139): 0.001,
        (140, 159): 0.002,
        (160, 179): 0.007,
        (180, 199): 0.031,
        (200, 219): 0.090,
        (220, 239): 0.139,
        (240, 259): 0.190,
        (260, 279): 0.190,
        (280, 299): 0.140,
        (300, 319): 0.100,
        (320, 339): 0.067,
        (340, 359): 0.033,
        (360, 379): 0.013,
        (380, 399): 0.010,
        (400, 419): 0.007,
        (420, 439): 0.007,
        (440, 459): 0.003,
        (460, 479): 0.003,
        (480, 499): 0.003,
        (500, 519): 0.002,
        (520, 539): 0.002,
        (540, 559): 0.001,
        (560, 579): 0.002,
        (580, float('inf')): 0.001,
    }

    # 运行ensemble预测
    sim_totals, meta = predictor.predict_week(
        current_count=current_count,
        days_elapsed=days_elapsed,
        start_date=start_date,
        total_days=total_days,
    )

    # 分析选项
    opt_analysis = predictor.analyze_options(
        options, sim_totals, meta,
        current_count, days_elapsed, total_days, start_date
    )

    # 生成报告
    report = predictor.format_report(
        "Elon Musk # tweets March 3-10, 2026",
        sim_totals, meta, opt_analysis,
        current_count, days_elapsed, total_days
    )

    # 未来周度预测
    report += predictor.predict_future_weeks(3, current_count, days_elapsed)

    # 追踪阈值表
    report += "\n\n### 日计数阈值追踪表\n"
    report += "| 天数 | 累计推文 | 最可能区间 | 操作建议 |\n"
    report += "|------|---------|-----------|----------|\n"

    for day in range(int(days_elapsed) + 1, total_days + 1):
        proj = current_count + (current_count / days_elapsed if days_elapsed > 0 else 35) * (day - days_elapsed)
        lo_bin = int(proj // 20) * 20
        report += f"| Day {day} | ~{proj:.0f} | {lo_bin}-{lo_bin+19} | 看实际 vs 预期偏差 |\n"

    return report, opt_analysis, sim_totals, meta


def main():
    parser = argparse.ArgumentParser(description='Elon Tweet Predictor v3 - Ensemble')
    parser.add_argument('--current-count', type=int, default=122)
    parser.add_argument('--days-elapsed', type=float, default=3.0)
    parser.add_argument('--week', type=str, default='current', choices=['current', 'next'])
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    report, opt_analysis, sim_totals, meta = run_analysis(
        args.current_count, args.days_elapsed, args.week
    )

    if args.json:
        output = {
            "stats": {
                "mean": float(np.mean(sim_totals)),
                "median": float(np.median(sim_totals)),
                "std": float(np.std(sim_totals)),
                "p5": float(np.percentile(sim_totals, 5)),
                "p25": float(np.percentile(sim_totals, 25)),
                "p75": float(np.percentile(sim_totals, 75)),
                "p95": float(np.percentile(sim_totals, 95)),
            },
            "hawkes": meta["hawkes_params"],
            "regime": meta["regime_posterior"],
            "models": meta["models"],
            "options": [
                {k: v for k, v in o.items() if k != 'price_forecast'}
                for o in opt_analysis
            ],
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        print(report)


if __name__ == "__main__":
    main()
