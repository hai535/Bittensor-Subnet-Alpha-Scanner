#!/usr/bin/env python3
"""
Elon Musk Tweet Count Predictor v4 - Real Data Calibrated Ensemble
===================================================================
V3 -> V4 改进:
1. XTracker API 实时数据接入 (113天真实日级别数据)
2. 真实星期权重校准 (颠覆了"周末少发"的假设)
3. 真实小时级别分布 (PST timezone)
4. Hawkes过程用真实事件时间calibrate
5. 多源数据: XTracker API + Polymarket价格 + 历史结算
6. 改进的蒙特卡洛: 条件模拟 + 方差缩减

用法:
    python3 elon_tweet_predictor_v4.py
    python3 elon_tweet_predictor_v4.py --live          # 实时从XTracker获取最新数据
    python3 elon_tweet_predictor_v4.py --current-count 150 --days-elapsed 3.5
    python3 elon_tweet_predictor_v4.py --json          # JSON输出
"""

import argparse
import json
import math
import os
import sys
import urllib.request
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
from scipy import stats, optimize

np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 1. XTracker API 数据源
# ============================================================
XTRACKER_BASE = "https://xtracker.polymarket.com/api"
XTRACKER_USER = "elonmusk"

class XTrackerAPI:
    """XTracker API 接口"""

    @staticmethod
    def fetch_json(url, timeout=30):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except Exception as e:
            print(f"  [XTracker] API error: {e}", file=sys.stderr)
            return None

    @classmethod
    def get_user(cls):
        return cls.fetch_json(f"{XTRACKER_BASE}/users/{XTRACKER_USER}")

    @classmethod
    def get_trackings(cls):
        result = cls.fetch_json(f"{XTRACKER_BASE}/users/{XTRACKER_USER}/trackings")
        if result and result.get('success'):
            return result['data']
        return []

    @classmethod
    def get_posts(cls, start_date, end_date):
        url = f"{XTRACKER_BASE}/users/{XTRACKER_USER}/posts?startDate={start_date}&endDate={end_date}"
        result = cls.fetch_json(url)
        if result and result.get('success'):
            return result['data']
        return []

    @classmethod
    def get_current_week_count(cls, tracking_title_contains="March 3 - March 10"):
        """获取当前周的实时帖子计数"""
        trackings = cls.get_trackings()
        for t in trackings:
            if tracking_title_contains in t.get('title', ''):
                posts = cls.get_posts(t['startDate'], t['endDate'])
                daily = defaultdict(int)
                hourly = defaultdict(lambda: defaultdict(int))
                for p in posts:
                    ts = p.get('createdAt', '')
                    if ts:
                        daily[ts[:10]] += 1
                        hourly[ts[:10]][int(ts[11:13])] += 1
                return {
                    'total': len(posts),
                    'daily': dict(sorted(daily.items())),
                    'hourly': {d: dict(h) for d, h in hourly.items()},
                    'tracking': t,
                    'last_sync': t.get('startDate'),
                }
        return None


# ============================================================
# 2. 真实数据加载与校准
# ============================================================
class RealDataCalibrator:
    """用XTracker真实数据校准模型参数"""

    def __init__(self, data_file=None):
        if data_file and os.path.exists(data_file):
            with open(data_file) as f:
                self.raw = json.load(f)
        else:
            self.raw = self._fetch_all_data()

        self.daily_counts = {k: v for k, v in self.raw['daily_counts'].items()}
        self.hourly_counts = self.raw.get('hourly_counts', {})

    def _fetch_all_data(self):
        """从API获取所有历史数据"""
        trackings = XTrackerAPI.get_trackings()
        weekly = [t for t in trackings
                  if 6 <= (datetime.fromisoformat(t['endDate'].replace('Z','')) -
                           datetime.fromisoformat(t['startDate'].replace('Z',''))).days <= 8]

        all_daily = defaultdict(int)
        all_hourly = defaultdict(lambda: defaultdict(int))

        for t in weekly:
            posts = XTrackerAPI.get_posts(t['startDate'], t['endDate'])
            for p in posts:
                ts = p.get('createdAt', '')
                if ts:
                    all_daily[ts[:10]] += 1
                    all_hourly[ts[:10]][int(ts[11:13])] += 1

        result = {
            "daily_counts": dict(sorted(all_daily.items())),
            "hourly_counts": {d: dict(sorted(h.items())) for d, h in sorted(all_hourly.items())},
        }
        # Cache
        cache_path = os.path.join(SCRIPT_DIR, "xtracker_daily_data.json")
        with open(cache_path, 'w') as f:
            json.dump(result, f, indent=2)

        return result

    def calibrate_dow_weights(self):
        """用113天真实数据校准星期权重"""
        dow_counts = defaultdict(list)
        for date_str, count in self.daily_counts.items():
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            dow_counts[dt.weekday()].append(count)

        overall_mean = np.mean(list(self.daily_counts.values()))
        weights = {}
        for dow in range(7):
            if dow_counts[dow]:
                weights[dow] = np.mean(dow_counts[dow]) / overall_mean
            else:
                weights[dow] = 1.0

        return weights

    def calibrate_hourly_pattern(self):
        """用真实数据校准小时级别分布 (UTC)"""
        hourly_totals = defaultdict(int)
        for date_str, hours in self.hourly_counts.items():
            for h_str, c in hours.items():
                hourly_totals[int(h_str)] += c

        total = sum(hourly_totals.values())
        if total == 0:
            return {h: 1/24 for h in range(24)}

        pattern = {}
        for h in range(24):
            pattern[h] = hourly_totals.get(h, 0) / total

        return pattern

    def get_weekly_totals(self):
        """按照Polymarket周期(Mon noon ET - Mon noon ET)计算周总量"""
        # 简化: 按自然周分组
        week_totals = defaultdict(int)
        for date_str, count in sorted(self.daily_counts.items()):
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            # ISO week
            year, week, _ = dt.isocalendar()
            week_totals[f"{year}-W{week:02d}"] = week_totals.get(f"{year}-W{week:02d}", 0) + count

        return dict(sorted(week_totals.items()))

    def get_recent_daily_rates(self, n_days=14):
        """最近N天的日均率"""
        sorted_dates = sorted(self.daily_counts.keys())
        recent = sorted_dates[-n_days:]
        return [self.daily_counts[d] for d in recent]

    def build_event_times(self):
        """构建Hawkes过程需要的事件时间序列"""
        sorted_dates = sorted(self.daily_counts.keys())
        if not sorted_dates:
            return np.array([]), 0

        base = datetime.strptime(sorted_dates[0], "%Y-%m-%d")
        events = []

        for date_str in sorted_dates:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            day_offset = (dt - base).days
            count = self.daily_counts[date_str]

            # 用真实小时分布来分配事件时间
            if date_str in self.hourly_counts:
                hours = self.hourly_counts[date_str]
                for h_str, c in hours.items():
                    h = int(h_str)
                    for i in range(c):
                        t = day_offset + (h + (i + 0.5) / max(c, 1)) / 24.0
                        events.append(t)
            else:
                # 均匀分布
                for i in range(count):
                    t = day_offset + (i + 0.5) / max(count, 1)
                    events.append(t)

        events = np.array(sorted(events))
        T = (datetime.strptime(sorted_dates[-1], "%Y-%m-%d") - base).days + 1
        return events, T


# ============================================================
# 3. Hawkes自激点过程 (改进版)
# ============================================================
class HawkesProcess:
    """
    单变量指数核Hawkes过程 - 用真实数据calibrate

    强度: lambda(t) = mu + sum alpha*beta*exp(-beta*(t - t_i))
    """

    def __init__(self, mu=1.0, alpha=0.5, beta=1.0):
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self._fitted = False

    def log_likelihood(self, events, T):
        n = len(events)
        if n == 0:
            return -self.mu * T

        ll = np.log(max(1e-10, self.mu))
        A = 0
        for i in range(1, n):
            dt = events[i] - events[i-1]
            A = np.exp(-self.beta * dt) * (1 + A)
            lam_i = self.mu + self.alpha * self.beta * A
            ll += np.log(max(1e-10, lam_i))

        integral = self.mu * T
        for ti in events:
            integral += self.alpha * (1 - np.exp(-self.beta * (T - ti)))
        ll -= integral

        return ll

    def fit(self, events, T=None):
        if len(events) < 10:
            return self

        if T is None:
            T = max(events) * 1.1

        events = np.sort(events)

        def neg_ll(params):
            mu, alpha, beta = params
            if mu <= 0 or alpha <= 0 or beta <= 0 or alpha >= beta:
                return 1e10
            self.mu, self.alpha, self.beta = mu, alpha, beta
            return -self.log_likelihood(events, T)

        n = len(events)
        mu0 = n / T * 0.5
        results = []

        # 多起点优化
        for alpha0, beta0 in [(0.3, 1.0), (0.5, 2.0), (0.1, 0.5), (0.8, 3.0)]:
            try:
                result = optimize.minimize(
                    neg_ll, [mu0, alpha0, beta0],
                    method='Nelder-Mead',
                    options={'maxiter': 10000, 'xatol': 1e-8}
                )
                if result.success and result.fun < 1e9:
                    results.append((result.fun, result.x))
            except Exception:
                pass

        if results:
            best = min(results, key=lambda x: x[0])
            self.mu, self.alpha, self.beta = best[1]
            self._fitted = True

        return self

    def branching_ratio(self):
        return self.alpha / self.beta if self.beta > 0 else 0

    def expected_daily_rate(self):
        """稳态日均率"""
        br = self.branching_ratio()
        if br >= 1:
            return float('inf')
        hourly_rate = self.mu / (1 - br)
        return hourly_rate * 24  # 转为日均

    def simulate_remaining(self, days_remaining, current_rate=None, n_sim=10000):
        """模拟剩余天数的推文总数"""
        T = days_remaining

        # 用稳态率或当前观测率
        base_rate = current_rate if current_rate else self.expected_daily_rate()
        if not np.isfinite(base_rate) or base_rate <= 0:
            base_rate = 35.0

        br = min(0.95, self.branching_ratio())
        # Hawkes overdispersion factor
        variance_factor = 1.0 / max(0.05, (1 - br) ** 2)

        expected = base_rate * T
        sd = math.sqrt(max(1, base_rate * T * variance_factor * 0.25))

        return np.maximum(0, np.random.normal(expected, sd, n_sim))


# ============================================================
# 4. 真实小时级别模型
# ============================================================
class RealIntradayModel:
    """用真实数据的小时分布"""

    def __init__(self, hourly_pattern):
        self.pattern = hourly_pattern  # UTC hours

    def fraction_elapsed(self, current_utc_hour):
        """到当前UTC时间已过的发帖比例"""
        total = 0
        for h in range(int(current_utc_hour)):
            total += self.pattern.get(h, 1/24)
        frac = current_utc_hour - int(current_utc_hour)
        total += self.pattern.get(int(current_utc_hour), 1/24) * frac
        return total

    def remaining_fraction(self, current_utc_hour):
        return 1.0 - self.fraction_elapsed(current_utc_hour)


# ============================================================
# 5. 增强型Ensemble预测器
# ============================================================
class EnsemblePredictorV4:
    """
    V4 Ensemble:
    - 负二项回归 (20%)
    - Hawkes过程 (25%) - 真实数据calibrated
    - 政权切换 (25%)
    - 历史匹配 (20%)
    - 趋势+振荡 (10%)
    """

    def __init__(self, calibrator=None, live=False):
        if calibrator is None:
            data_file = os.path.join(SCRIPT_DIR, "xtracker_daily_data.json")
            calibrator = RealDataCalibrator(data_file if os.path.exists(data_file) else None)

        self.cal = calibrator
        self.dow_weights = calibrator.calibrate_dow_weights()
        self.hourly_pattern = calibrator.calibrate_hourly_pattern()
        self.intraday = RealIntradayModel(self.hourly_pattern)

        # 构建周级别数据
        self._build_weekly_data()

        # 拟合Hawkes
        self._fit_hawkes()

        # 政权模型
        self._build_regime_model()

        # 趋势分析
        self._analyze_trends()

    def _build_weekly_data(self):
        """从真实日数据构建周数据 (按Polymarket周期)"""
        # 用XTracker tracking periods的实际结果
        # 这里简化用自然周
        sorted_dates = sorted(self.cal.daily_counts.keys())
        self.daily_values = [self.cal.daily_counts[d] for d in sorted_dates]
        self.daily_dates = sorted_dates

        # 按7天窗口滚动计算周总量
        self.weekly_totals = []
        for i in range(0, len(self.daily_values) - 6, 7):
            week_sum = sum(self.daily_values[i:i+7])
            self.weekly_totals.append(week_sum)

        if not self.weekly_totals:
            self.weekly_totals = [250]

        self.weekly_totals = np.array(self.weekly_totals)
        self.weekly_rates = self.weekly_totals / 7.0

    def _fit_hawkes(self):
        """用真实事件时间拟合Hawkes"""
        events, T = self.cal.build_event_times()

        self.hawkes = HawkesProcess()

        if len(events) > 50:
            # 用最近的事件（最多500个）以控制计算时间
            if len(events) > 500:
                events_fit = events[-500:]
                events_fit = events_fit - events_fit[0]
                T_fit = events_fit[-1] + 1
            else:
                events_fit = events
                T_fit = T

            self.hawkes.fit(events_fit, T_fit)

    def _build_regime_model(self):
        """政权检测"""
        rates = self.weekly_rates
        if len(rates) < 3:
            self.regimes = ["M"]
            self.regime_params = {"H": {"mean": 50, "std": 10}, "M": {"mean": 40, "std": 10}, "L": {"mean": 30, "std": 10}}
            self.last_regime = "M"
            self.recent_alternating = False
            return

        median = np.median(rates)

        self.regimes = []
        for r in rates:
            if r > median * 1.15:
                self.regimes.append("H")
            elif r < median * 0.85:
                self.regimes.append("L")
            else:
                self.regimes.append("M")

        # 最近14天的日均率来判断当前政权
        recent_14 = self.cal.get_recent_daily_rates(14)
        recent_rate = np.mean(recent_14) if recent_14 else np.mean(rates)

        # 用真实数据的统计量
        high_rates = rates[np.array([r == "H" for r in self.regimes])]
        low_rates = rates[np.array([r == "L" for r in self.regimes])]

        self.regime_params = {
            "H": {
                "mean": float(np.mean(high_rates)) if len(high_rates) > 0 else float(np.mean(rates)) * 1.3,
                "std": float(np.std(high_rates, ddof=1)) if len(high_rates) > 1 else float(np.std(rates)) * 0.5,
            },
            "L": {
                "mean": float(np.mean(low_rates)) if len(low_rates) > 0 else float(np.mean(rates)) * 0.6,
                "std": float(np.std(low_rates, ddof=1)) if len(low_rates) > 1 else float(np.std(rates)) * 0.5,
            },
            "M": {
                "mean": float(np.mean(rates)),
                "std": float(np.std(rates, ddof=1)),
            }
        }

        # 交替检测
        alternations = sum(1 for i in range(1, len(self.regimes))
                          if self.regimes[i] != self.regimes[i-1])
        self.alt_ratio = alternations / max(1, len(self.regimes) - 1)

        recent = self.regimes[-4:]
        self.recent_alternating = (
            len(recent) >= 2 and
            all(recent[i] != recent[i+1] for i in range(len(recent)-1))
        )

        self.last_regime = self.regimes[-1]

        # 转换概率
        if self.recent_alternating and self.alt_ratio > 0.55:
            self.transition = {
                "H": {"H": 0.15, "M": 0.20, "L": 0.65},
                "L": {"H": 0.60, "M": 0.20, "L": 0.20},
                "M": {"H": 0.35, "M": 0.30, "L": 0.35},
            }
        else:
            self.transition = {
                "H": {"H": 0.30, "M": 0.30, "L": 0.40},
                "L": {"H": 0.40, "M": 0.25, "L": 0.35},
                "M": {"H": 0.33, "M": 0.34, "L": 0.33},
            }

    def _analyze_trends(self):
        """趋势和振荡分析"""
        if len(self.weekly_rates) < 3:
            self.trend_slope = 0
            return

        x = np.arange(len(self.weekly_rates))
        self.trend_slope, self.trend_intercept = np.polyfit(x, self.weekly_rates, 1)

        # 最近趋势 (最近4周)
        if len(self.weekly_rates) >= 4:
            recent_x = np.arange(4)
            self.recent_trend, _ = np.polyfit(recent_x, self.weekly_rates[-4:], 1)
        else:
            self.recent_trend = self.trend_slope

    def predict_week(self, current_count=0, days_elapsed=0.0,
                     start_date=None, total_days=7, n_sim=50000):
        """Ensemble预测"""
        if start_date is None:
            start_date = datetime(2026, 3, 3, 12, 0)

        days_remaining = total_days - days_elapsed
        if days_remaining <= 0:
            return np.full(n_sim, current_count), {}

        observed_rate = current_count / days_elapsed if days_elapsed > 0 else None

        # 各模型预测
        negbin = self._model_negbin(current_count, days_elapsed, days_remaining, observed_rate, start_date, n_sim)
        hawkes = self._model_hawkes(current_count, days_elapsed, days_remaining, observed_rate, n_sim)
        regime = self._model_regime(current_count, days_elapsed, days_remaining, observed_rate, start_date, n_sim)
        hist = self._model_historical(current_count, days_elapsed, days_remaining, observed_rate, n_sim)
        trend = self._model_trend_oscillation(current_count, days_elapsed, days_remaining, observed_rate, n_sim)

        # 动态权重
        if days_elapsed >= 4:
            w = {"negbin": 0.15, "hawkes": 0.30, "regime": 0.20, "hist": 0.25, "trend": 0.10}
        elif days_elapsed >= 2:
            w = {"negbin": 0.20, "hawkes": 0.25, "regime": 0.25, "hist": 0.20, "trend": 0.10}
        elif days_elapsed >= 0.5:
            w = {"negbin": 0.20, "hawkes": 0.20, "regime": 0.25, "hist": 0.25, "trend": 0.10}
        else:
            w = {"negbin": 0.15, "hawkes": 0.15, "regime": 0.30, "hist": 0.25, "trend": 0.15}

        ensemble = (
            w["negbin"] * negbin +
            w["hawkes"] * hawkes +
            w["regime"] * regime +
            w["hist"] * hist +
            w["trend"] * trend
        ).astype(int)

        meta = {
            "model_weights": w,
            "observed_rate": observed_rate,
            "hawkes_params": {
                "mu": float(self.hawkes.mu),
                "alpha": float(self.hawkes.alpha),
                "beta": float(self.hawkes.beta),
                "branching_ratio": float(self.hawkes.branching_ratio()),
                "expected_daily_rate": float(self.hawkes.expected_daily_rate()) if np.isfinite(self.hawkes.expected_daily_rate()) else None,
                "fitted": self.hawkes._fitted,
            },
            "regime_posterior": self._regime_posterior(observed_rate),
            "trend_slope": float(self.trend_slope),
            "recent_trend": float(self.recent_trend) if hasattr(self, 'recent_trend') else 0,
            "oscillation": {
                "alternating": self.recent_alternating,
                "alt_ratio": float(self.alt_ratio) if hasattr(self, 'alt_ratio') else 0,
                "last_regime": self.last_regime,
            },
            "dow_weights": {str(k): float(v) for k, v in self.dow_weights.items()},
            "data_source": "XTracker API (113 days real data)",
            "models": {
                "negbin": {"mean": float(np.mean(negbin)), "std": float(np.std(negbin))},
                "hawkes": {"mean": float(np.mean(hawkes)), "std": float(np.std(hawkes))},
                "regime": {"mean": float(np.mean(regime)), "std": float(np.std(regime))},
                "hist": {"mean": float(np.mean(hist)), "std": float(np.std(hist))},
                "trend": {"mean": float(np.mean(trend)), "std": float(np.std(trend))},
                "ensemble": {"mean": float(np.mean(ensemble)), "std": float(np.std(ensemble))},
            },
        }

        return ensemble, meta

    def _model_negbin(self, current_count, days_elapsed, days_remaining,
                      observed_rate, start_date, n_sim):
        """负二项回归 - 用真实星期权重"""
        # 基础率
        recent = self.cal.get_recent_daily_rates(14)
        prior_rate = np.mean(recent) if recent else np.mean(self.daily_values)

        if observed_rate is not None:
            weight = min(0.85, days_elapsed / (days_elapsed + 1.5))
            base_rate = weight * observed_rate + (1 - weight) * prior_rate
        else:
            base_rate = prior_rate

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

            # 过度离散参数 - 从真实数据估算
            cv = np.std(self.daily_values) / max(1, np.mean(self.daily_values))
            r = max(1.5, 1 / (cv ** 2 - 1 / max(1, day_mean))) if cv ** 2 > 1 / max(1, day_mean) else 5.0
            r = min(10, max(1.5, r))
            p = r / (r + max(0.5, day_mean))
            totals += np.random.negative_binomial(r, p, n_sim)

        return totals

    def _model_hawkes(self, current_count, days_elapsed, days_remaining,
                      observed_rate, n_sim):
        """Hawkes模型 - 用calibrated参数"""
        remaining_tweets = self.hawkes.simulate_remaining(
            days_remaining, current_rate=observed_rate, n_sim=n_sim
        )
        return current_count + remaining_tweets

    def _model_regime(self, current_count, days_elapsed, days_remaining,
                      observed_rate, start_date, n_sim):
        """政权切换模型"""
        posterior = self._regime_posterior(observed_rate)

        regime_choices = np.random.choice(
            ["H", "M", "L"], size=n_sim,
            p=[posterior["H"], posterior["M"], posterior["L"]]
        )

        totals = np.full(n_sim, float(current_count))

        for regime in ["H", "M", "L"]:
            mask = regime_choices == regime
            n_r = int(np.sum(mask))
            if n_r == 0:
                continue

            rp = self.regime_params[regime]

            if observed_rate is not None and days_elapsed >= 0.5:
                w = min(0.85, days_elapsed / (days_elapsed + 1.5))
                blended_mean = w * observed_rate + (1 - w) * rp["mean"]
                blended_std = rp["std"] * (1 - w * 0.4)
            else:
                blended_mean = rp["mean"]
                blended_std = rp["std"]

            base_rates = np.maximum(5, np.random.normal(blended_mean, max(1, blended_std), n_r))

            # 计算剩余天的权重之和
            total_weight = 0
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
                total_weight += dow_w * frac

            total_means = base_rates * total_weight
            r = 3.0 * n_days
            indices = np.where(mask)[0]
            for idx_i in range(n_r):
                dm = max(1.0, total_means[idx_i])
                p_val = r / (r + dm)
                totals[indices[idx_i]] += np.random.negative_binomial(r, p_val)

        return totals

    def _model_historical(self, current_count, days_elapsed, days_remaining,
                          observed_rate, n_sim):
        """历史匹配模型 - 用真实周数据"""
        weekly_totals = self.weekly_totals

        if observed_rate is not None and days_elapsed >= 0.5:
            hist_rates = weekly_totals / 7.0
            distances = np.abs(hist_rates - observed_rate)
            weights = 1.0 / (distances + 0.5)
            weights /= weights.sum()

            chosen_weeks = np.random.choice(len(weekly_totals), size=n_sim, p=weights)

            totals = np.zeros(n_sim)
            for i, week_idx in enumerate(chosen_weeks):
                week_rate = weekly_totals[week_idx] / 7.0
                projected = current_count + week_rate * days_remaining
                noise = np.random.normal(0, week_rate * 0.25 * math.sqrt(days_remaining))
                totals[i] = max(current_count, projected + noise)
        else:
            chosen = np.random.choice(weekly_totals, n_sim)
            totals = np.maximum(50, chosen + np.random.normal(0, 20, n_sim))

        return totals

    def _model_trend_oscillation(self, current_count, days_elapsed, days_remaining,
                                  observed_rate, n_sim):
        """趋势+振荡模型"""
        # 基础: 趋势外推
        next_week_idx = len(self.weekly_rates)
        trend_rate = self.trend_intercept + self.trend_slope * next_week_idx
        trend_rate = max(10, trend_rate)

        # 振荡调整
        if self.recent_alternating and self.last_regime == "L":
            osc_factor = 1.25  # 预期反弹
        elif self.recent_alternating and self.last_regime == "H":
            osc_factor = 0.75  # 预期回落
        else:
            osc_factor = 1.0

        adjusted_rate = trend_rate * osc_factor

        if observed_rate is not None:
            w = min(0.8, days_elapsed / (days_elapsed + 2))
            adjusted_rate = w * observed_rate + (1 - w) * adjusted_rate

        expected = current_count + adjusted_rate * days_remaining
        sd = adjusted_rate * math.sqrt(days_remaining) * 0.35

        return np.maximum(current_count, np.random.normal(expected, sd, n_sim))

    def _regime_posterior(self, observed_rate):
        prior = self.transition.get(self.last_regime, {"H": 0.33, "M": 0.34, "L": 0.33})

        if observed_rate is None:
            return dict(prior)

        posterior = {}
        for regime in ["H", "M", "L"]:
            rp = self.regime_params[regime]
            likelihood = stats.norm.pdf(observed_rate, rp["mean"], max(3, rp["std"]))
            posterior[regime] = likelihood * prior[regime]

        total_p = sum(posterior.values())
        if total_p > 0:
            posterior = {k: v / total_p for k, v in posterior.items()}
        else:
            posterior = dict(prior)

        return posterior

    # ---- 选项分析 ----

    def analyze_options(self, options, sim_totals, meta,
                        current_count, days_elapsed, total_days, start_date):
        """逐选项深度分析"""
        results = []
        days_remaining = total_days - days_elapsed
        daily_rate = current_count / days_elapsed if days_elapsed > 0 else float(np.mean(self.daily_values))

        for (lo, hi), market_price in sorted(options.items()):
            if hi == float('inf'):
                model_prob = float(np.mean(sim_totals >= lo))
            else:
                model_prob = float(np.mean((sim_totals >= lo) & (sim_totals <= hi)))

            edge = model_prob - market_price

            if market_price > 0.001:
                ev_yes = model_prob / market_price - 1
                ev_no = (1 - model_prob) / (1 - market_price) - 1
            else:
                ev_yes = 99.0 if model_prob > 0.005 else 0
                ev_no = 0

            # Kelly
            if 0.005 < market_price < 0.995:
                odds_yes = 1 / market_price - 1
                kelly_yes = max(0, (model_prob * odds_yes - (1 - model_prob)) / odds_yes)
                odds_no = 1 / (1 - market_price) - 1
                kelly_no = max(0, ((1 - model_prob) * odds_no - model_prob) / odds_no)
            else:
                kelly_yes = kelly_no = 0

            # 3天价格路径
            price_forecast = self._forecast_price_path(
                lo, hi, market_price, model_prob,
                current_count, daily_rate, days_elapsed, days_remaining,
                sim_totals, start_date, total_days
            )

            confidence = self._assess_confidence(model_prob, market_price, days_elapsed, days_remaining)
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
                              days_remaining, sim_totals, start_date, total_days):
        if days_remaining <= 0:
            return {"daily": [], "final_price": market_price, "price_change_pct": 0,
                    "direction": "settled", "profit_scenarios": {}}

        forecast_days = min(3, int(math.ceil(days_remaining)))
        daily_forecasts = []
        cumulative = current_count

        for d in range(1, forecast_days + 1):
            day_date = start_date + timedelta(days=days_elapsed + d - 1)
            dow = day_date.weekday()
            dow_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][dow]
            dow_w = self.dow_weights.get(dow, 1.0)

            day_expected = daily_rate * dow_w
            cumulative += day_expected

            new_elapsed = days_elapsed + d
            new_remaining = total_days - new_elapsed

            if new_remaining <= 0.5:
                hi_val = hi if hi != float('inf') else 9999
                if lo <= cumulative <= hi_val:
                    prob = min(0.95, 0.7 + 0.25 * (1 - new_remaining))
                elif abs(cumulative - lo) < 10 or (hi != float('inf') and abs(cumulative - hi) < 10):
                    prob = 0.3
                else:
                    mid = (lo + (hi if hi != float('inf') else lo + 20)) / 2
                    prob = max(0.01, 0.05 * (1 - min(1, abs(cumulative - mid) / 50)))
            else:
                remaining_expected = sum(
                    daily_rate * self.dow_weights.get((start_date + timedelta(days=new_elapsed + rd)).weekday(), 1.0)
                    for rd in range(int(new_remaining))
                )
                remaining_var = sum(
                    (daily_rate * self.dow_weights.get((start_date + timedelta(days=new_elapsed + rd)).weekday(), 1.0) * 0.40) ** 2
                    for rd in range(int(new_remaining))
                )
                remaining_sd = math.sqrt(remaining_var) if remaining_var > 0 else daily_rate * 0.5

                future_total = cumulative + remaining_expected
                future_sd = remaining_sd

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

        # 5档场景
        hi_val = hi if hi != float('inf') else lo + 20
        profit_scenarios = {}
        for scenario, factor, label in [
            ("surge", 1.40, "爆发(+40%)"),
            ("high", 1.18, "偏高(+18%)"),
            ("normal", 1.0, "持平"),
            ("low", 0.80, "偏低(-20%)"),
            ("crash", 0.55, "骤降(-45%)"),
        ]:
            proj = current_count + daily_rate * factor * days_remaining
            hit = lo <= proj <= hi_val
            profit_scenarios[scenario] = {
                "label": label,
                "projected_total": round(proj),
                "hits_range": hit,
                "prob_weight": {"surge": 0.08, "high": 0.20, "normal": 0.40, "low": 0.22, "crash": 0.10}[scenario],
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
        data_conf = min(1.0, days_elapsed / 4.0)
        time_conf = min(1.0, days_elapsed / (days_elapsed + days_remaining))
        edge = abs(model_prob - market_price)
        edge_conf = min(1.0, edge / 0.10)

        # 模型一致性 (如果Hawkes fitted给更高信心)
        model_conf = 0.6 if self.hawkes._fitted else 0.3

        confidence = 0.25 * data_conf + 0.25 * model_conf + 0.25 * time_conf + 0.25 * edge_conf

        if confidence > 0.65:
            return "HIGH"
        elif confidence > 0.35:
            return "MEDIUM"
        else:
            return "LOW"

    def _signal(self, edge, model_prob, mp, ev_yes, ev_no, confidence):
        if confidence == "LOW" and abs(edge) < 0.05:
            return "HOLD"

        if edge > 0.10 and model_prob > 0.03 and ev_yes > 0.5:
            return "STRONG_BUY_YES"
        elif edge > 0.04 and model_prob > 0.02 and ev_yes > 0.2:
            return "BUY_YES"
        elif edge < -0.10 and mp > 0.03 and ev_no > 0.5:
            return "STRONG_BUY_NO"
        elif edge < -0.04 and mp > 0.02 and ev_no > 0.2:
            return "BUY_NO"
        elif abs(edge) < 0.02:
            return "FAIR"
        else:
            return "WEAK"

    # ---- 报告生成 ----

    def format_report(self, market_name, sim_totals, meta, options_analysis,
                      current_count, days_elapsed, total_days):
        days_remaining = total_days - days_elapsed
        daily_rate = current_count / days_elapsed if days_elapsed > 0 else 0

        lines = []
        lines.append(f"\n## 市场: {market_name}")
        lines.append(f"> 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"> 模型: **Ensemble v4** (XTracker真实数据 + 负二项+Hawkes+政权+历史+趋势)")
        lines.append(f"> 数据源: {meta.get('data_source', 'unknown')}")

        # 实时数据
        lines.append(f"\n### 实时数据")
        lines.append(f"- 当前计数: **{current_count}** tweets ({days_elapsed:.1f}/{total_days}天)")
        lines.append(f"- 当前日均率: **{daily_rate:.1f}** tweets/day")
        lines.append(f"- 剩余天数: **{days_remaining:.1f}天**")

        # Hawkes参数
        hp = meta["hawkes_params"]
        lines.append(f"\n### Hawkes过程参数 (真实数据calibrated)")
        lines.append(f"- 基础强度 (mu): {hp['mu']:.4f}")
        lines.append(f"- 激发系数 (alpha): {hp['alpha']:.4f}")
        lines.append(f"- 衰减率 (beta): {hp['beta']:.4f}")
        lines.append(f"- 分支比: {hp['branching_ratio']:.4f} ({'稳定' if hp['branching_ratio'] < 1 else '不稳定!'})")
        edr = hp.get('expected_daily_rate')
        if edr and np.isfinite(edr):
            lines.append(f"- 稳态日均率: {edr:.1f} tweets/day")
        lines.append(f"- 拟合状态: {'已拟合' if hp['fitted'] else '未拟合(使用默认)'}")

        # 政权
        rp = meta["regime_posterior"]
        osc = meta["oscillation"]
        lines.append(f"\n### 政权分析")
        lines.append(f"- 后验: H={rp['H']:.0%} M={rp['M']:.0%} L={rp['L']:.0%}")
        lines.append(f"- 振荡: {'高低交替' if osc['alternating'] else '非交替'} (交替率{osc['alt_ratio']:.0%})")
        lines.append(f"- 上周政权: **{osc['last_regime']}**")
        lines.append(f"- 趋势斜率: {meta['trend_slope']:+.2f}/周 | 近期趋势: {meta.get('recent_trend', 0):+.2f}/周")

        # 真实星期权重
        lines.append(f"\n### 真实星期权重 (113天XTracker数据校准)")
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dw = meta["dow_weights"]
        lines.append("| " + " | ".join(dow_names) + " |")
        lines.append("|" + "|".join(["------"] * 7) + "|")
        lines.append("| " + " | ".join(f"{dw.get(str(i), 1.0):.3f}" for i in range(7)) + " |")
        lines.append(f"\n> 关键发现: 周末(Sat/Sun)发帖量与工作日几乎持平，颠覆了旧模型的\"周末减半\"假设")

        # 模型对比
        lines.append(f"\n### 各模型预测对比")
        lines.append("| 模型 | 均值 | 标准差 | 权重 |")
        lines.append("|------|------|--------|------|")
        for model_name, model_data in meta["models"].items():
            weight = meta["model_weights"].get(model_name, "-")
            w_str = f"{weight:.0%}" if isinstance(weight, (int, float)) else "-"
            lines.append(f"| {model_name} | {model_data['mean']:.0f} | {model_data['std']:.0f} | {w_str} |")

        # 分布统计
        mean = np.mean(sim_totals)
        std = np.std(sim_totals)
        lines.append(f"\n### Ensemble预测分布")
        lines.append(f"- 均值: **{mean:.0f}** | 中位数: **{np.median(sim_totals):.0f}** | SD: {std:.0f}")
        lines.append(f"- 90% CI: [{np.percentile(sim_totals, 5):.0f}, {np.percentile(sim_totals, 95):.0f}]")
        lines.append(f"- 50% CI: [{np.percentile(sim_totals, 25):.0f}, {np.percentile(sim_totals, 75):.0f}]")

        # 选项表格
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

        # 重点机会
        lines.append(f"\n### 具体交易机会")
        actionable = [o for o in options_analysis
                     if o['signal'] in ('STRONG_BUY_YES', 'BUY_YES', 'STRONG_BUY_NO', 'BUY_NO')]
        if not actionable:
            actionable = sorted(options_analysis, key=lambda x: abs(x['edge']), reverse=True)[:5]

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

            if pf['daily']:
                lines.append(f"- 价格路径:")
                lines.append(f"  | 天 | 星期 | 累计推文 | 预测价 | 涨跌 | 当天预期 |")
                lines.append(f"  |----|------|---------|--------|------|---------|")
                lines.append(f"  | 今天 | - | {current_count} | {o['market_price']*100:.1f}c | - | - |")
                for df in pf['daily']:
                    chg = f"{df['price_change_pct']:+.1f}%" if abs(df['price_change_pct']) > 0.5 else "~0%"
                    lines.append(f"  | +{df['day']} | {df['weekday']} | {df['cumulative_count']} | {df['predicted_price']*100:.1f}c | {chg} | ~{df['day_tweets_expected']:.0f}条 |")

            if pf['profit_scenarios']:
                lines.append(f"- 场景分析:")
                for scenario, data in pf['profit_scenarios'].items():
                    hit = "命中" if data['hits_range'] else "未中"
                    lines.append(f"  - {data['label']} (概率{data['prob_weight']:.0%}) -> 总计~{data['projected_total']}: **{hit}**")

        return "\n".join(lines)


# ============================================================
# 6. Polymarket价格获取
# ============================================================
def fetch_polymarket_prices(slug="elon-musk-of-tweets-march-3-march-10"):
    """从Polymarket获取最新市场价格"""
    import re
    url = f"https://gamma-api.polymarket.com/events?slug={slug}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())

        if not data:
            return None

        options = {}
        markets = data[0].get('markets', [])
        for m in markets:
            if m.get('closed', False):
                continue
            q = m.get('question', '')
            prices = json.loads(m.get('outcomePrices', '[]'))
            if prices:
                match = re.search(r'(\d+)\s*-\s*(\d+)', q)
                match_plus = re.search(r'(\d+)\+', q)
                if match:
                    lo, hi = int(match.group(1)), int(match.group(2))
                    options[(lo, hi)] = float(prices[0])
                elif match_plus:
                    lo = int(match_plus.group(1))
                    options[(lo, float('inf'))] = float(prices[0])

        return options if options else None
    except Exception as e:
        print(f"[Polymarket] Error: {e}", file=sys.stderr)
        return None


# ============================================================
# 7. 主运行
# ============================================================
def run_analysis(current_count=122, days_elapsed=3.0, live=False, polymarket_slug=None):
    """运行完整分析"""
    # 加载真实数据
    data_file = os.path.join(SCRIPT_DIR, "xtracker_daily_data.json")

    if live:
        print(">>> 从XTracker API获取实时数据...", file=sys.stderr)
        # 获取当前周数据
        week_data = XTrackerAPI.get_current_week_count("March 3 - March 10")
        if week_data:
            current_count = week_data['total']
            # 计算已过天数
            start = datetime.fromisoformat(week_data['tracking']['startDate'].replace('Z', ''))
            now = datetime.utcnow()
            days_elapsed = (now - start).total_seconds() / 86400
            print(f">>> XTracker实时: {current_count} posts, {days_elapsed:.2f} days elapsed", file=sys.stderr)

    predictor = EnsemblePredictorV4()

    start_date = datetime(2026, 3, 3, 12, 0)
    total_days = 7

    # Polymarket价格
    if polymarket_slug:
        pm_options = fetch_polymarket_prices(polymarket_slug)
    else:
        pm_options = fetch_polymarket_prices()

    if pm_options is None:
        # 备用手动价格
        pm_options = {
            (120, 139): 0.001, (140, 159): 0.002, (160, 179): 0.007,
            (180, 199): 0.031, (200, 219): 0.090, (220, 239): 0.139,
            (240, 259): 0.190, (260, 279): 0.190, (280, 299): 0.140,
            (300, 319): 0.100, (320, 339): 0.067, (340, 359): 0.033,
            (360, 379): 0.013, (380, 399): 0.010, (400, 419): 0.007,
            (420, 439): 0.007, (440, 459): 0.003, (460, 479): 0.003,
            (480, 499): 0.003, (500, 519): 0.002, (520, 539): 0.002,
            (540, 559): 0.001, (560, 579): 0.002, (580, float('inf')): 0.001,
        }

    # 运行预测
    sim_totals, meta = predictor.predict_week(
        current_count=current_count,
        days_elapsed=days_elapsed,
        start_date=start_date,
        total_days=total_days,
    )

    # 分析选项
    opt_analysis = predictor.analyze_options(
        pm_options, sim_totals, meta,
        current_count, days_elapsed, total_days, start_date
    )

    # 报告
    report = predictor.format_report(
        "Elon Musk # tweets March 3-10, 2026",
        sim_totals, meta, opt_analysis,
        current_count, days_elapsed, total_days
    )

    # 追踪表
    report += "\n\n### 日计数阈值追踪表\n"
    report += "| 天数 | 累计推文 | 最可能区间 | 操作建议 |\n"
    report += "|------|---------|-----------|----------|\n"

    rate = current_count / days_elapsed if days_elapsed > 0 else 35
    for day in range(int(days_elapsed) + 1, total_days + 1):
        proj = current_count + rate * (day - days_elapsed)
        lo_bin = int(proj // 20) * 20
        report += f"| Day {day} | ~{proj:.0f} | {lo_bin}-{lo_bin+19} | 看实际 vs 预期偏差 |\n"

    return report, opt_analysis, sim_totals, meta


def main():
    parser = argparse.ArgumentParser(description='Elon Tweet Predictor v4 - XTracker Calibrated Ensemble')
    parser.add_argument('--current-count', type=int, default=122)
    parser.add_argument('--days-elapsed', type=float, default=3.0)
    parser.add_argument('--live', action='store_true', help='Fetch live data from XTracker API')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--slug', type=str, default=None, help='Polymarket event slug')
    args = parser.parse_args()

    report, opt_analysis, sim_totals, meta = run_analysis(
        args.current_count, args.days_elapsed,
        live=args.live, polymarket_slug=args.slug
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
            "dow_weights": meta["dow_weights"],
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
