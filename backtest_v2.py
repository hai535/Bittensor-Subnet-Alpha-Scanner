#!/usr/bin/env python3
"""
马斯克推文预测模型 V2 回测 - 大幅改进版
========================================
改进:
1. ARIMA(p,d,q) - 经典时序模型
2. Ridge回归 + 特征工程 - 滞后值/滚动统计/波动率
3. Dampened Trend - 带衰减的趋势外推
4. Regime-Switching - 统计检测高/低波动期
5. Online Learning Ensemble - 按最近表现动态调权
6. 保留表现还行的 EMA 和 Naive 作为基线
"""

import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

CSV_PATH = "/root/.cache/kagglehub/datasets/dadalyndell/elon-musk-tweets-2010-to-2025-march/versions/11/all_musk_posts.csv"

# ============================================================
# 1. 数据加载
# ============================================================
def load_weekly_data():
    df = pd.read_csv(CSV_PATH, usecols=['createdAt'], low_memory=False)
    df['createdAt'] = pd.to_datetime(df['createdAt'], utc=True)
    df = df.sort_values('createdAt')

    # 从2023年1月开始，尽可能多的历史
    cutoff = pd.Timestamp('2023-01-01', tz='UTC')
    recent = df[df['createdAt'] >= cutoff].copy()

    # ET时区, Polymarket周 = Mon 12PM ET -> Mon 12PM ET
    recent['et'] = recent['createdAt'].dt.tz_convert('US/Eastern')
    recent['shifted'] = recent['et'] - pd.Timedelta(hours=12)
    recent['week_start'] = recent['shifted'].dt.to_period('W-MON').apply(lambda x: x.start_time)

    weekly = recent.groupby('week_start').agg(count=('createdAt', 'size')).reset_index()
    weekly['week_start'] = pd.to_datetime(weekly['week_start'])
    weekly = weekly.sort_values('week_start').reset_index(drop=True)

    # 日级别数据
    recent['date'] = recent['et'].dt.date
    daily = recent.groupby('date').size().reset_index(name='count')
    daily['date'] = pd.to_datetime(daily['date'])

    return weekly, daily


# ============================================================
# 2. 特征工程
# ============================================================
def build_features(counts, idx):
    """为第idx周构建特征向量，用idx之前的数据"""
    if idx < 8:
        return None

    history = counts[:idx]
    n = len(history)

    features = {}

    # 滞后特征
    features['lag1'] = history[-1]
    features['lag2'] = history[-2]
    features['lag3'] = history[-3]
    features['lag4'] = history[-4]

    # 滚动统计
    features['ma4'] = np.mean(history[-4:])
    features['ma8'] = np.mean(history[-8:])
    features['std4'] = np.std(history[-4:], ddof=1) if len(history[-4:]) > 1 else 0
    features['std8'] = np.std(history[-8:], ddof=1) if len(history[-8:]) > 1 else 0

    # 变化率
    features['pct_change1'] = (history[-1] - history[-2]) / (history[-2] + 1)
    features['pct_change2'] = (history[-1] - history[-3]) / (history[-3] + 1) if n >= 3 else 0

    # 波动率 (变异系数)
    features['cv4'] = features['std4'] / (features['ma4'] + 1)

    # 趋势 (最近4周线性斜率)
    x = np.arange(min(4, n))
    y = np.array(history[-4:])
    if len(y) > 1:
        slope, _, _, _, _ = stats.linregress(x, y)
        features['slope4'] = slope
    else:
        features['slope4'] = 0

    # 最近值相对长期均值的偏离
    long_mean = np.mean(history) if n > 0 else 0
    features['deviation'] = (history[-1] - long_mean) / (long_mean + 1)

    # Min/Max ratio
    features['min4'] = min(history[-4:])
    features['max4'] = max(history[-4:])

    return features


# ============================================================
# 3. 模型定义
# ============================================================

def model_naive(history):
    """上周=本周"""
    return history[-1]


def model_ema(history, alpha=0.3):
    """指数移动平均, alpha=0.3"""
    ema = history[0]
    for v in history[1:]:
        ema = alpha * v + (1 - alpha) * ema
    return ema


def model_ema_optimized(history):
    """自适应EMA: 根据最近波动调整alpha"""
    if len(history) < 8:
        return model_ema(history, 0.3)

    # 高波动 -> 低alpha(更平滑), 低波动 -> 高alpha(更敏感)
    recent_cv = np.std(history[-8:], ddof=1) / (np.mean(history[-8:]) + 1)

    if recent_cv > 0.3:  # 高波动
        alpha = 0.2
    elif recent_cv > 0.15:  # 中波动
        alpha = 0.35
    else:  # 低波动
        alpha = 0.5

    ema = history[0]
    for v in history[1:]:
        ema = alpha * v + (1 - alpha) * ema
    return ema


def model_arima_simple(history):
    """简单ARIMA(2,0,1): 手写避免依赖statsmodels"""
    data = np.array(history, dtype=float)
    n = len(data)
    if n < 10:
        return np.mean(data[-4:])

    # AR(2): y_t = c + phi1*y_{t-1} + phi2*y_{t-2} + e_t
    # 用最小二乘拟合
    Y = data[2:]
    X = np.column_stack([np.ones(n-2), data[1:-1], data[:-2]])

    try:
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        c, phi1, phi2 = beta

        # 一步预测
        pred = c + phi1 * data[-1] + phi2 * data[-2]

        # MA(1)近似: 用最近的残差修正
        fitted = X @ beta
        residuals = Y - fitted
        theta = 0.3  # MA系数
        ma_correction = theta * residuals[-1]

        pred += ma_correction

        # 合理性检查
        pred = np.clip(pred, np.mean(data[-8:]) * 0.3, np.mean(data[-8:]) * 2.5)
        return pred
    except:
        return np.mean(data[-4:])


def model_dampened_trend(history):
    """带衰减的趋势外推 - 趋势随时间衰减"""
    data = np.array(history)
    n = len(data)
    if n < 4:
        return np.mean(data)

    window = min(8, n)
    y = data[-window:]
    x = np.arange(window)

    slope, intercept, r_val, _, _ = stats.linregress(x, y)

    # 衰减因子: R^2越低, 衰减越多 (趋势不可靠时保守)
    damping = 0.5 * abs(r_val)  # R^2高时衰减少
    pred = intercept + slope * (window + damping)

    # 限制范围
    hist_mean = np.mean(data[-8:])
    pred = np.clip(pred, hist_mean * 0.4, hist_mean * 2.0)
    return pred


def model_regime_switching(history):
    """Regime-switching: 检测高/低波动期, 分别建模"""
    data = np.array(history)
    n = len(data)
    if n < 12:
        return np.mean(data[-4:])

    # 用滚动std检测regime
    window = 4
    rolling_std = []
    for i in range(window, n):
        rolling_std.append(np.std(data[i-window:i], ddof=1))

    if not rolling_std:
        return np.mean(data[-4:])

    median_std = np.median(rolling_std)
    current_std = rolling_std[-1] if rolling_std else median_std

    if current_std > median_std * 1.3:
        # 高波动期 -> 更强的均值回归
        long_mean = np.mean(data[-12:])
        last = data[-1]
        pred = last + 0.4 * (long_mean - last)
    elif current_std < median_std * 0.7:
        # 低波动期 -> 相信趋势
        pred = model_ema(list(data), 0.4)
    else:
        # 正常期 -> 加权平均
        pred = 0.5 * data[-1] + 0.3 * np.mean(data[-4:]) + 0.2 * np.mean(data[-8:])

    pred = np.clip(pred, np.mean(data[-8:]) * 0.3, np.mean(data[-8:]) * 2.5)
    return pred


def model_ridge(history, all_features_cache, idx):
    """Ridge回归 + 特征工程"""
    if idx < 12 or len(all_features_cache) < 8:
        return np.mean(history[-4:])

    # 构建训练数据: 用之前的特征预测下一周
    train_X = []
    train_y = []

    for past_idx in range(8, idx):
        if past_idx in all_features_cache and past_idx < len(history):
            feat = all_features_cache[past_idx]
            feat_vec = [feat[k] for k in sorted(feat.keys())]
            train_X.append(feat_vec)
            train_y.append(history[past_idx])

    if len(train_X) < 5:
        return np.mean(history[-4:])

    X = np.array(train_X)
    y = np.array(train_y)

    # 标准化
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std
    y_mean = y.mean()
    y_centered = y - y_mean

    # Ridge回归: (X^T X + lambda I)^-1 X^T y
    lam = 1.0  # 正则化系数
    try:
        XtX = X_norm.T @ X_norm
        Xty = X_norm.T @ y_centered
        beta = np.linalg.solve(XtX + lam * np.eye(XtX.shape[0]), Xty)

        # 预测当前周
        if idx in all_features_cache:
            curr_feat = all_features_cache[idx]
            curr_vec = np.array([curr_feat[k] for k in sorted(curr_feat.keys())])
            curr_norm = (curr_vec - X_mean) / X_std
            pred = curr_norm @ beta + y_mean
        else:
            return np.mean(history[-4:])

        pred = np.clip(pred, np.mean(history[-8:]) * 0.3, np.mean(history[-8:]) * 2.5)
        return pred
    except:
        return np.mean(history[-4:])


def model_median_ensemble(predictions):
    """中位数集成 - 比加权平均更鲁棒"""
    vals = list(predictions.values())
    return np.median(vals)


def model_online_ensemble(predictions, model_errors, lookback=6):
    """
    在线学习集成: 根据最近N周的表现动态调权
    表现好(误差小)的模型权重高
    """
    models = list(predictions.keys())

    weights = {}
    for m in models:
        if m in model_errors and len(model_errors[m]) > 0:
            # 用最近lookback周的MAE的倒数作为权重
            recent_errors = model_errors[m][-lookback:]
            avg_error = np.mean(recent_errors) + 1  # +1避免除零
            weights[m] = 1.0 / avg_error
        else:
            weights[m] = 1.0  # 没有历史，给默认权重

    # 归一化
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    pred = sum(predictions[k] * weights[k] for k in models)
    return pred, weights


# ============================================================
# 4. 滚动回测
# ============================================================
def run_backtest(weekly):
    counts = list(weekly['count'])
    n = len(counts)

    # 预计算所有特征
    all_features = {}
    for i in range(8, n):
        feat = build_features(counts, i)
        if feat is not None:
            all_features[i] = feat

    # 模型列表 (不含集成)
    base_models = ['naive', 'ema', 'ema_opt', 'arima', 'dampened', 'regime', 'ridge']

    # 在线学习的误差记录
    model_errors = {m: [] for m in base_models}

    min_train = 12  # 至少12周历史才开始预测
    results = []

    for i in range(min_train, n - 1):
        history = counts[:i+1]
        actual = counts[i+1]

        # 各基础模型预测
        preds = {}
        preds['naive'] = model_naive(history)
        preds['ema'] = model_ema(history)
        preds['ema_opt'] = model_ema_optimized(history)
        preds['arima'] = model_arima_simple(history)
        preds['dampened'] = model_dampened_trend(history)
        preds['regime'] = model_regime_switching(history)
        preds['ridge'] = model_ridge(history, all_features, i)

        # 集成模型
        preds['median_ens'] = model_median_ensemble(preds)
        preds['online_ens'], ens_weights = model_online_ensemble(preds, model_errors, lookback=6)

        # 记录结果
        week_start = weekly.iloc[i+1]['week_start']
        result = {
            'week': week_start,
            'actual': actual,
            'idx': i+1,
        }

        for name, pred in preds.items():
            result[f'pred_{name}'] = pred
            result[f'err_{name}'] = pred - actual
            result[f'ae_{name}'] = abs(pred - actual)
            result[f'pct_err_{name}'] = abs(pred - actual) / actual * 100 if actual > 0 else 0

        results.append(result)

        # 更新在线学习的误差记录
        for m in base_models:
            model_errors[m].append(abs(preds[m] - actual))

    return results


# ============================================================
# 5. 分析报告
# ============================================================
def analyze(results):
    models = ['naive', 'ema', 'ema_opt', 'arima', 'dampened', 'regime', 'ridge', 'median_ens', 'online_ens']

    print("=" * 100)
    print("V2 回测报告 - 改进模型")
    print("=" * 100)
    print(f"回测周数: {len(results)}")
    print(f"时间: {str(results[0]['week'])[:10]} ~ {str(results[-1]['week'])[:10]}")

    actuals = [r['actual'] for r in results]
    print(f"实际值范围: {min(actuals)}-{max(actuals)}, 均值={np.mean(actuals):.0f}, 标准差={np.std(actuals):.0f}")

    print(f"\n{'模型':<14} {'MAE':>7} {'RMSE':>7} {'MAPE%':>7} {'Bias':>8} {'胜率%':>7} {'方向%':>7} {'区间%':>7}")
    print("-" * 80)

    model_metrics = {}
    for model in models:
        aes = [r[f'ae_{model}'] for r in results]
        errs = [r[f'err_{model}'] for r in results]
        pct_errs = [r[f'pct_err_{model}'] for r in results]

        mae = np.mean(aes)
        rmse = np.sqrt(np.mean([e**2 for e in errs]))
        mape = np.mean(pct_errs)
        bias = np.mean(errs)
        wins = sum(1 for p in pct_errs if p < 20) / len(pct_errs) * 100

        # 方向准确率
        dir_correct = 0
        dir_total = 0
        for j in range(1, len(results)):
            prev = results[j-1]['actual']
            act_dir = results[j]['actual'] > prev
            pred_dir = results[j][f'pred_{model}'] > prev
            if act_dir == pred_dir:
                dir_correct += 1
            dir_total += 1
        dir_acc = dir_correct / dir_total * 100 if dir_total > 0 else 0

        # 区间准确率 (40条宽区间)
        bucket_correct = 0
        for r in results:
            pred = r[f'pred_{model}']
            actual = r['actual']
            pred_bucket = int(pred // 40) * 40
            if pred_bucket - 20 <= actual <= pred_bucket + 59:
                bucket_correct += 1
        bucket_acc = bucket_correct / len(results) * 100

        model_metrics[model] = {
            'mae': mae, 'rmse': rmse, 'mape': mape,
            'bias': bias, 'winrate': wins, 'dir_acc': dir_acc,
            'bucket_acc': bucket_acc
        }

        print(f"{model:<14} {mae:>7.1f} {rmse:>7.1f} {mape:>7.1f} {bias:>+8.1f} {wins:>7.1f} {dir_acc:>7.1f} {bucket_acc:>7.1f}")

    # 最佳
    print(f"\n--- 各指标最佳 ---")
    best_mae = min(model_metrics, key=lambda m: model_metrics[m]['mae'])
    best_mape = min(model_metrics, key=lambda m: model_metrics[m]['mape'])
    best_wr = max(model_metrics, key=lambda m: model_metrics[m]['winrate'])
    best_dir = max(model_metrics, key=lambda m: model_metrics[m]['dir_acc'])
    best_bucket = max(model_metrics, key=lambda m: model_metrics[m]['bucket_acc'])
    print(f"MAE:  {best_mae} ({model_metrics[best_mae]['mae']:.1f})")
    print(f"MAPE: {best_mape} ({model_metrics[best_mape]['mape']:.1f}%)")
    print(f"胜率: {best_wr} ({model_metrics[best_wr]['winrate']:.1f}%)")
    print(f"方向: {best_dir} ({model_metrics[best_dir]['dir_acc']:.1f}%)")
    print(f"区间: {best_bucket} ({model_metrics[best_bucket]['bucket_acc']:.1f}%)")

    # 综合评分
    print(f"\n--- 综合评分 (MAE 25% + MAPE 20% + 胜率 20% + 方向 20% + 区间 15%) ---")

    maes = {m: model_metrics[m]['mae'] for m in models}
    mapes = {m: model_metrics[m]['mape'] for m in models}
    winrates = {m: model_metrics[m]['winrate'] for m in models}
    dir_accs = {m: model_metrics[m]['dir_acc'] for m in models}
    bucket_accs = {m: model_metrics[m]['bucket_acc'] for m in models}

    def normalize_lower_better(d):
        mn, mx = min(d.values()), max(d.values())
        rng = mx - mn + 0.001
        return {k: 1 - (v - mn) / rng for k, v in d.items()}

    def normalize_higher_better(d):
        mn, mx = min(d.values()), max(d.values())
        rng = mx - mn + 0.001
        return {k: (v - mn) / rng for k, v in d.items()}

    n_mae = normalize_lower_better(maes)
    n_mape = normalize_lower_better(mapes)
    n_wr = normalize_higher_better(winrates)
    n_dir = normalize_higher_better(dir_accs)
    n_bucket = normalize_higher_better(bucket_accs)

    scores = {}
    for m in models:
        scores[m] = (n_mae[m] * 0.25 + n_mape[m] * 0.20 +
                     n_wr[m] * 0.20 + n_dir[m] * 0.20 + n_bucket[m] * 0.15)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (model, score) in enumerate(ranked, 1):
        m = model_metrics[model]
        marker = " <--" if rank <= 3 else ""
        print(f"  {rank}. {model:<14} Score={score:.3f}  MAE={m['mae']:.1f}  MAPE={m['mape']:.1f}%  "
              f"胜率={m['winrate']:.1f}%  方向={m['dir_acc']:.1f}%  区间={m['bucket_acc']:.1f}%{marker}")

    # 对比V1
    print(f"\n{'='*100}")
    print("V1 vs V2 对比 (V1最佳: MAE=98.7, MAPE=28.8%, 胜率=38.9%, 方向=69.8%)")
    print(f"{'='*100}")
    v1_best = {'mae': 98.7, 'mape': 28.8, 'winrate': 38.9, 'dir_acc': 69.8}
    best_model = ranked[0][0]
    v2_best = model_metrics[best_model]
    for metric, label in [('mae', 'MAE'), ('mape', 'MAPE'), ('winrate', '胜率'), ('dir_acc', '方向')]:
        v1_val = v1_best[metric]
        v2_val = v2_best[metric]
        if metric in ['mae', 'mape']:
            diff = v1_val - v2_val
            better = "改善" if diff > 0 else "退步"
        else:
            diff = v2_val - v1_val
            better = "改善" if diff > 0 else "退步"
        print(f"  {label}: V1={v1_val:.1f} -> V2={v2_val:.1f}  ({better} {abs(diff):.1f})")

    # 逐周详情(最近20周)
    print(f"\n--- 最近20周预测详情 ---")
    print(f"{'周':<12} {'实际':>6} {'naive':>7} {'ema':>7} {'arima':>7} {'regime':>7} {'ridge':>7} {'online':>7} {'最佳':<12}")
    print("-" * 90)
    for r in results[-20:]:
        week_str = str(r['week'])[:10]
        actual = r['actual']
        best = min(models, key=lambda m: r[f'ae_{m}'])
        print(f"{week_str:<12} {actual:>6} {r['pred_naive']:>7.0f} {r['pred_ema']:>7.0f} "
              f"{r['pred_arima']:>7.0f} {r['pred_regime']:>7.0f} {r['pred_ridge']:>7.0f} "
              f"{r['pred_online_ens']:>7.0f} {best:<12}")

    return model_metrics, ranked


if __name__ == "__main__":
    print("加载数据...")
    weekly, daily = load_weekly_data()
    print(f"共 {len(weekly)} 周数据")

    print("开始V2回测...\n")
    results = run_backtest(weekly)
    metrics, ranked = analyze(results)
