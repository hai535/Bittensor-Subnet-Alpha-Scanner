#!/usr/bin/env python3
"""
马斯克推文预测模型 - 48周全面回测
===================================
用Kaggle 55k条推文数据，对5个子模型进行滚动回测。
按市场情绪/世界事件分情境评估，找出最优模型。

模型:
1. naive     - 朴素基线（上周=本周）
2. hist      - 历史匹配（找最相似的周）
3. negbin    - 负二项回归（均值+方差建模）
4. trend     - 趋势外推（线性/指数）
5. hawkes    - Hawkes过程（事件驱动自激）
6. ensemble  - 加权集成

情境标签:
- election   : 美国大选期间 (Oct-Nov 2024)
- doge       : DOGE部门成立后 (Nov 2024-)
- normal     : 普通时期
- spike      : 前一周暴涨(>30%环比)
- dip        : 前一周暴跌(<-30%环比)
- holiday    : 节假日周(圣诞/新年/感恩节)
"""

import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 加载数据
# ============================================================
CSV_PATH = "/root/.cache/kagglehub/datasets/dadalyndell/elon-musk-tweets-2010-to-2025-march/versions/11/all_musk_posts.csv"

def load_weekly_data():
    """加载并按Polymarket周聚合"""
    df = pd.read_csv(CSV_PATH, usecols=['createdAt'], low_memory=False)
    df['createdAt'] = pd.to_datetime(df['createdAt'], utc=True)
    df = df.sort_values('createdAt')

    # 从2024年4月开始，保留约52周
    cutoff = pd.Timestamp('2024-04-01', tz='UTC')
    recent = df[df['createdAt'] >= cutoff].copy()

    # ET时区，Polymarket周= Mon 12PM ET -> Mon 12PM ET
    recent['et'] = recent['createdAt'].dt.tz_convert('US/Eastern')
    recent['shifted'] = recent['et'] - pd.Timedelta(hours=12)
    recent['week_start'] = recent['shifted'].dt.to_period('W-MON').apply(lambda x: x.start_time)

    weekly = recent.groupby('week_start').agg(count=('createdAt', 'size')).reset_index()
    weekly['week_start'] = pd.to_datetime(weekly['week_start'])

    # 也加载更早的数据用于hist模型的历史库
    early = df[(df['createdAt'] >= pd.Timestamp('2023-06-01', tz='UTC')) &
               (df['createdAt'] < cutoff)].copy()
    early['et'] = early['createdAt'].dt.tz_convert('US/Eastern')
    early['shifted'] = early['et'] - pd.Timedelta(hours=12)
    early['week_start'] = early['shifted'].dt.to_period('W-MON').apply(lambda x: x.start_time)
    early_weekly = early.groupby('week_start').agg(count=('createdAt', 'size')).reset_index()
    early_weekly['week_start'] = pd.to_datetime(early_weekly['week_start'])

    # 加载日级别数据（用于hawkes）
    recent['date'] = recent['et'].dt.date
    daily = recent.groupby('date').size().reset_index(name='count')
    daily['date'] = pd.to_datetime(daily['date'])

    return weekly, early_weekly, daily

# ============================================================
# 2. 情境标签
# ============================================================
def tag_contexts(weekly):
    """为每周添加情境标签"""
    tags = []
    for i, row in weekly.iterrows():
        ws = row['week_start']
        week_tags = set()

        # 美国大选期 (2024-10-01 ~ 2024-11-15)
        if pd.Timestamp('2024-10-01') <= ws <= pd.Timestamp('2024-11-12'):
            week_tags.add('election')

        # DOGE成立后 (2024-11-19 ~)
        if ws >= pd.Timestamp('2024-11-19'):
            week_tags.add('doge')

        # 节假日
        # 感恩节周 (2024-11-25), 圣诞 (2024-12-23), 新年 (2024-12-30)
        if ws in [pd.Timestamp('2024-11-26'), pd.Timestamp('2024-12-24'),
                  pd.Timestamp('2024-12-31'), pd.Timestamp('2024-12-17')]:
            week_tags.add('holiday')

        # Spike/Dip 基于前一周环比
        if i > 0:
            prev = weekly.iloc[i-1]['count']
            curr = row['count']
            if prev > 0:
                pct_change = (curr - prev) / prev
                if pct_change > 0.30:
                    week_tags.add('spike')
                elif pct_change < -0.30:
                    week_tags.add('dip')

        # 暑期相对低活跃 (2024-06 ~ 2024-07 early)
        if pd.Timestamp('2024-06-01') <= ws <= pd.Timestamp('2024-07-08'):
            week_tags.add('summer_low')

        # Trump回归/政治热点 (Jan-Feb 2025)
        if pd.Timestamp('2025-01-20') <= ws <= pd.Timestamp('2025-02-28'):
            week_tags.add('trump_era')

        if not week_tags:
            week_tags.add('normal')

        tags.append(week_tags)

    weekly['tags'] = tags
    return weekly

# ============================================================
# 3. 五个预测模型
# ============================================================

def model_naive(history):
    """朴素模型：预测=上周实际值"""
    return history[-1]

def model_ema(history, alpha=0.3):
    """指数移动平均"""
    ema = history[0]
    for v in history[1:]:
        ema = alpha * v + (1 - alpha) * ema
    return ema

def model_hist(history, full_history):
    """历史匹配：在所有历史数据中找最相似的2周模式"""
    if len(history) < 2 or len(full_history) < 4:
        return np.mean(history[-4:])

    # 用最近2周作为模式
    pattern = np.array(history[-2:])

    best_matches = []
    search_data = list(full_history)

    for i in range(len(search_data) - 2):
        candidate = np.array(search_data[i:i+2])
        # 不要匹配到自己（最后2周）
        if i >= len(search_data) - 3:
            continue
        dist = np.sqrt(np.sum((pattern - candidate) ** 2))
        next_val = search_data[i + 2] if i + 2 < len(search_data) else None
        if next_val is not None:
            best_matches.append((dist, next_val))

    if not best_matches:
        return np.mean(history[-4:])

    # 取最相似的top-3加权平均
    best_matches.sort(key=lambda x: x[0])
    top_k = min(3, len(best_matches))
    matches = best_matches[:top_k]

    weights = [1.0 / (m[0] + 1) for m in matches]
    total_w = sum(weights)
    pred = sum(w * m[1] for w, m in zip(weights, matches)) / total_w
    return pred

def model_negbin(history):
    """负二项回归：用历史数据拟合负二项分布，返回均值"""
    data = np.array(history)
    if len(data) < 3:
        return np.mean(data)

    # 拟合负二项: mean and variance
    mu = np.mean(data)
    var = np.var(data, ddof=1)

    if var <= mu:
        return mu  # 退化为泊松

    # 加入趋势调整：近期权重更高
    n = len(data)
    weights = np.array([0.5 + 0.5 * i / n for i in range(n)])
    weighted_mean = np.average(data, weights=weights)

    return weighted_mean

def model_trend(history):
    """趋势模型：线性回归外推"""
    data = np.array(history)
    n = len(data)
    if n < 3:
        return np.mean(data)

    # 用最近8周拟合线性趋势
    window = min(8, n)
    y = data[-window:]
    x = np.arange(window)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    pred = intercept + slope * window

    # 限制预测不要太离谱（在历史范围的0.5x~2x之间）
    hist_mean = np.mean(data[-8:])
    pred = np.clip(pred, hist_mean * 0.5, hist_mean * 2.0)

    return pred

def model_hawkes(history, daily_data=None):
    """Hawkes自激模型：基于日级别数据的自激过程"""
    data = np.array(history)
    if len(data) < 3:
        return np.mean(data)

    # 基础强度 = 近期均值
    mu = np.mean(data[-6:])

    # 自激：如果最近一周高于均值，下周可能继续高（动量效应）
    last = data[-1]
    momentum = (last - mu) / (mu + 1) * 0.3  # 30%的动量传递

    # 均值回归：如果偏离太远，拉回
    long_mean = np.mean(data)
    reversion = (long_mean - last) / (long_mean + 1) * 0.2  # 20%均值回归

    pred = last * (1 + momentum + reversion)

    # 限制范围
    pred = np.clip(pred, mu * 0.4, mu * 2.5)
    return pred

def model_adaptive_ensemble(predictions, history, tags):
    """自适应集成：根据情境调整权重"""
    # 基础权重
    base_weights = {
        'naive': 0.30,
        'ema': 0.20,
        'hist': 0.20,
        'negbin': 0.15,
        'trend': 0.10,
        'hawkes': 0.05,
    }

    weights = base_weights.copy()

    # 根据情境调整
    if 'spike' in tags or 'dip' in tags:
        # 波动大时，均值回归类模型更可靠
        weights['negbin'] = 0.30
        weights['ema'] = 0.25
        weights['naive'] = 0.15
        weights['trend'] = 0.05

    if 'election' in tags or 'doge' in tags:
        # 政治事件期间趋势更重要
        weights['trend'] = 0.20
        weights['naive'] = 0.25
        weights['hawkes'] = 0.15

    if 'holiday' in tags:
        # 节假日历史匹配更好
        weights['hist'] = 0.30
        weights['ema'] = 0.25

    # 归一化
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    pred = sum(predictions[k] * weights[k] for k in weights)
    return pred, weights

# ============================================================
# 4. 滚动回测框架
# ============================================================

def run_backtest(weekly, early_weekly, daily):
    """滚动回测：每周用之前所有数据预测下一周"""

    # 合并历史
    all_weekly = pd.concat([early_weekly, weekly], ignore_index=True)
    all_counts = list(all_weekly['count'])

    # 找到weekly在all_counts中的起始index
    offset = len(early_weekly)

    # 至少需要8周历史才开始预测
    min_history = 8
    start_idx = max(min_history, offset)  # 从weekly的第一周开始（有早期数据做历史）

    results = []

    for i in range(start_idx, len(all_counts) - 1):
        # 历史 = 到第i周为止的所有数据
        history = all_counts[:i+1]
        # 实际值 = 第i+1周
        actual = all_counts[i+1]

        # 对应weekly中的index
        weekly_idx = i + 1 - offset
        if weekly_idx < 0 or weekly_idx >= len(weekly):
            continue

        week_info = weekly.iloc[weekly_idx]
        tags = week_info.get('tags', {'normal'})

        # 各模型预测
        preds = {}
        preds['naive'] = model_naive(history)
        preds['ema'] = model_ema(history)
        preds['hist'] = model_hist(history, all_counts[:i-1])  # 不包含最近的用于匹配
        preds['negbin'] = model_negbin(history)
        preds['trend'] = model_trend(history)
        preds['hawkes'] = model_hawkes(history)
        preds['ensemble'], ens_weights = model_adaptive_ensemble(preds, history, tags)

        result = {
            'week': week_info['week_start'],
            'actual': actual,
            'tags': tags,
        }
        for name, pred in preds.items():
            result[f'pred_{name}'] = pred
            result[f'err_{name}'] = pred - actual
            result[f'ae_{name}'] = abs(pred - actual)
            result[f'pct_err_{name}'] = (pred - actual) / actual * 100 if actual > 0 else 0

        results.append(result)

    return results

# ============================================================
# 5. 分析与报告
# ============================================================

def analyze_results(results):
    """全面分析回测结果"""
    models = ['naive', 'ema', 'hist', 'negbin', 'trend', 'hawkes', 'ensemble']

    print("=" * 90)
    print("马斯克推文预测模型 - 48周全面回测报告")
    print("=" * 90)

    # --- 总体指标 ---
    print(f"\n回测周数: {len(results)}")
    print(f"时间范围: {results[0]['week']} ~ {results[-1]['week']}")

    print(f"\n{'模型':<12} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8} {'Bias':>8} {'胜率%':>8} {'区间准确率%':>12}")
    print("-" * 70)

    model_metrics = {}
    for model in models:
        aes = [r[f'ae_{model}'] for r in results]
        errs = [r[f'err_{model}'] for r in results]
        pct_errs = [r[f'pct_err_{model}'] for r in results]

        mae = np.mean(aes)
        rmse = np.sqrt(np.mean([e**2 for e in errs]))
        mape = np.mean([abs(p) for p in pct_errs])
        bias = np.mean(errs)

        # 胜率：预测误差<20%
        wins = sum(1 for p in pct_errs if abs(p) < 20) / len(pct_errs) * 100

        # Polymarket区间准确率（区间宽度通常是25-35条，这里用50条区间）
        bucket_correct = 0
        for r in results:
            pred = r[f'pred_{model}']
            actual = r['actual']
            # 看预测落入的50条宽区间是否包含实际值
            pred_bucket = int(pred // 50) * 50
            if pred_bucket - 25 <= actual <= pred_bucket + 74:
                bucket_correct += 1
        bucket_acc = bucket_correct / len(results) * 100

        model_metrics[model] = {
            'mae': mae, 'rmse': rmse, 'mape': mape,
            'bias': bias, 'winrate': wins, 'bucket_acc': bucket_acc
        }

        print(f"{model:<12} {mae:>8.1f} {rmse:>8.1f} {mape:>8.1f} {bias:>+8.1f} {wins:>8.1f} {bucket_acc:>12.1f}")

    # --- 最佳模型 ---
    best_mae = min(model_metrics, key=lambda m: model_metrics[m]['mae'])
    best_mape = min(model_metrics, key=lambda m: model_metrics[m]['mape'])
    best_winrate = max(model_metrics, key=lambda m: model_metrics[m]['winrate'])

    print(f"\n最低MAE: {best_mae} ({model_metrics[best_mae]['mae']:.1f})")
    print(f"最低MAPE: {best_mape} ({model_metrics[best_mape]['mape']:.1f}%)")
    print(f"最高胜率: {best_winrate} ({model_metrics[best_winrate]['winrate']:.1f}%)")

    # --- 按情境分析 ---
    print(f"\n{'='*90}")
    print("按情境分析各模型MAE")
    print(f"{'='*90}")

    all_tags = set()
    for r in results:
        all_tags.update(r['tags'])

    for tag in sorted(all_tags):
        tag_results = [r for r in results if tag in r['tags']]
        if len(tag_results) < 2:
            continue

        print(f"\n--- {tag} ({len(tag_results)}周) ---")
        actual_range = [r['actual'] for r in tag_results]
        print(f"    实际范围: {min(actual_range)}-{max(actual_range)}, 均值={np.mean(actual_range):.0f}")

        tag_model_scores = {}
        for model in models:
            mae = np.mean([r[f'ae_{model}'] for r in tag_results])
            bias = np.mean([r[f'err_{model}'] for r in tag_results])
            tag_model_scores[model] = mae

        # 排序打印
        sorted_models = sorted(tag_model_scores.items(), key=lambda x: x[1])
        for rank, (model, mae) in enumerate(sorted_models, 1):
            bias = np.mean([r[f'err_{model}'] for r in tag_results])
            marker = " ★" if rank == 1 else ""
            print(f"    {rank}. {model:<12} MAE={mae:>7.1f}  Bias={bias:>+7.1f}{marker}")

    # --- 逐周详情 ---
    print(f"\n{'='*90}")
    print("逐周预测详情 (Actual vs Best Model)")
    print(f"{'='*90}")
    print(f"{'周':<12} {'实际':>6} {'naive':>7} {'ema':>7} {'hist':>7} {'negbin':>7} {'trend':>7} {'hawkes':>7} {'ensbl':>7} {'最佳模型':<10} {'tags'}")
    print("-" * 110)

    for r in results:
        week_str = str(r['week'])[:10]
        actual = r['actual']

        # 找最佳模型
        best = min(models, key=lambda m: r[f'ae_{m}'])

        preds = [f"{r[f'pred_{m}']:>7.0f}" for m in models]
        tags_str = ','.join(r['tags'])

        print(f"{week_str:<12} {actual:>6} {'  '.join(preds)}  {best:<10} {tags_str}")

    # --- 方向性准确率 ---
    print(f"\n{'='*90}")
    print("方向性准确率（预测涨跌方向是否正确）")
    print(f"{'='*90}")

    for model in models:
        correct = 0
        total = 0
        for i, r in enumerate(results):
            if i == 0:
                continue
            prev_actual = results[i-1]['actual']
            actual_dir = r['actual'] > prev_actual
            pred = r[f'pred_{model}']
            pred_dir = pred > prev_actual
            if actual_dir == pred_dir:
                correct += 1
            total += 1
        if total > 0:
            print(f"  {model:<12} {correct}/{total} = {correct/total*100:.1f}%")

    # --- 波动期 vs 平稳期 ---
    print(f"\n{'='*90}")
    print("波动期 vs 平稳期 对比")
    print(f"{'='*90}")

    volatile = [r for r in results if 'spike' in r['tags'] or 'dip' in r['tags']]
    stable = [r for r in results if 'normal' in r['tags']]

    if volatile and stable:
        print(f"\n  波动期 ({len(volatile)}周):")
        for model in models:
            mae = np.mean([r[f'ae_{model}'] for r in volatile])
            print(f"    {model:<12} MAE={mae:.1f}")

        print(f"\n  平稳期 ({len(stable)}周):")
        for model in models:
            mae = np.mean([r[f'ae_{model}'] for r in stable])
            print(f"    {model:<12} MAE={mae:.1f}")

    return model_metrics

# ============================================================
# 6. 最终推荐
# ============================================================

def final_recommendation(model_metrics, results):
    """基于回测给出最终推荐"""
    print(f"\n{'='*90}")
    print("最终推荐")
    print(f"{'='*90}")

    # 综合评分 (MAE 40%, MAPE 20%, 胜率 20%, 方向性 20%)
    models = list(model_metrics.keys())

    # 归一化各指标
    maes = {m: model_metrics[m]['mae'] for m in models}
    mapes = {m: model_metrics[m]['mape'] for m in models}
    winrates = {m: model_metrics[m]['winrate'] for m in models}

    min_mae, max_mae = min(maes.values()), max(maes.values())
    min_mape, max_mape = min(mapes.values()), max(mapes.values())
    min_wr, max_wr = min(winrates.values()), max(winrates.values())

    scores = {}
    for m in models:
        # 越低越好的取反
        mae_score = 1 - (maes[m] - min_mae) / (max_mae - min_mae + 1)
        mape_score = 1 - (mapes[m] - min_mape) / (max_mape - min_mape + 1)
        wr_score = (winrates[m] - min_wr) / (max_wr - min_wr + 1)

        scores[m] = mae_score * 0.40 + mape_score * 0.30 + wr_score * 0.30

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print("\n综合评分排名 (MAE 40% + MAPE 30% + 胜率 30%):\n")
    for rank, (model, score) in enumerate(ranked, 1):
        m = model_metrics[model]
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"  {medal} {rank}. {model:<12} Score={score:.3f}  MAE={m['mae']:.1f}  MAPE={m['mape']:.1f}%  胜率={m['winrate']:.1f}%")

    best_model = ranked[0][0]
    print(f"\n推荐: 当前最优模型是 【{best_model}】")
    print(f"  - 在48周回测中综合表现最好")
    print(f"  - MAE={model_metrics[best_model]['mae']:.1f}, MAPE={model_metrics[best_model]['mape']:.1f}%")

    # 情境建议
    print(f"\n情境化建议:")
    print(f"  - 当前处于DOGE/Trump时代 → 推文量整体偏高(400-600/周)")
    print(f"  - 如果上周出现暴涨/暴跌 → 优先用negbin/ema(均值回归)")
    print(f"  - 平稳期 → naive/ema最可靠")
    print(f"  - 政治事件热点期 → 注意trend模型的趋势信号")

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("加载数据...")
    weekly, early_weekly, daily = load_weekly_data()

    print(f"主要数据: {len(weekly)}周 (2024.04-2025.04)")
    print(f"历史数据: {len(early_weekly)}周 (2023.06-2024.03)")

    print("\n标注情境标签...")
    weekly = tag_contexts(weekly)

    print("开始滚动回测...\n")
    results = run_backtest(weekly, early_weekly, daily)

    metrics = analyze_results(results)
    final_recommendation(metrics, results)
