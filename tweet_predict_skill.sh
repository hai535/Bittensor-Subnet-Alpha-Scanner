#!/bin/bash
# =============================================================
# Elon Musk Tweet Predictor Skill v4 - XTracker Calibrated
# =============================================================
# 用法:
#   ./tweet_predict_skill.sh                     # 使用默认参数
#   ./tweet_predict_skill.sh --live              # 从XTracker API实时获取数据
#   ./tweet_predict_skill.sh 150 3.5             # 指定当前推文数和已过天数
#   ./tweet_predict_skill.sh --json              # JSON输出
#   ./tweet_predict_skill.sh --update            # 运行并更新报告文件
#   ./tweet_predict_skill.sh --refresh-data      # 重新从XTracker拉取全部历史数据
# =============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PREDICTOR="$SCRIPT_DIR/elon_tweet_predictor_v4.py"
REPORT="$SCRIPT_DIR/polymarket_analysis.md"
DATA_CACHE="$SCRIPT_DIR/xtracker_daily_data.json"

# 默认参数
CURRENT_COUNT="122"
DAYS_ELAPSED="3.0"
OUTPUT_FORMAT="text"
DO_UPDATE=false
DO_LIVE=false
DO_REFRESH=false
EXTRA_ARGS=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --live)
            DO_LIVE=true
            shift
            ;;
        --json)
            OUTPUT_FORMAT="json"
            shift
            ;;
        --update)
            DO_UPDATE=true
            shift
            ;;
        --refresh-data)
            DO_REFRESH=true
            shift
            ;;
        *)
            if [[ "$1" =~ ^[0-9]+$ ]]; then
                CURRENT_COUNT="$1"
                shift
                if [[ "$1" =~ ^[0-9.]+$ ]]; then
                    DAYS_ELAPSED="$1"
                    shift
                fi
            else
                shift
            fi
            ;;
    esac
done

# 刷新历史数据缓存
if [ "$DO_REFRESH" = true ] || [ ! -f "$DATA_CACHE" ]; then
    echo ">>> Refreshing XTracker historical data..."
    python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from elon_tweet_predictor_v4 import RealDataCalibrator
cal = RealDataCalibrator()
print(f'>>> Cached {len(cal.daily_counts)} days of data')
" 2>&1
    echo ""
fi

echo "============================================"
echo "  Elon Musk Tweet Predictor v4"
echo "  XTracker Calibrated Ensemble"
echo "============================================"

if [ "$DO_LIVE" = true ]; then
    echo "  Mode: LIVE (fetching from XTracker API)"
    EXTRA_ARGS="--live"
else
    echo "  Tweets so far:  $CURRENT_COUNT"
    echo "  Days elapsed:   $DAYS_ELAPSED"
fi

echo "  Models: NegBin + Hawkes + Regime + Hist + Trend"
echo "  Data: 113+ days real XTracker data"
echo "============================================"
echo ""

if [ "$DO_LIVE" = true ]; then
    if [ "$OUTPUT_FORMAT" = "json" ]; then
        python3 "$PREDICTOR" --live --json 2>/dev/null
    else
        RESULT=$(python3 "$PREDICTOR" --live 2>&1)
        echo "$RESULT"
    fi
else
    if [ "$OUTPUT_FORMAT" = "json" ]; then
        python3 "$PREDICTOR" --current-count "$CURRENT_COUNT" --days-elapsed "$DAYS_ELAPSED" --json 2>/dev/null
    else
        RESULT=$(python3 "$PREDICTOR" --current-count "$CURRENT_COUNT" --days-elapsed "$DAYS_ELAPSED" 2>&1)
        echo "$RESULT"
    fi
fi

if [ "$DO_UPDATE" = true ] && [ "$OUTPUT_FORMAT" != "json" ]; then
    echo ""
    echo ">>> Updating $REPORT ..."

    python3 << PYEOF
report_path = "$REPORT"
marker = "## 九、"

try:
    with open(report_path, 'r') as f:
        content = f.read()
except FileNotFoundError:
    content = "# Polymarket Analysis\n\n"

import datetime
now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

new_section = f"""## 九、Elon Musk 推特预测 v4 - XTracker Calibrated ({now} 更新)

> 模型: **5模型Ensemble** (负二项 + Hawkes + 政权切换 + 历史匹配 + 趋势振荡)
> 数据源: XTracker API (113天真实日级别数据)
> 预测脚本: \`elon_tweet_predictor_v4.py\`
> 快捷指令: \`bash tweet_predict_skill.sh --live\`

$RESULT
"""

if marker in content:
    before = content[:content.index(marker)]
    sources_marker = "## 数据来源"
    sources = ""
    if sources_marker in content:
        sources = content[content.index(sources_marker):]
    with open(report_path, 'w') as f:
        f.write(before + new_section + "\n---\n\n" + sources)
else:
    with open(report_path, 'a') as f:
        f.write("\n\n---\n\n" + new_section)

print(">>> Report updated!")
PYEOF
fi
