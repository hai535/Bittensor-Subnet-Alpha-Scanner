# ⛓ Bittensor Intelligence Hub

[English](#english) | [中文](#中文)

---

<a id="english"></a>

## 🇺🇸 English

### Overview

**Bittensor Intelligence Hub** is a real-time on-chain analytics platform for the Bittensor ecosystem. It aggregates subnet emission data, whale transaction monitoring, and ecosystem news into a single, unified dashboard.

### 🌐 Live Demo

**http://46.224.8.188:7683**

### ✨ Features

#### ⛏ Subnet Alpha Scanner
- Real-time scanning of all ~129 Bittensor subnets
- Filter subnets by Alpha token emission intervals (1-12h)
- Track active miner count, price volatility, and registration anomalies
- 4-layer filtering: Miners > 10 + 24h volatility < 5% + Registration anomaly detection + Emission interval
- Registration anomaly scoring (0-100) with rush/instant-full detection

#### 🐋 Whale Monitor
- On-chain transaction crawling for large TAO holders (>10,000 TAO)
- 4 analysis tabs: Anomaly Alerts / Hot Subnets / Whale List / Split Wallet Detection
- Detect: Heavy buys, concentrated accumulation, stealth building, high-frequency trading
- Split wallet detection: Multiple wallets buying the same subnet simultaneously
- 1d/3d/7d behavioral analysis per whale

#### 📰 Latest News
- Aggregated from: Cryptocurrency news sites, Reddit r/bittensor_, Bittensor chain events, CoinGecko market data
- Time-based filtering: Last 1h / 1d / 1w / Important events
- Live $TAO price ticker with 24h/7d changes, market cap, volume
- Community metrics: Reddit subscribers, Telegram members

#### 🌍 Language Support
- **Chinese / English** toggle in the top-right corner
- Preference saved in localStorage

### 🛠 Tech Stack

- **Backend**: Python 3 + Flask + aiohttp (async API calls)
- **Frontend**: Single-page HTML with vanilla JavaScript
- **Data Sources**: TaoMarketCap API, CoinGecko API, RSS feeds, Bittensor chain
- **Deployment**: systemd service on Linux

### 🚀 Quick Start

```bash
# Install dependencies
pip install flask flask-cors aiohttp

# Run
python3 subnet_scanner.py
# Server starts at http://0.0.0.0:7683
```

### 📦 Files

| File | Description |
|------|-------------|
| `subnet_scanner.py` | Main application (backend + frontend) |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |
| `.gitignore` | Git ignore rules |

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard |
| `/api/scan` | GET | Scan all subnets for Alpha emissions |
| `/api/whales` | GET | Scan whale on-chain transactions |
| `/api/whale/<address>` | GET | Get specific whale detail (params: `days=7`) |
| `/api/news` | GET | Aggregate ecosystem news |

---

<a id="中文"></a>

## 🇨🇳 中文

### 概述

**Bittensor Intelligence Hub** 是一个实时链上分析平台，专为 Bittensor 生态系统打造。将子网排放数据、鲸鱼交易监测和生态新闻整合到一个统一的仪表板中。

### 🌐 在线演示

**http://46.224.8.188:7683**

### ✨ 功能

#### ⛏ 子网 Alpha 扫描器
- 实时扫描全部 ~129 个 Bittensor 子网
- 按 Alpha 代币排放间隔（1-12小时）筛选子网
- 追踪活跃矿工数、价格波动率、注册异常
- 4 层过滤：矿工>10 + 24h波动<5% + 注册异常检测 + 排放间隔
- 注册异常评分（0-100），检测秒满/抢注模式

#### 🐋 鲸鱼监测
- 爬取链上大户（持仓>10,000 TAO）交易记录
- 4 个分析标签：异常预警 / 热门子网 / 鲸鱼列表 / 分钱包检测
- 检测：大额买入、集中建仓、隐蔽建仓、高频交易
- 分钱包检测：多个钱包同时买入同一子网
- 每个鲸鱼的 1天/3天/7天 行为分析

#### 📰 最新消息
- 聚合来源：加密货币新闻网站、Reddit r/bittensor_、链上事件、CoinGecko 市场数据
- 时间筛选：近1小时 / 近1天 / 近1周 / 重要事件
- $TAO 实时行情：24h/7d涨跌、市值、交易量
- 社区指标：Reddit 订阅者、Telegram 成员

#### 🌍 多语言支持
- 右上角 **中文/英文** 一键切换
- 语言偏好自动保存在 localStorage

### 🛠 技术栈

- **后端**：Python 3 + Flask + aiohttp（异步API调用）
- **前端**：单页 HTML + 原生 JavaScript
- **数据源**：TaoMarketCap API、CoinGecko API、RSS 订阅、Bittensor 链上数据
- **部署**：Linux systemd 服务

### 🚀 快速开始

```bash
# 安装依赖
pip install flask flask-cors aiohttp

# 运行
python3 subnet_scanner.py
# 服务启动于 http://0.0.0.0:7683
```

### 📦 文件说明

| 文件 | 说明 |
|------|------|
| `subnet_scanner.py` | 主程序（后端 + 前端） |
| `requirements.txt` | Python 依赖 |
| `README.md` | 本文件 |
| `.gitignore` | Git 忽略规则 |

### API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 主仪表板 |
| `/api/scan` | GET | 扫描所有子网 Alpha 排放 |
| `/api/whales` | GET | 扫描鲸鱼链上交易 |
| `/api/whale/<地址>` | GET | 查询特定鲸鱼详情（参数：`days=7`） |
| `/api/news` | GET | 聚合生态新闻 |

---

## 📜 License

MIT

## 👤 Author

Built with ❤️ by hai535
