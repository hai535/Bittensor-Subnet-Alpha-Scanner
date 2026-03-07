import sys
sys.path.insert(0, "/root/claude-chat")
from send_mail import send_email

subject_matter = "Bitget Wallet Skill (测试版) 功能介绍"

body_content = "\n".join([
    "以下是 Bitget Wallet Skill (测试版) 的相关信息汇总:",
    "",
    "1. 功能概述",
    "   Bitget Wallet Skill 是宝宝集成的链上钱包管理技能，用于在 Base 链上进行批量操作和资产管理。",
    "",
    "2. 钱包配置",
    "   - 主钱包: wallet_main.json",
    "   - 子钱包 (20个): wallet_01~20.json",
    "   - 默认操作链: Base",
    "",
    "3. 核心功能",
    "   - 批量转账: 支持从主钱包向多个子钱包批量转账",
    "   - 自动归集: 支持将子钱包资产自动归集到主钱包",
    "   - 钱包余额查询: 批量查询所有钱包的链上余额",
    "",
    "4. 安全措施",
    "   - 私钥/助记词绝不在任何输出中显示",
    "   - 所有钱包文件存放在服务器本地，不上传任何远程服务",
    "   - 转账操作前需主人确认",
    "",
    "5. 配套安全检测工具",
    "   - 代币安全检测脚本: token-check.sh",
    "   - 持仓操纵检测: holder-analysis.py",
    "   - 快捷指令: 主人发送 /CA + 合约地址，宝宝自动跑安全检测",
    "",
    "6. 使用方式",
    "   - 主人直接告诉宝宝操作需求即可 (如 查余额、转0.01 ETH到子钱包、归集所有子钱包 等)",
    "   - 宝宝会先确认操作细节，得到主人确认后再执行",
    "",
    "注意: 当前为测试版，功能持续迭代中。如有新需求请随时告诉宝宝。",
])

send_email(subject_matter, body_content)
