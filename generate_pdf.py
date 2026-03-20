#!/usr/bin/env python3
from fpdf import FPDF

class PDF(FPDF):
    def __init__(self):
        super().__init__()
        # Add Chinese font
        self.add_font("noto", "", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", uni=True)
        self.add_font("noto", "B", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", uni=True)
        self.set_auto_page_break(auto=True, margin=20)

    def colored_table(self, headers, data, col_widths=None):
        if col_widths is None:
            col_widths = [self.epw / len(headers)] * len(headers)
        # Header
        self.set_font("noto", "B", 9)
        self.set_fill_color(26, 86, 219)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 8, h, border=1, fill=True, align="C")
        self.ln()
        # Data
        self.set_font("noto", "", 9)
        self.set_text_color(30, 30, 30)
        for row_idx, row in enumerate(data):
            if row_idx % 2 == 1:
                self.set_fill_color(245, 247, 252)
                fill = True
            else:
                self.set_fill_color(255, 255, 255)
                fill = True
            max_lines = 1
            cell_texts = []
            for i, val in enumerate(row):
                lines = self.multi_cell(col_widths[i], 6, val, split_only=True)
                cell_texts.append(lines)
                if len(lines) > max_lines:
                    max_lines = len(lines)
            row_h = max_lines * 6
            # Check page break
            if self.get_y() + row_h > self.h - 20:
                self.add_page()
            y_start = self.get_y()
            x_start = self.get_x()
            for i, lines in enumerate(cell_texts):
                x = x_start + sum(col_widths[:i])
                self.set_xy(x, y_start)
                self.rect(x, y_start, col_widths[i], row_h, "DF" if fill else "D")
                for j, line in enumerate(lines):
                    self.set_xy(x + 1, y_start + j * 6)
                    self.cell(col_widths[i] - 2, 6, line)
            self.set_xy(x_start, y_start + row_h)
        self.ln(3)

    def section_title(self, num, title):
        self.ln(6)
        self.set_font("noto", "B", 14)
        self.set_text_color(26, 86, 219)
        self.cell(0, 10, f"{num}、{title}", ln=True)
        self.set_draw_color(26, 86, 219)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)
        self.set_text_color(30, 30, 30)

    def sub_title(self, text):
        self.ln(3)
        self.set_font("noto", "B", 11)
        self.set_text_color(50, 50, 50)
        self.cell(0, 8, text, ln=True)
        self.set_text_color(30, 30, 30)

    def body_text(self, text):
        self.set_font("noto", "", 10)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def alert_box(self, text, box_type="warning"):
        colors = {
            "warning": (255, 243, 205, 255, 193, 7),
            "danger": (248, 215, 218, 220, 53, 69),
            "success": (212, 237, 218, 40, 167, 69),
            "info": (209, 236, 241, 23, 162, 184),
        }
        bg_r, bg_g, bg_b, br_r, br_g, br_b = colors.get(box_type, colors["warning"])
        x = self.get_x()
        y = self.get_y()
        self.set_font("noto", "", 9)
        lines = self.multi_cell(self.epw - 6, 5.5, text, split_only=True)
        h = len(lines) * 5.5 + 8
        if y + h > self.h - 20:
            self.add_page()
            y = self.get_y()
        self.set_fill_color(bg_r, bg_g, bg_b)
        self.rect(x, y, self.epw, h, "F")
        self.set_draw_color(br_r, br_g, br_b)
        self.line(x, y, x, y + h)
        self.line(x, y, x, y + h)
        self.rect(x, y, 1.5, h, "F")
        self.set_fill_color(br_r, br_g, br_b)
        self.rect(x, y, 1.5, h, "F")
        self.set_xy(x + 4, y + 3)
        self.multi_cell(self.epw - 10, 5.5, text)
        self.set_xy(x, y + h + 3)

    def bullet_list(self, items):
        self.set_font("noto", "", 10)
        for item in items:
            self.cell(5, 6, "·")
            self.multi_cell(self.epw - 10, 6, item)
        self.ln(2)


pdf = PDF()
pdf.set_title("TikTok跨境电商专线 - 用户使用须知")
pdf.set_author("跨境电商技术服务")

# === Cover Page ===
pdf.add_page()
pdf.ln(60)
pdf.set_font("noto", "B", 28)
pdf.set_text_color(26, 86, 219)
pdf.cell(0, 15, "TikTok跨境电商专线", ln=True, align="C")
pdf.ln(5)
pdf.set_font("noto", "", 14)
pdf.set_text_color(100, 100, 100)
pdf.cell(0, 10, "用户使用须知与安全操作规范", ln=True, align="C")
pdf.ln(3)
pdf.set_draw_color(26, 86, 219)
pdf.line(75, pdf.get_y(), 135, pdf.get_y())
pdf.ln(30)
pdf.set_font("noto", "", 11)
pdf.set_text_color(120, 120, 120)
for line in ["版本：V1.0", "生效日期：2026年3月9日", "适用范围：全体用户", "密级：内部文件"]:
    pdf.cell(0, 8, line, ln=True, align="C")

# === Page 2: 服务概述 ===
pdf.add_page()
pdf.section_title("一", "服务概述")
pdf.body_text("本服务为TikTok跨境电商运营者提供稳定、安全的网络加速通道，支持以下业务场景：")
pdf.colored_table(
    ["支持场景", "说明", "推荐套餐"],
    [
        ["TikTok账号日常运营", "刷视频、发布内容、互动", "基础版及以上"],
        ["TikTok直播带货", "实时直播，低延迟不断线", "直播专线版"],
        ["TikTok Shop店铺管理", "商品上架、订单处理", "专业版及以上"],
        ["短视频剪辑上传", "大文件上传，带宽充足", "基础版及以上"],
        ["多账号矩阵运营", "多店铺/多账号独立IP运营", "团队版"],
    ],
    [60, 70, 40],
)

# === 账号安全 ===
pdf.section_title("二", "账号安全 — 核心操作规范")
pdf.alert_box("【重要提醒】以下规范直接关系到您TikTok账号的安全。违反规范导致的账号异常或封禁，服务方不承担责任。", "danger")

pdf.sub_title("2.1 一号一IP原则（最重要）")
pdf.colored_table(
    ["规则", "正确做法", "错误做法"],
    [
        ["IP绑定", "每个TikTok账号固定使用一个IP", "多个账号共用同一个IP"],
        ["设备绑定", "一台手机只登录一个TikTok账号", "同一手机来回切换多个账号"],
        ["节点选择", "选定节点后长期使用，不频繁更换", "今天用新加坡，明天用美国"],
        ["IP一致性", "注册、养号、运营全程使用同一IP", "注册用A节点，运营用B节点"],
    ],
    [35, 70, 65],
)

pdf.sub_title("2.2 设备环境配置")
pdf.colored_table(
    ["配置项", "要求", "设置方法"],
    [
        ["系统语言", "与目标市场一致", "如做美区设English(US)"],
        ["系统时区", "与节点IP所在地一致", "新加坡节点设GMT+8，美西设GMT-8"],
        ["GPS/定位", "必须关闭", "设置→隐私→定位服务→关闭"],
        ["SIM卡", "取出国内SIM卡或开飞行模式再连WiFi", "确保无中国运营商信号"],
        ["App Store地区", "切换到目标市场", "苹果用海外Apple ID"],
        ["TikTok版本", "使用国际版TikTok（非抖音）", "从Google Play/海外App Store下载"],
    ],
    [35, 65, 70],
)

pdf.alert_box("【注意】使用模拟器（如夜神、雷电）运营TikTok风险极高，强烈建议使用真实手机操作。", "warning")

pdf.sub_title("2.3 新号养号规范（前7天关键期）")
pdf.colored_table(
    ["天数", "操作内容", "时长", "禁止事项"],
    [
        ["第1天", "注册账号，完善资料，浏览推荐页", "30-60分钟", "不发视频、不关注"],
        ["第2-3天", "浏览视频，点赞5-10个，关注3-5个同行", "每天30-60分钟", "不发视频、不私信"],
        ["第4-5天", "评论3-5条，分享1-2个视频，浏览直播间", "每天45-60分钟", "不发广告评论"],
        ["第6-7天", "发布第一条视频（原创），继续日常互动", "每天60分钟", "不挂购物车链接"],
        ["第2周起", "逐步增加发布频率（每天1-2条）", "按需", "避免一天发超过5条"],
    ],
    [25, 65, 35, 45],
)

pdf.alert_box("【养号核心】模拟真实用户行为，让TikTok算法认为你是正常用户。越自然越安全。", "success")

# === 连接使用规范 ===
pdf.add_page()
pdf.section_title("三", "连接使用规范")

pdf.sub_title("3.1 正确的连接流程")
pdf.colored_table(
    ["步骤", "操作", "说明"],
    [
        ["①", "先连接VPN节点", "确认连接成功后再操作"],
        ["②", "检查IP地址", "浏览器打开 ip.sb 确认IP正确"],
        ["③", "检查IP纯净度（可选）", "访问 whoer.net 检测，建议>70分"],
        ["④", "打开TikTok使用", "正常运营"],
        ["⑤", "使用结束后再断开VPN", "先关TikTok，再断VPN"],
    ],
    [20, 55, 95],
)

pdf.alert_box("【严禁】在未连接VPN的情况下打开TikTok！这会暴露您的真实中国IP，可能导致账号被标记。", "danger")

pdf.sub_title("3.2 连接异常处理")
pdf.colored_table(
    ["异常情况", "处理方法", "注意事项"],
    [
        ["VPN断开", "立即关闭TikTok，重连VPN后再打开", "不要在断线状态下继续使用"],
        ["延迟过高(>300ms)", "联系客服切换备用节点", "不要自行频繁切换节点"],
        ["无法连接", "检查订阅是否过期，联系客服", "不要用其他VPN临时替代"],
        ["直播卡顿", "降低画质，或联系客服升级线路", "避开高峰期(晚8-11点)直播"],
    ],
    [40, 65, 65],
)

# === 严禁行为 ===
pdf.section_title("四", "严禁行为清单")
pdf.alert_box("以下行为将导致【立即终止服务且不退款】：", "danger")
pdf.colored_table(
    ["序号", "禁止行为", "后果"],
    [
        ["1", "将节点/订阅信息分享给他人使用", "终止服务"],
        ["2", "使用自动化脚本批量注册账号", "终止服务 + IP污染"],
        ["3", "使用刷量/刷粉等黑灰产工具", "终止服务 + 账号封禁"],
        ["4", "进行网络攻击、扫描、渗透测试", "立即终止 + 报警"],
        ["5", "发布违法违规内容（涉黄、涉赌、涉诈）", "立即终止 + 报警"],
        ["6", "利用服务进行网络诈骗", "立即终止 + 报警"],
        ["7", "私自搭建二级代理转售", "终止服务"],
        ["8", "对服务进行逆向工程或破解", "终止服务"],
    ],
    [15, 80, 75],
)

# === 直播专项规范 ===
pdf.section_title("五", "直播专项规范")
pdf.colored_table(
    ["项目", "建议", "原因"],
    [
        ["网络带宽", "上行至少10Mbps", "直播画质保障"],
        ["连接方式", "WiFi优先，避免移动数据", "WiFi更稳定"],
        ["直播前测试", "开播前10分钟连接VPN并测速", "提前发现问题"],
        ["备用方案", "准备备用节点，卡顿时快速切换", "减少直播中断时间"],
        ["直播时段", "避开目标市场凌晨时段", "不符合正常用户行为"],
        ["画质设置", "720P为主，网络好时可用1080P", "平衡画质与流量消耗"],
    ],
    [35, 70, 65],
)

# === 流量使用参考 ===
pdf.add_page()
pdf.section_title("六", "流量使用参考")
pdf.colored_table(
    ["使用场景", "预估流量消耗", "基础版(1TB/月)可用时长"],
    [
        ["刷视频/浏览", "约0.5-1GB/小时", "约1000-2000小时"],
        ["上传短视频(1分钟)", "约50-200MB/个", "约5000-20000个"],
        ["TikTok直播(720P)", "约1-2GB/小时", "约500-1000小时"],
        ["TikTok直播(1080P)", "约2-3GB/小时", "约333-500小时"],
        ["TikTok Shop管理", "约0.2-0.5GB/小时", "约2000-5000小时"],
    ],
    [55, 55, 60],
)
pdf.alert_box("【提示】流量用量可在客户端实时查看。建议每周检查一次用量，合理规划使用。", "info")

# === FAQ ===
pdf.section_title("七", "常见问题（FAQ）")
pdf.colored_table(
    ["问题", "解答"],
    [
        ["连上VPN后TikTok显示\"网络不可用\"", "检查是否开启全局代理模式；尝试清除TikTok缓存后重试"],
        ["TikTok提示\"该地区不可用\"", "检查GPS是否关闭、SIM卡是否取出、系统语言是否正确"],
        ["视频发布后0播放", "可能是内容问题而非网络问题；检查是否有搬运/违规内容"],
        ["直播间观众看到的位置不对", "TikTok直播位置基于IP，确认IP所在地与期望一致"],
        ["多久可以换一次IP？", "不建议频繁更换；确需更换请提前24小时联系客服"],
        ["可以用电脑端运营吗？", "可以，但需在电脑上也配置VPN客户端，保持IP一致"],
        ["节点订阅怎么更新？", "客户端一般自动更新；如失效请联系客服获取最新订阅链接"],
    ],
    [65, 105],
)

# === 客服与技术支持 ===
pdf.section_title("八", "客服与技术支持")
pdf.colored_table(
    ["支持渠道", "联系方式", "响应时间"],
    [
        ["日常咨询", "微信客服 / Telegram群组", "24小时内"],
        ["紧急故障（直播中断等）", "专属客服热线", "15分钟内"],
        ["账号安全咨询", "一对一专属顾问", "2小时内"],
    ],
    [50, 65, 55],
)
pdf.alert_box("【服务承诺】月度可用性≥99.5%。如因服务方原因导致连续中断超过4小时，将按比例补偿服务时长。", "info")

# === 免责声明 ===
pdf.section_title("九", "免责声明")
pdf.bullet_list([
    "本服务仅提供网络加速通道，不对用户在第三方平台（TikTok等）的账号安全承担责任。",
    "因用户违反本须知中的操作规范导致的账号异常、封禁等问题，服务方不承担责任。",
    "用户应遵守所在国家/地区的法律法规，因违法使用导致的一切后果由用户自行承担。",
    "服务方有权根据实际情况更新本须知，更新后将通过客服渠道通知用户。",
    "本服务不鼓励、不支持任何违反TikTok平台规则的行为。",
])

pdf.ln(10)
pdf.set_font("noto", "", 8)
pdf.set_text_color(150, 150, 150)
pdf.set_draw_color(200, 200, 200)
pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
pdf.ln(3)
pdf.cell(0, 5, "本文件为内部使用文件，请勿外传  |  版本 V1.0  |  2026年3月9日", ln=True, align="C")

output_path = "/root/claude-chat/TikTok跨境电商专线-用户使用须知V1.0.pdf"
pdf.output(output_path)
print(f"PDF saved to: {output_path}")
