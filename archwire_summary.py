from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1080, 1920
BG_COLOR = (15, 23, 42)
ACCENT = (56, 189, 248)
WHITE = (255, 255, 255)
GRAY = (148, 163, 184)
CARD_BG = (30, 41, 59)

img = Image.new('RGB', (WIDTH, HEIGHT), BG_COLOR)
draw = ImageDraw.Draw(img)

title_font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc", 48)
subtitle_font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc", 32)
body_font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 26)
small_font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 22)

archwires = [
    {
        "name": "镍钛弓丝 (NiTi)",
        "color": (34, 197, 94),
        "points": [
            "超弹性，形状记忆效应",
            "持续轻柔力量，适合早期排齐",
            "弹性范围大，适合严重错位牙",
        ]
    },
    {
        "name": "热激活镍钛弓丝 (Thermal NiTi)",
        "color": (251, 146, 60),
        "points": [
            "受口腔温度激活后恢复形状",
            "力量更柔和、更持久",
            "患者舒适度更高，适合初期矫治",
        ]
    },
    {
        "name": "不锈钢弓丝 (SS)",
        "color": (168, 162, 158),
        "points": [
            "刚性强，精确控制牙齿移动",
            "适合中后期精细调整和关闭间隙",
            "可弯制各种曲（摇椅弓、TPA等）",
        ]
    },
    {
        "name": "β-钛弓丝 (TMA)",
        "color": (196, 181, 253),
        "points": [
            "弹性介于NiTi和SS之间",
            "可成型性好，适合精细弯曲",
            "适合中期过渡和个性化调整",
        ]
    },
    {
        "name": "钴铬弓丝 (CoCr/Elgiloy)",
        "color": (251, 207, 74),
        "points": [
            "硬度高，抗疲劳性好",
            "热处理后可增加刚性",
            "适合后期维持和稳定阶段",
        ]
    },
    {
        "name": "多股绞丝弓丝 (Braided)",
        "color": (248, 113, 113),
        "points": [
            "柔韧性极佳，力量轻柔",
            "适合早期严重拥挤的排齐",
            "多股结构提供持续弹性",
        ]
    },
]

y = 60
draw.rectangle([(40, y), (WIDTH-40, y+4)], fill=ACCENT)
y += 25

draw.text((WIDTH//2, y), "口腔正畸学", font=title_font, fill=WHITE, anchor="mt")
y += 65
draw.text((WIDTH//2, y), "各类弓丝的作用总结", font=subtitle_font, fill=ACCENT, anchor="mt")
y += 55
draw.rectangle([(200, y), (WIDTH-200, y+2)], fill=GRAY)
y += 30

card_margin = 40

for wire in archwires:
    card_height = 30 + len(wire["points"]) * 42 + 20

    draw.rounded_rectangle(
        [(card_margin, y), (WIDTH - card_margin, y + card_height)],
        radius=12, fill=CARD_BG,
    )
    draw.rounded_rectangle(
        [(card_margin, y), (card_margin + 6, y + card_height)],
        radius=3, fill=wire["color"],
    )

    name_y = y + 18
    draw.text((card_margin + 24, name_y), wire["name"], font=subtitle_font, fill=wire["color"])

    point_y = name_y + 45
    for point in wire["points"]:
        draw.ellipse(
            [(card_margin + 30, point_y + 8), (card_margin + 38, point_y + 16)],
            fill=wire["color"]
        )
        draw.text((card_margin + 50, point_y), point, font=body_font, fill=WHITE)
        point_y += 42

    y += card_height + 16

y += 10
draw.rectangle([(200, y), (WIDTH-200, y+2)], fill=GRAY)
y += 20
draw.text((WIDTH//2, y), "弓丝选择原则：根据矫治阶段、牙齿移动需求", font=small_font, fill=GRAY, anchor="mt")
y += 32
draw.text((WIDTH//2, y), "及患者个体情况综合考虑", font=small_font, fill=GRAY, anchor="mt")
y += 45
draw.text((WIDTH//2, y), "By 宝宝 | 2026.03.07", font=small_font, fill=(100, 116, 139), anchor="mt")

draw.rectangle([(40, HEIGHT-60), (WIDTH-40, HEIGHT-56)], fill=ACCENT)

output_path = "/root/claude-chat/archwire_summary.png"
img.save(output_path, quality=95)
print(f"图片已保存到: {output_path}")
print(f"尺寸: {WIDTH}x{HEIGHT}")
