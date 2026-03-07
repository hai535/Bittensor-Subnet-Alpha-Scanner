#!/usr/bin/env python3
"""槟城15天天气预报查询"""

import requests
import json
from datetime import datetime

# 槟城坐标
LAT = 5.4164
LON = 100.3327

def get_weather():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,weathercode,wind_speed_10m_max",
        "timezone": "Asia/Kuala_Lumpur",
        "forecast_days": 16
    }

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    # 天气代码映射
    wmo_codes = {
        0: "晴天", 1: "大部晴", 2: "多云", 3: "阴天",
        45: "雾", 48: "雾凇",
        51: "小毛毛雨", 53: "中毛毛雨", 55: "大毛毛雨",
        61: "小雨", 63: "中雨", 65: "大雨",
        71: "小雪", 73: "中雪", 75: "大雪",
        80: "小阵雨", 81: "中阵雨", 82: "大阵雨",
        95: "雷暴", 96: "雷暴+冰雹", 99: "强雷暴+冰雹"
    }

    daily = data["daily"]

    print("=" * 65)
    print(f"{'槟城 (Penang) 未来15天天气预报':^55}")
    print(f"{'查询时间: ' + datetime.now().strftime('%Y-%m-%d %H:%M'):^55}")
    print("=" * 65)
    print(f"{'日期':<12} {'天气':<8} {'温度°C':<12} {'降雨mm':<8} {'降雨%':<6} {'风km/h':<6}")
    print("-" * 65)

    good_days = []

    for i in range(len(daily["time"])):
        date = daily["time"][i]
        code = daily["weathercode"][i]
        weather = wmo_codes.get(code, f"未知({code})")
        t_max = daily["temperature_2m_max"][i]
        t_min = daily["temperature_2m_min"][i]
        rain = daily["precipitation_sum"][i]
        rain_prob = daily["precipitation_probability_max"][i]
        wind = daily["wind_speed_10m_max"][i]

        # 判断星期
        dt = datetime.strptime(date, "%Y-%m-%d")
        weekday = ["周一","周二","周三","周四","周五","周六","周日"][dt.weekday()]

        # 标记适合出游的日子
        is_good = rain_prob <= 30 and code <= 2
        marker = " *" if is_good else ""
        if is_good:
            good_days.append(f"{date} {weekday}")

        print(f"{date} {weekday:<4} {weather:<8} {t_min:.0f}-{t_max:.0f}°C    {rain:<8.1f} {rain_prob:<6}% {wind:<6.0f}{marker}")

    print("-" * 65)
    print("* 标记 = 适合出游（降雨概率≤30%，晴到多云）")

    if good_days:
        print(f"\n推荐出游日：{'、'.join(good_days)}")
    else:
        print("\n未来15天降雨概率都偏高，槟城属热带，建议备伞随时出行~")

    print()

if __name__ == "__main__":
    get_weather()
