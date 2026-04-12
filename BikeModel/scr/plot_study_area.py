import pandas as pd
import numpy as np
import h3
import folium
from pathlib import Path

# 配置
FLOW_FINAL_DIR = Path("data/flow_final")
OUTPUT_FILE = "outputs/study_area_map.html"

def get_h3_center(h3_id):
    try:
        return h3.cell_to_latlng(h3_id)
    except:
        return h3.h3_to_geo(h3_id)

def get_h3_polygon(h3_id):
    try:
        boundary = h3.cell_to_boundary(h3_id)
    except:
        boundary = h3.h3_to_geo_boundary(h3_id)
    return [(lat, lng) for lat, lng in boundary]

# 读取所有 H3
print("正在扫描 flow_final 目录...")
all_h3 = set()
for f in FLOW_FINAL_DIR.glob("*.csv"):
    df = pd.read_csv(f)
    if 'h3_id' in df.columns:
        all_h3.update(df['h3_id'].astype(str).str.lower().unique())

print(f"共发现 {len(all_h3)} 个唯一 H3 网格")

# 计算地图中心
lats, lngs = [], []
for h in all_h3:
    lat, lng = get_h3_center(h)
    lats.append(lat)
    lngs.append(lng)
center = [np.mean(lats), np.mean(lngs)]

# 创建地图
m = folium.Map(location=center, zoom_start=11, tiles='OpenStreetMap')

# 绘制所有 H3 网格（蓝色边框，无填充）
for h in all_h3:
    poly = get_h3_polygon(h)
    folium.Polygon(
        locations=poly,
        color='#3388ff',
        weight=1.5,
        fill=False,
        popup=h
    ).add_to(m)

# 保存
Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
m.save(OUTPUT_FILE)
print(f"✅ 研究区域示意图已保存至 {OUTPUT_FILE}")