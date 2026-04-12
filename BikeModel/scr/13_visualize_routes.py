import pandas as pd
import numpy as np
import h3
import folium
from folium.features import GeoJsonTooltip
from pathlib import Path
from typing import List, Tuple, Dict, Set
import sys
import colorsys
import json

# 配置
PREDICTION_CSV = "data/prediction.csv"
ROUTES_CSV = "data/transport_routes.csv"
OUTPUT_DIR = "outputs/route_maps"
H3_RESOLUTION = 8

# 热力图颜色配置：潮汐指数从负到正（蓝 -> 白 -> 红）
def tide_color(tide_index: float) -> str:
    """将潮汐指数映射到颜色：负值(流入>流出)蓝色，正值红色，零值白色"""
    # 限制在 [-1, 1] 范围内
    t = max(-1.0, min(1.0, tide_index))
    if t < 0:
        # 蓝色：负值越强蓝色越深
        intensity = int(255 * (1 + t))  # t=-1 -> 0, t=0 -> 255
        return f'#{0:02x}{0:02x}{intensity:02x}'
    else:
        # 红色：正值越强红色越深
        intensity = int(255 * (1 - t))  # t=1 -> 0, t=0 -> 255
        return f'#{intensity:02x}{0:02x}{0:02x}'

# 车辆颜色生成（30+ 种不同颜色）
def generate_vehicle_colors(n: int) -> List[str]:
    colors = []
    for i in range(n):
        hue = (i * 0.618033988749895) % 1.0  # 黄金比例分布
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append(f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}')
    return colors

# 预生成足够颜色
VEHICLE_COLORS = generate_vehicle_colors(50)

# H3 辅助函数 
def get_h3_center(h3_id: str) -> Tuple[float, float]:
    try:
        return h3.cell_to_latlng(h3_id)
    except AttributeError:
        return h3.h3_to_geo(h3_id)

def get_h3_polygon(h3_id: str) -> List[Tuple[float, float]]:
    try:
        boundary = h3.cell_to_boundary(h3_id)
    except AttributeError:
        boundary = h3.h3_to_geo_boundary(h3_id)
    return [(lat, lng) for lat, lng in boundary]

# 数据加载
def load_prediction_data() -> Tuple[pd.DataFrame, Dict[int, Dict[str, Dict]]]:
    """加载预测数据，并按 time_slot 组织成字典便于快速查询"""
    print(f"读取预测数据: {PREDICTION_CSV}")
    df = pd.read_csv(PREDICTION_CSV)
    # 确保列名小写
    df.columns = df.columns.str.lower()
    df['h3_id'] = df['h3_id'].astype(str).str.lower()
    df['time_slot'] = df['time_slot'].astype(int)
    # 按 time_slot 分组
    slot_data = {}
    for slot, group in df.groupby('time_slot'):
        slot_dict = {}
        for _, row in group.iterrows():
            h = row['h3_id']
            slot_dict[h] = {
                'D_in': row['d_in'],
                'D_out': row['d_out'],
                'S_t': row['s_t'] if not pd.isna(row['s_t']) else None
            }
        slot_data[slot] = slot_dict
    return df, slot_data

def load_routes_data() -> pd.DataFrame:
    print(f"读取路线数据: {ROUTES_CSV}")
    return pd.read_csv(ROUTES_CSV)

# 地图生成
def create_map_for_schedule(schedule_time: int, routes_group: pd.DataFrame,
                            all_h3_in_region: Set[str], slot_data: Dict[str, Dict],
                            warehouse: str) -> folium.Map:
    """为单个调度时刻生成地图"""
    # 获取对应 time_slot 的预测数据（调度时刻即 time_slot）
    pred_dict = slot_data.get(schedule_time, {})
    
    # 收集所有需要显示的 H3（全区域 + 调度涉及站点）
    scheduled_h3 = set()
    for route_str in routes_group['route']:
        for h in route_str.split(' -> '):
            scheduled_h3.add(h.strip())
    all_h3 = all_h3_in_region.union(scheduled_h3)
    
    # 计算中心
    lats, lngs = [], []
    for h in all_h3:
        lat, lng = get_h3_center(h)
        lats.append(lat)
        lngs.append(lng)
    center_lat = np.mean(lats)
    center_lng = np.mean(lngs)
    
    m = folium.Map(location=[center_lat, center_lng], zoom_start=12, tiles='OpenStreetMap')
    
    # 1. 绘制全区域 H3 六边形（潮汐指数热力图）
    for h in all_h3:
        poly = get_h3_polygon(h)
        data = pred_dict.get(h, {})
        d_in = data.get('D_in', 0)
        d_out = data.get('D_out', 0)
        s_t = data.get('S_t', 'N/A')
        tide = (d_in - d_out) / (d_in + d_out + 1e-6)
        tide_str = f"{tide:.3f}"
        
        # 边框样式：调度站点用粗深色边框，其他用极淡边框
        if h in scheduled_h3:
            border_color = '#000000'
            border_width = 2
            fill_opacity = 0.6
        else:
            border_color = '#cccccc'
            border_width = 0.5
            fill_opacity = 0.4
        
        fill_color = tide_color(tide)
        
        popup_text = f"""
        <b>H3:</b> {h}<br>
        <b>D_in:</b> {d_in:.1f}<br>
        <b>D_out:</b> {d_out:.1f}<br>
        <b>S_t:</b> {s_t}<br>
        <b>Tide Index:</b> {tide_str}
        """
        
        folium.Polygon(
            locations=poly,
            color=border_color,
            weight=border_width,
            fill=True,
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            popup=folium.Popup(popup_text, max_width=250)
        ).add_to(m)
    
    # 2. 绘制仓库（红色五角星）
    warehouse_lat, warehouse_lng = get_h3_center(warehouse)
    folium.Marker(
        location=[warehouse_lat, warehouse_lng],
        popup=f"仓库: {warehouse}",
        icon=folium.Icon(color='red', icon='star', prefix='fa')
    ).add_to(m)
    
    # 3. 绘制调度路线
    max_vehicle = routes_group['vehicle_id'].max()
    for _, row in routes_group.iterrows():
        vehicle_id = row['vehicle_id']
        route_str = row['route']
        h3_list = [h.strip() for h in route_str.split(' -> ')]
        points = [get_h3_center(h) for h in h3_list]
        
        color = VEHICLE_COLORS[(vehicle_id - 1) % len(VEHICLE_COLORS)]
        
        folium.PolyLine(
            locations=points,
            color=color,
            weight=4,
            opacity=0.9,
            popup=f"车辆 {vehicle_id} | 距离: {row['distance_km']} km | 时间: {row['time_hours']} h"
        ).add_to(m)
        
        # 在站点添加小圆点（可选，已用多边形边框强调，这里可省略避免过密）
        # 但为了路线清晰，在站点中心加小圆点
        for h in h3_list[1:-1]:  # 跳过首尾仓库
            lat, lng = get_h3_center(h)
            folium.CircleMarker(
                location=[lat, lng],
                radius=4,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                popup=f"{h}<br>车辆 {vehicle_id}"
            ).add_to(m)
    
    return m

def main():
    # 读取数据
    pred_df, slot_data = load_prediction_data()
    routes_df = load_routes_data()
    
    # 获取全区域 H3（来自 prediction.csv）
    all_h3_region = set(pred_df['h3_id'].unique())
    
    # 推断仓库（取所有路线起点众数）
    warehouse_candidates = []
    for route_str in routes_df['route']:
        parts = route_str.split(' -> ')
        if parts:
            warehouse_candidates.append(parts[0].strip())
    from collections import Counter
    warehouse = Counter(warehouse_candidates).most_common(1)[0][0]
    print(f"仓库 H3: {warehouse}")
    
    # 创建输出目录
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # 按调度时刻分组生成地图
    for schedule_time, group in routes_df.groupby('schedule_time'):
        print(f"生成调度时刻 {schedule_time} 的地图...")
        m = create_map_for_schedule(schedule_time, group, all_h3_region, slot_data, warehouse)
        out_file = Path(OUTPUT_DIR) / f"schedule_{schedule_time}.html"
        m.save(str(out_file))
        print(f"  已保存: {out_file}")
    
    print(f"\n✅ 所有地图已生成，保存在 {OUTPUT_DIR} 目录下。")

if __name__ == "__main__":
    main()