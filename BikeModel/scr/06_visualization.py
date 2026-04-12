import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import h3
from shapely.geometry import Polygon
import geopandas as gpd
import requests

# 配置
INPUT_DIR = Path("data/second_preprocessed")
OUTPUT_MAP_DIR = Path("output_maps")
SZ_CENTER = [22.5431, 114.0579]
DEFAULT_ZOOM = 11

SZ_GEOJSON_URL = "https://geo.datav.aliyun.com/areas_v3/bound/440300_full.json"
SZ_GEOJSON_LOCAL = Path("data/sz_boundary.geojson")

TIDE_COLORS = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]
ST_COLORS = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]


def download_sz_boundary():
    if SZ_GEOJSON_LOCAL.exists():
        print(f"✔ 使用本地边界文件: {SZ_GEOJSON_LOCAL}")
        return SZ_GEOJSON_LOCAL
    print("📥 下载深圳市边界...")
    try:
        resp = requests.get(SZ_GEOJSON_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        for feature in data["features"]:
            if feature["properties"]["adcode"] == "440300":
                SZ_GEOJSON_LOCAL.parent.mkdir(parents=True, exist_ok=True)
                with open(SZ_GEOJSON_LOCAL, "w", encoding="utf-8") as f:
                    json.dump(feature, f, ensure_ascii=False)
                print(f"✔ 边界已保存")
                return SZ_GEOJSON_LOCAL
        return None
    except Exception as e:
        print(f"⚠️ 下载失败: {e}")
        return None


def h3_to_geojson(h3_ids, max_errors=5):
    """将 H3 索引转换为 GeoJSON（兼容 h3 4.x 和 3.x）"""
    features = []
    errors = 0
    print(f"📦 H3 版本: {h3.__version__}")
    for hid in h3_ids:
        try:
            # h3 4.x 新 API (无 geo_json 参数)
            if hasattr(h3, 'cell_to_boundary'):
                boundary = h3.cell_to_boundary(hid)   # 返回 [(lat, lng), ...]
                # 转换为 GeoJSON 顺序: (lng, lat)
                boundary = [[lng, lat] for lat, lng in boundary]
            # h3 3.x 旧 API
            elif hasattr(h3, 'h3_to_geo_boundary'):
                boundary = h3.h3_to_geo_boundary(hid, geo_json=True)
            else:
                errors += 1
                continue
            if boundary and len(boundary) > 0:
                if boundary[0] != boundary[-1]:
                    boundary.append(boundary[0])
                polygon = Polygon(boundary)
                if polygon.is_valid:
                    features.append({
                        "type": "Feature",
                        "geometry": polygon.__geo_interface__,
                        "properties": {"h3_id": hid}
                    })
        except Exception as e:
            errors += 1
            if errors <= max_errors:
                print(f"⚠️ H3 转换失败 {hid[:10]}...: {e}")
    print(f"✅ 成功转换 {len(features)} / {len(h3_ids)} 个 H3 网格")
    return {"type": "FeatureCollection", "features": features}


def load_time_series_data(date_str, metric):
    file_path = INPUT_DIR / f"{date_str}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"未找到数据文件: {file_path}")
    df = pd.read_csv(file_path)
    if "time_bin" not in df.columns:
        raise ValueError(f"缺少 time_bin 列")
    df["time_bin"] = pd.to_datetime(df["time_bin"], errors="coerce")
    df = df.dropna(subset=["time_bin"])
    if df.empty:
        raise ValueError(f"{date_str} 无有效数据")
    df["slot_str"] = df["time_bin"].dt.strftime("%H:%M:%S")
    time_slots = sorted(df["slot_str"].unique())
    h3_ids = sorted(df["h3_id"].unique())
    if metric not in df.columns:
        if metric == "tide_index" and "d_in" in df.columns and "d_out" in df.columns:
            print("   ⚠️ 重新计算 tide_index...")
            df["tide_index"] = (df["d_in"] - df["d_out"]) / (df["d_in"] + df["d_out"] + 1)
        else:
            raise ValueError(f"缺少 {metric} 列")
    values_matrix = []
    for ts in time_slots:
        sub = df[df["slot_str"] == ts]
        val_dict = dict(zip(sub["h3_id"], sub[metric]))
        row = [val_dict.get(hid, np.nan) for hid in h3_ids]
        values_matrix.append(row)
    global_mean = np.nanmean(values_matrix)
    if np.isnan(global_mean):
        global_mean = 0
    values_matrix = np.nan_to_num(values_matrix, nan=global_mean)
    return time_slots, h3_ids, values_matrix


def generate_html(date_str, metric, time_slots, h3_ids, values_matrix, geojson_data, boundary_geojson=None):
    if metric == "tide_index":
        vmin, vmax = -1.0, 1.0
        colors = TIDE_COLORS
        caption = "潮汐指数 (tide_index)  负→净流出  正→净流入"
    else:
        vmin = float(np.nanmin(values_matrix))
        vmax = float(np.nanmax(values_matrix))
        colors = ST_COLORS
        caption = f"实时库存 S_t (范围: {vmin:.1f} ~ {vmax:.1f})"
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5
    values_json = json.dumps(values_matrix.tolist())
    time_slots_json = json.dumps(time_slots)
    h3_ids_json = json.dumps(h3_ids)
    geojson_str = json.dumps(geojson_data)
    boundary_str = json.dumps(boundary_geojson) if boundary_geojson else "null"
    colors_json = json.dumps(colors)
    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>深圳 {date_str} - {metric}</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script scr="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin:0; padding:0; }}
        #map {{ position: absolute; top:0; bottom:40px; width:100%; }}
        #slider-container {{
            position: absolute; bottom: 10px; left: 50px; right: 50px;
            background: white; padding: 12px 20px; border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3); z-index: 1000;
            text-align: center; font-family: Arial, sans-serif;
        }}
        input {{ width: 80%; margin: 0 10px; }}
        #time-label {{ font-weight: bold; margin-left: 10px; }}
        .info {{
            position: absolute; top: 10px; right: 10px;
            background: white; padding: 6px 12px; border-radius: 5px;
            box-shadow: 0 1px 5px rgba(0,0,0,0.2); z-index: 1000;
            font-family: Arial, sans-serif; font-size: 14px;
        }}
        .legend {{
            position: absolute; bottom: 80px; right: 10px;
            background: white; padding: 8px 12px; border-radius: 5px;
            box-shadow: 0 1px 5px rgba(0,0,0,0.2); z-index: 1000;
            font-size: 12px;
        }}
        .legend-color {{ width: 20px; height: 20px; display: inline-block; margin-right: 5px; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div id="slider-container">
        <span>时间: </span>
        <input type="range" id="time-slider" min="0" max="{len(time_slots)-1}" step="1" value="0">
        <span id="time-label">{time_slots[0]}</span>
    </div>
    <div class="info"><strong>{metric}</strong><br>{caption}</div>
    <div class="legend" id="legend"><strong>图例</strong><br></div>
    <script>
        const timeSlots = {time_slots_json};
        const h3Ids = {h3_ids_json};
        const valuesMatrix = {values_json};
        const geojsonData = {geojson_str};
        const boundaryData = {boundary_str};
        const vmin = {vmin};
        const vmax = {vmax};
        const colors = {colors_json};
        function getColor(value) {{
            if (isNaN(value)) return "#cccccc";
            let t = (value - vmin) / (vmax - vmin);
            t = Math.min(1.0, Math.max(0.0, t));
            const idx = t * (colors.length - 1);
            const i1 = Math.floor(idx);
            const i2 = Math.min(i1 + 1, colors.length - 1);
            return colors[i1];
        }}
        const map = L.map('map').setView({SZ_CENTER}, {DEFAULT_ZOOM});
        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>',
            subdomains: 'abcd', maxZoom: 19
        }}).addTo(map);
        if (boundaryData) {{
            L.geoJSON(boundaryData, {{
                style: {{ color: "#2c3e50", weight: 2, fillOpacity: 0 }}
            }}).addTo(map);
        }}
        const legendDiv = document.getElementById('legend');
        for (let i = 0; i <= 5; i++) {{
            const val = vmin + (vmax - vmin) * i / 5;
            const color = getColor(val);
            legendDiv.innerHTML += `<div><span class="legend-color" style="background:${{color}};"></span> ${{val.toFixed(2)}}</div>`;
        }}
        let currentLayer;
        function updateLayer(index) {{
            const valueMap = {{}};
            for (let i = 0; i < h3Ids.length; i++) valueMap[h3Ids[i]] = valuesMatrix[index][i];
            if (currentLayer) map.removeLayer(currentLayer);
            currentLayer = L.geoJSON(geojsonData, {{
                style: function(feature) {{
                    const val = valueMap[feature.properties.h3_id];
                    return {{ fillColor: getColor(val), color: "black", weight: 0.5, fillOpacity: 0.7 }};
                }},
                onEachFeature: function(feature, layer) {{
                    const val = valueMap[feature.properties.h3_id];
                    layer.bindTooltip(`H3: ${{feature.properties.h3_id}}<br>值: ${{val.toFixed(3)}}`);
                }}
            }}).addTo(map);
        }}
        const slider = document.getElementById('time-slider');
        const timeLabel = document.getElementById('time-label');
        slider.addEventListener('input', function(e) {{
            const idx = parseInt(e.target.value);
            timeLabel.innerText = timeSlots[idx];
            updateLayer(idx);
        }});
        updateLayer(0);
        L.control.scale().addTo(map);
    </script>
</body>
</html>'''
    return html


def process_single_date(date_str, metric, boundary_geojson, h3_cache):
    print(f"\n📅 处理: {date_str}")
    try:
        time_slots, h3_ids, values_matrix = load_time_series_data(date_str, metric)
    except Exception as e:
        print(f"   ❌ 加载失败: {e}")
        return False
    # 使用 h3_ids 的元组作为缓存键（取前100个作为代表，因为所有日期的 h3 集合相同）
    cache_key = tuple(sorted(h3_ids)[:100])
    if cache_key not in h3_cache:
        h3_cache[cache_key] = h3_to_geojson(h3_ids)
    geojson_data = h3_cache[cache_key]
    if not geojson_data["features"]:
        print(f"   ❌ 没有有效的 H3 多边形")
        return False
    html = generate_html(date_str, metric, time_slots, h3_ids, values_matrix, geojson_data, boundary_geojson)
    metric_dir = OUTPUT_MAP_DIR / metric
    metric_dir.mkdir(parents=True, exist_ok=True)
    out_file = metric_dir / f"{date_str}.html"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"   ✅ 已保存: {out_file}")
    return True


def generate_index_page(dates, metrics):
    html = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>深圳共享单车可视化 - 所有日期</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #2c3e50; }
        .metric-section { margin-bottom: 30px; }
        .metric-title { background: #3498db; color: white; padding: 10px; border-radius: 5px; }
        .date-list { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 15px; }
        .date-card { background: #ecf0f1; padding: 10px 15px; border-radius: 5px; text-decoration: none; color: #2c3e50; transition: 0.3s; }
        .date-card:hover { background: #bdc3c7; transform: translateY(-2px); }
        .stats { margin-top: 20px; padding: 10px; background: #e8f4f8; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>🚲 深圳共享单车时空可视化</h1>
'''
    for metric in metrics:
        metric_name = "潮汐指数 (tide_index)" if metric == "tide_index" else "实时库存 (S_t)"
        html += f'''
    <div class="metric-section">
        <div class="metric-title">{metric_name}</div>
        <div class="date-list">
'''
        for date in dates:
            html += f'            <a href="{metric}/{date}.html" class="date-card">{date}</a>\n'
        html += f'''
        </div>
    </div>
'''
    html += f'''
    <div class="stats">
        <strong>📊 统计信息</strong><br>
        总日期数: {len(dates)} 天<br>
        生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    </div>
</body>
</html>'''
    index_file = OUTPUT_MAP_DIR / "index.html"
    with open(index_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n📑 索引页面已保存: {index_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="both", choices=["tide_index", "S_t", "both"])
    parser.add_argument("--date", help="指定单个日期")
    args = parser.parse_args()

    files = sorted(INPUT_DIR.glob("*.csv"))
    if not files:
        print(f"❌ 在 {INPUT_DIR} 中没有找到数据文件")
        sys.exit(1)
    dates = [f.stem for f in files]
    print(f"📊 找到 {len(dates)} 个日期的数据")
    if args.date:
        if args.date not in dates:
            print(f"❌ 日期 {args.date} 不在可用数据中")
            sys.exit(1)
        dates = [args.date]

    metrics = ["tide_index", "S_t"] if args.metric == "both" else [args.metric]

    boundary_file = download_sz_boundary()
    boundary_geojson = None
    if boundary_file:
        try:
            boundary_geojson = json.loads(gpd.read_file(boundary_file).to_json())
        except Exception as e:
            print(f"⚠️ 加载边界失败: {e}")

    for metric in metrics:
        print(f"\n{'='*50}")
        print(f"🎨 生成 {metric} 可视化地图")
        print(f"{'='*50}")
        success = 0
        h3_cache = {}
        for i, date in enumerate(dates, 1):
            print(f"\n[{i}/{len(dates)}]", end=" ")
            if process_single_date(date, metric, boundary_geojson, h3_cache):
                success += 1
        print(f"\n📈 {metric} 完成: {success}/{len(dates)}")

    generate_index_page(dates, metrics)
    print(f"\n🎉 完成！打开 {OUTPUT_MAP_DIR / 'index.html'} 查看")


if __name__ == "__main__":
    main()