import h3
import json
from pathlib import Path
from typing import List, Tuple, Dict

# 参数配置
H3_RESOLUTION = 8  # 与03_h3_encoding保持一致

# 坐标系转换

def _transform_latlng(x, y):
    """GCJ-02变换参数"""
    PI = 3.1415926535897932384626
    a = 6378245.0
    ee = 0.006693421622965943

    def transform_lat(x, y):
        ret = -100 + 2*x + 3*y + 0.2*y*y + 0.1*x*y + 0.2*abs(x)
        ret += (20*abs(x) + 20*y) * PI
        return ret * 180 / ((a*(1-ee)) / (1 - ee*(y/180)**2)**1.5 * PI)

    def transform_lng(x, y):
        ret = 300 + x + 2*y + 0.1*x*x + 0.1*x*y + 0.1*abs(x-y)
        ret += (20*x + 40*y) * PI
        return ret * 180 / (a / (1 - ee*(y/180)**2)**0.5 * PI)

    dlat = transform_lat(x-105, y-35)
    dlng = transform_lng(x-105, y-35)
    return dlng, dlat

def gcj2wgs(lng: float, lat: float) -> Tuple[float, float]:
    """
    GCJ-02 → WGS84
    输入: 百度地图/高德地图坐标
    输出: GPS坐标
    """
    if 72.004 <= lng <= 137.8347 and 0.8293 <= lat <= 55.8271:
        dlng, dlat = _transform_latlng(lng, lat)
        return lng - dlng, lat - dlat
    return lng, lat

def wgs2gcj(lng: float, lat: float) -> Tuple[float, float]:
    """
    WGS84 → GCJ-02
    输入: GPS坐标
    输出: 百度地图/高德地图坐标
    """
    if 72.004 <= lng <= 137.8347 and 0.8293 <= lat <= 55.8271:
        dlng, dlat = _transform_latlng(lng, lat)
        return lng + dlng, lat + dlat
    return lng, lat

# 经纬度 → H3 I
def latlng_to_h3(
    lat: float, 
    lng: float, 
    resolution: int = H3_RESOLUTION,
    coordinate_system: str = "wgs84"
) -> str:
    """
    将经纬度转换为H3 ID
    
    参数:
        lat: 纬度
        lng: 经度
        resolution: H3分辨率 (默认9)
        coordinate_system: "wgs84" (GPS) 或 "gcj02" (高德/百度)
    
    返回: H3 ID字符串
    """
    # 如果输入是GCJ-02，先转到WGS84
    if coordinate_system.lower() == "gcj02":
        lng, lat = gcj2wgs(lng, lat)
    
    # 转换为H3
    h3_id = h3.latlng_to_cell(lat, lng, resolution)
    return h3_id

def batch_latlng_to_h3(
    latlng_list: List[Tuple[float, float]], 
    resolution: int = H3_RESOLUTION,
    coordinate_system: str = "wgs84"
) -> List[str]:
    """
    批量转换经纬度列表为H3 ID
    
    参数:
        latlng_list: [(lat, lng), (lat, lng), ...] 列表
        resolution: H3分辨率
        coordinate_system: 坐标系类型
    
    返回: [h3_id1, h3_id2, ...] 列表
    """
    return [
        latlng_to_h3(lat, lng, resolution, coordinate_system)
        for lat, lng in latlng_list
    ]

# H3 ID + 半径 → 圆内所有H3 ID
def h3_disk(
    center_h3: str, 
    radius_km: float,
    resolution: int = H3_RESOLUTION
) -> List[str]:
    """
    根据中心H3 ID和半径，获取圆内所有H3 ID
    
    参数:
        center_h3: 中心H3 ID
        radius_km: 半径（公里）
        resolution: H3分辨率（必须与center_h3一致或用于标准化）
    
    返回: 圆内所有H3 ID列表
    
    注意: H3网格距离是离散的，此函数返回grid_disk的结果
          如果需要精确的地理距离过滤，需要额外计算
    """
    # 获取中心点坐标
    center_lat, center_lng = h3.cell_to_latlng(center_h3)
    
    # 估算H3 ring半径 (粗略转换)
    cell_edge_km = 0.461  # resolution 8 的边长（单位：km）
    
    # 计算应该查询的ring数量
    max_k = max(1, int(radius_km / cell_edge_km) + 1)
    
    # 使用grid_disk获取距离<=k的所有hexagon
    h3_ids = h3.grid_disk(center_h3, max_k)
    
    # 如果需要精确的距离过滤，可以进一步计算
    filtered_h3_ids = []
    for h3_id in h3_ids:
        lat, lng = h3.cell_to_latlng(h3_id)
        # 使用简单的欧几里得距离（对于小范围足够精确）
        # 更精确的做法需要使用Haversine公式
        dist = ((lat - center_lat)**2 + (lng - center_lng)**2)**0.5 * 111  # 粗略转为km
        if dist <= radius_km:
            filtered_h3_ids.append(h3_id)
    
    return filtered_h3_ids

def h3_polygon(
    center_h3: str,
    radius_km: float,
    output_format: str = "list"
) -> Dict | List[str]:
    """
    获取圆形区域内的H3网格，输出为JSON兼容格式
    
    参数:
        center_h3: 中心H3 ID
        radius_km: 半径（公里）
        output_format: "list" (H3 ID列表) 或 "dict" (可用于prediction_target)
    
    返回: H3 ID列表或字典
    """
    h3_ids = h3_disk(center_h3, radius_km)
    
    if output_format == "dict":
        return {
            "center_h3": center_h3,
            "radius_km": radius_km,
            "count": len(h3_ids),
            "region_h3_list": h3_ids
        }
    return h3_ids

# JSON输出工具
def save_h3_region_to_json(
    h3_ids: List[str],
    target_date: str,
    output_path: str = "data/prediction_target.json",
    holiday: int = 0,
    is_preholiday: int = 0
):
    """
    将H3 ID列表保存为prediction_target.json格式
    
    参数:
        h3_ids: H3 ID列表
        target_date: 目标日期 (YYYY-MM-DD)
        output_path: 输出文件路径
        holiday: 是否为假期 (0或1，默认0)
        is_preholiday: 是否为前假日 (0或1，默认0)
    """
    config = {
        "target_date": target_date,
        "holiday": holiday,
        "is_preholiday": is_preholiday,
        "region_h3_list": h3_ids
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 已保存至 {output_path}")
    print(f"   目标日期: {target_date} | Holiday: {holiday} | Pre-holiday: {is_preholiday}")
    print(f"   包含 {len(h3_ids)} 个H3网格")

# CLI工具
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="H3网格工具")
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 经纬度转H3
    parser_latlng = subparsers.add_parser("latlng2h3", help="经纬度转H3 ID")
    parser_latlng.add_argument("lat", type=float, help="纬度")
    parser_latlng.add_argument("lng", type=float, help="经度")
    parser_latlng.add_argument("--sys", default="wgs84", help="坐标系 (wgs84/gcj02)")
    parser_latlng.add_argument("--res", type=int, default=H3_RESOLUTION, help="H3分辨率")
    
    # H3+半径获取区域
    parser_disk = subparsers.add_parser("h3disk", help="H3 ID + 半径获取圆形区域")
    parser_disk.add_argument("h3_id", help="中心H3 ID")
    parser_disk.add_argument("radius", type=float, help="半径(km)")
    parser_disk.add_argument("--date", type=str, help="目标日期，用于生成prediction_target.json")
    parser_disk.add_argument("--output", type=str, help="输出文件路径")
    
    args = parser.parse_args()
    
    if args.command == "latlng2h3":
        h3_id = latlng_to_h3(args.lat, args.lng, args.res, args.sys)
        print(f"H3 ID: {h3_id}")
    
    elif args.command == "h3disk":
        h3_ids = h3_disk(args.h3_id, args.radius)
        print(f"✅ 找到 {len(h3_ids)} 个H3网格")
        print(f"H3 IDs: {h3_ids}")
        
        if args.date:
            output_path = args.output or "data/prediction_target.json"
            save_h3_region_to_json(h3_ids, args.date, output_path)
    
    else:
        parser.print_help()
