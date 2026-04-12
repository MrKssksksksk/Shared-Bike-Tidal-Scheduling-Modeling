"""
12_transport_solution.py (多车时间窗口版 - 输出优化)
- 精简控制台输出，避免警告刷屏
- 展示每次尝试的最长路线时间
- 汇总强制分配站点信息
"""

import pandas as pd
import numpy as np
import h3
from pathlib import Path
from typing import List, Tuple, Dict, Set
import sys
from functools import lru_cache
from math import radians, sin, cos, sqrt, atan2

# 配置
SCHEDULE_TASKS_PATH = "data/schedule_tasks.csv"
OUTPUT_PATH = "data/transport_routes.csv"
VEHICLE_CAPACITY = 36                 # 每辆车最多装载车辆数
VEHICLE_SPEED_KMH = 35                 # 行驶速度（km/h）
LOAD_UNLOAD_TIME_PER_BIKE_MIN = 0.5    # 每辆车装卸时间（分钟）
TIME_WINDOW_HOURS = 1.0                # 时间窗口（小时）
MIN_SCHEDULE_QUANTITY = 5            # 最小调度数量，过滤微小变动
MAX_VEHICLES = 40                   # 最大尝试车辆数
DEFAULT_WAREHOUSE_H3 = None         # 默认仓库H3，如果为None则自动计算中心点作为仓库

CACHE_SIZE = 10000

# 经纬度缓存与距离
@lru_cache(maxsize=CACHE_SIZE)
def get_latlng(cell: str) -> Tuple[float, float]:
    try:
        return h3.cell_to_latlng(cell)
    except AttributeError:
        return h3.h3_to_geo(cell)

def haversine_distance(h3_a: str, h3_b: str) -> float:
    lat1, lng1 = get_latlng(h3_a)
    lat2, lng2 = get_latlng(h3_b)
    R = 6371.0
    lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlng/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def safe_latlng_to_cell(lat: float, lng: float, res: int) -> str:
    try:
        return h3.latlng_to_cell(lat, lng, res)
    except AttributeError:
        return h3.geo_to_h3(lat, lng, res)

# 距离矩阵构建
def build_distance_matrix(all_h3: List[str]) -> np.ndarray:
    n = len(all_h3)
    dist = np.zeros((n, n))
    total_pairs = n * (n - 1) // 2
    cnt = 0
    for i in range(n):
        for j in range(i+1, n):
            d = haversine_distance(all_h3[i], all_h3[j])
            dist[i, j] = d
            dist[j, i] = d
            cnt += 1
            print(f"    距离矩阵: {cnt}/{total_pairs} ({100*cnt/total_pairs:.1f}%)", end='\r')
    print("")
    return dist

# 站点拆分
def split_large_demands(tasks_df: pd.DataFrame, capacity: int) -> pd.DataFrame:
    new_rows = []
    for _, row in tasks_df.iterrows():
        qty = row['net_change']
        h3_id = row['h3_id']
        if abs(qty) <= capacity:
            new_rows.append(row.to_dict())
        else:
            num_full = abs(qty) // capacity
            remainder = abs(qty) % capacity
            sign = 1 if qty > 0 else -1
            for _ in range(num_full):
                new_rows.append({'h3_id': h3_id, 'net_change': sign * capacity})
            if remainder > 0:
                new_rows.append({'h3_id': h3_id, 'net_change': sign * remainder})
    return pd.DataFrame(new_rows)

# 路线时间计算
def compute_route_time(route_indices: List[int], dist: np.ndarray,
                       sites: List[Dict]) -> float:
    if len(route_indices) < 2:
        return 0.0
    travel_dist = sum(dist[route_indices[k], route_indices[k+1]] for k in range(len(route_indices)-1))
    travel_time = travel_dist / VEHICLE_SPEED_KMH
    total_load_unload = sum(abs(sites[i-1]['demand']) for i in route_indices if i != 0)
    load_unload_time = (total_load_unload * LOAD_UNLOAD_TIME_PER_BIKE_MIN) / 60.0
    return travel_time + load_unload_time

# 多车贪心分配
def assign_sites_to_vehicles_greedy(sites: List[Dict], dist: np.ndarray,
                                    capacity: int, time_window: float,
                                    warehouse_idx: int = 0) -> Tuple[List[List[int]], bool, Set[str]]:
    n_sites = len(sites)
    unvisited = set(range(1, n_sites + 1))
    routes = []
    forced_sites = set()   # 记录被强制分配的站点h3

    while unvisited:
        route = [warehouse_idx]
        load = 0
        current = warehouse_idx

        while unvisited:
            best = None
            best_dist = float('inf')
            best_new_load = None
            for nxt in list(unvisited):
                demand = sites[nxt-1]['demand']
                new_load = load + demand
                if 0 <= new_load <= capacity:
                    d = dist[current, nxt]
                    if d < best_dist:
                        best_dist = d
                        best = nxt
                        best_new_load = new_load
            if best is None:
                break

            temp_route = route + [best, warehouse_idx]
            temp_time = compute_route_time(temp_route, dist, sites)
            if temp_time > time_window:
                break

            route.append(best)
            load = best_new_load
            current = best
            unvisited.remove(best)

        if len(route) == 1:
            # 强制分配最近的一个站点
            best = min(unvisited, key=lambda x: dist[current, x])
            route.append(best)
            route.append(warehouse_idx)
            forced_sites.add(sites[best-1]['h3'])
            unvisited.remove(best)
        else:
            route.append(warehouse_idx)

        routes.append(route)

    all_within_window = all(
        compute_route_time(r, dist, sites) <= time_window for r in routes
    )
    return routes, all_within_window, forced_sites

# 主求解函数
def solve_for_schedule_time(tasks_df: pd.DataFrame, warehouse_h3: str,
                            capacity: int, time_window: float) -> List[Dict]:
    tasks_df = split_large_demands(tasks_df, capacity)
    if tasks_df.empty:
        return []

    sites = [{'h3': row['h3_id'], 'demand': row['net_change']} for _, row in tasks_df.iterrows()]
    all_h3 = [warehouse_h3] + [s['h3'] for s in sites]
    dist = build_distance_matrix(all_h3)

    best_routes = None
    success = False
    forced_sites_final = set()

    print(f"  站点总数: {len(sites)} (拆分后)")

    for num_vehicles in range(1, MAX_VEHICLES + 1):
        routes, all_ok, forced = assign_sites_to_vehicles_greedy(
            sites, dist, capacity, time_window
        )
        actual_vehicles = len(routes)
        max_route_time = max(compute_route_time(r, dist, sites) for r in routes) if routes else 0
        status = "✓" if all_ok else "✗"
        print(f"    尝试 {num_vehicles:2d} 辆车 -> 实际 {actual_vehicles:2d} 辆, 满足窗口: {status} (最长 {max_route_time:.2f}h)")

        if all_ok:
            best_routes = routes
            forced_sites_final = forced
            success = True
            break
        else:
            # 保存最后一次尝试的结果以备全部失败时使用
            best_routes = routes
            forced_sites_final = forced

    if not success:
        print(f"    ⚠️ 在 {MAX_VEHICLES} 辆车内无法完全满足时间窗口，使用最后一次尝试结果")
        if forced_sites_final:
            print(f"    强制分配站点 (共 {len(forced_sites_final)} 个): {', '.join(sorted(forced_sites_final))}")
    else:
        if forced_sites_final:
            print(f"    ℹ️ 强制分配站点 (共 {len(forced_sites_final)} 个): {', '.join(sorted(forced_sites_final))}")

    # 构建输出
    result = []
    for veh_idx, route_indices in enumerate(best_routes):
        route_h3 = [all_h3[i] for i in route_indices]
        total_dist = sum(dist[route_indices[k], route_indices[k+1]] for k in range(len(route_indices)-1))
        total_time = compute_route_time(route_indices, dist, sites)

        load = 0
        details = []
        for k, idx in enumerate(route_indices):
            if idx == 0:
                if k == 0:
                    details.append("仓库出发")
                else:
                    details.append("返回仓库")
            else:
                qty = sites[idx-1]['demand']
                load += qty
                details.append(f"{all_h3[idx]} 装卸 {qty:+.1f} 辆 (载货 {load:.1f})")

        result.append({
            'vehicle_id': veh_idx + 1,
            'route': ' -> '.join(route_h3),
            'distance_km': round(total_dist, 2),
            'time_hours': round(total_time, 2),
            'details': ' | '.join(details)
        })
    return result

# 主程序
def main():
    print(f"读取调度任务: {SCHEDULE_TASKS_PATH}")
    tasks_df = pd.read_csv(SCHEDULE_TASKS_PATH)
    if tasks_df.empty:
        print("无调度任务，退出")
        return

    # 仓库
    if DEFAULT_WAREHOUSE_H3:
        warehouse = DEFAULT_WAREHOUSE_H3
    else:
        lats, lngs = [], []
        for h in tasks_df['h3_id'].unique():
            try:
                lat, lng = get_latlng(h)
                lats.append(lat)
                lngs.append(lng)
            except Exception:
                continue
        if not lats:
            print("无有效坐标，退出")
            return
        center_lat = np.mean(lats)
        center_lng = np.mean(lngs)
        warehouse = safe_latlng_to_cell(center_lat, center_lng, 8)
    print(f"仓库 H3: {warehouse}")

    all_routes = []
    schedule_times = tasks_df['schedule_time'].unique()
    for idx, st in enumerate(schedule_times, 1):
        group = tasks_df[tasks_df['schedule_time'] == st]
        print(f"\n[{idx}/{len(schedule_times)}] 调度时刻 {st}，原始 {len(group)} 个站点")

        group = group[group['net_change'].abs() >= MIN_SCHEDULE_QUANTITY]
        if group.empty:
            print("  过滤后无任务，跳过")
            continue

        valid_rows = []
        for _, row in group.iterrows():
            try:
                get_latlng(row['h3_id'])
                valid_rows.append(row)
            except Exception:
                print(f"  跳过非法站点 {row['h3_id']}")
        if not valid_rows:
            continue
        group = pd.DataFrame(valid_rows)

        routes_info = solve_for_schedule_time(group, warehouse, VEHICLE_CAPACITY, TIME_WINDOW_HOURS)
        for info in routes_info:
            info['schedule_time'] = st
            all_routes.append(info)

        print(f"  ✅ 完成，使用 {len(routes_info)} 辆车")

    if all_routes:
        routes_df = pd.DataFrame(all_routes)
        routes_df = routes_df[['schedule_time', 'vehicle_id', 'route', 'distance_km', 'time_hours', 'details']]
        routes_df.to_csv(OUTPUT_PATH, index=False)
        print(f"\n✅ 调度路径已保存至 {OUTPUT_PATH}")
        print(routes_df[['schedule_time', 'vehicle_id', 'distance_km', 'time_hours']].to_string(index=False))
    else:
        print("⚠️ 未生成任何有效路径。")


if __name__ == "__main__":
    main()