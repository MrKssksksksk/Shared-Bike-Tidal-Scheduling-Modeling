import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys
import h3   # 新增：用于计算网格面积

# 路径 & 参数
INPUT_DIR = Path("data/flow_final")
OUTPUT_DIR = Path("data/second_preprocessed")
CAPACITY_FILE = Path("data/capacity.csv")

WARMUP_DAYS = 14
INIT_S0_RATIO = 0.5

# 低流量过滤阈值
MIN_TOTAL_FLOW = 5

# 容量下限修正参数（基于面积）
# 每辆共享单车平均占地面积（平方米），可根据实际情况调整
AREA_PER_BIKE_M2 = 3.0
# 最小容量保护值（即使面积很小也至少保留此数量）
MIN_CAPACITY_FLOOR = 5

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 加载数据
def load_multiday_data():
    files = sorted(INPUT_DIR.glob("*.csv"))
    if not files:
        print("❌ 没有找到输入文件")
        return None

    print(f"📂 找到 {len(files)} 个文件，开始加载...")
    all_dfs = []
    for idx, f in enumerate(files, 1):
        print(f"  [{idx}/{len(files)}] {f.name}")
        df = pd.read_csv(f)
        df.columns = df.columns.str.lower()
        required = ["h3_id", "time_bin", "d_in", "d_out", "slot"]
        if not all(c in df.columns for c in required):
            print(f"⚠️ 跳过 {f.name}（缺字段）")
            continue
        df["time_bin"] = pd.to_datetime(df["time_bin"], errors="coerce")
        df = df.dropna(subset=["time_bin"])
        all_dfs.append(df)

    if not all_dfs:
        print("❌ 无有效数据")
        return None

    merged = pd.concat(all_dfs, ignore_index=True)
    print(f"✅ 加载完成，总行数: {len(merged)}，有效文件: {len(all_dfs)}")
    return merged


# 过滤低流量 H3
def filter_low_flow_h3(df, min_flow=MIN_TOTAL_FLOW):
    """剔除总流量低于阈值的 H3 格子"""
    print(f"🔍 过滤总流量 < {min_flow} 的 H3 格子...")
    total_flow = df.groupby("h3_id")[["d_in", "d_out"]].sum().sum(axis=1)
    keep_h3 = total_flow[total_flow >= min_flow].index
    filtered_df = df[df["h3_id"].isin(keep_h3)]
    removed = df["h3_id"].nunique() - filtered_df["h3_id"].nunique()
    print(f"   保留 {filtered_df['h3_id'].nunique()} 个 H3，移除 {removed} 个低流量格子")
    return filtered_df


# 尝试导入 numba
try:
    from numba import jit, prange
    HAS_NUMBA = True
    print("✅ 检测到 numba，将使用 JIT 加速")
except ImportError:
    HAS_NUMBA = False
    print("⚠️ 未安装 numba，将使用普通 Python 循环（可能较慢）")

if HAS_NUMBA:
    @jit(nopython=True)
    def compute_s_and_peaks(d_in, d_out, init_S):
        n = len(d_in)
        S_arr = np.empty(n, dtype=np.float64)
        S = init_S
        for i in range(n):
            delta = d_in[i] - d_out[i]
            S = S + delta
            if S < 0:
                S = 0.0
            S_arr[i] = S
        return S_arr
else:
    def compute_s_and_peaks(d_in, d_out, init_S):
        S = init_S
        S_arr = np.empty_like(d_in)
        for i in range(len(d_in)):
            S = max(0.0, S + d_in[i] - d_out[i])
            S_arr[i] = S
        return S_arr


# 容量下限修正（基于 H3 面积）
def adjust_capacity_with_area(capacity_df, area_per_bike=AREA_PER_BIKE_M2, min_floor=MIN_CAPACITY_FLOOR):
    """
    使用 H3 网格面积对容量进行下限修正，避免容量过小。
    """
    new_caps = []
    for _, row in capacity_df.iterrows():
        h = row["h3_id"]
        hist_cap = row["capacity"]
        try:
            area = h3.cell_area(h)  # 单位：平方米
            area_based_cap = int(area / area_per_bike)
        except Exception:
            area_based_cap = min_floor  # 如果 H3 字符串无效，使用最小值
        adjusted = max(hist_cap, area_based_cap, min_floor)
        new_caps.append(adjusted)
    capacity_df["capacity"] = new_caps
    return capacity_df


# 处理所有数据（按 H3 和日期）
def process_all_data(df):
    print("🚀 开始处理数据（按 H3 和日期分组）...")
    # 确保排序
    df = df.sort_values(["h3_id", "time_bin"]).reset_index(drop=True)
    df["date"] = df["time_bin"].dt.date

    # 准备结果列
    df["S_t"] = 0.0

    # 存储每个 H3 的每日峰值
    daily_peaks = defaultdict(list)

    # 获取唯一 H3
    h3_list = df["h3_id"].unique()
    total_h3 = len(h3_list)
    print(f"共有 {total_h3} 个 H3 网格，开始逐网格处理...")

    # 第一步：使用初始库存 0 计算容量（记录每日峰值）
    print("  第1步：计算容量（初始库存 0）...")
    for idx, h3 in enumerate(h3_list, 1):
        if idx % 100 == 0:
            print(f"    处理 H3 进度: {idx}/{total_h3}")
        mask = df["h3_id"] == h3
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
        dates = df.loc[indices, "date"].values
        d_in = df.loc[indices, "d_in"].values.astype(np.float64)
        d_out = df.loc[indices, "d_out"].values.astype(np.float64)

        current_date = dates[0]
        start = 0
        S = 0.0
        for i in range(1, len(dates)):
            if dates[i] != current_date:
                day_d_in = d_in[start:i]
                day_d_out = d_out[start:i]
                if HAS_NUMBA:
                    day_S = compute_s_and_peaks(day_d_in, day_d_out, S)
                else:
                    day_S = compute_s_and_peaks(day_d_in, day_d_out, S)
                daily_peaks[h3].append(day_S.max())
                S = day_S[-1]
                current_date = dates[i]
                start = i
        if start < len(dates):
            day_d_in = d_in[start:]
            day_d_out = d_out[start:]
            if HAS_NUMBA:
                day_S = compute_s_and_peaks(day_d_in, day_d_out, S)
            else:
                day_S = compute_s_and_peaks(day_d_in, day_d_out, S)
            daily_peaks[h3].append(day_S.max())

    # 计算容量
    capacity_dict = {}
    for h3, peaks in daily_peaks.items():
        if peaks:
            capacity = np.percentile(peaks, 95) # 取95分位数作为容量估计
        else:
            capacity = 1.0
        capacity_dict[h3] = max(capacity, 1.0)
    capacity_df = pd.DataFrame(list(capacity_dict.items()), columns=["h3_id", "capacity"])
    print(f"  容量计算完成，站点数 {len(capacity_df)}")

    # 应用容量下限修正
    print("  应用基于面积的容量下限修正...")
    capacity_df = adjust_capacity_with_area(capacity_df)
    print(f"  修正后容量统计：min={capacity_df['capacity'].min():.1f}, max={capacity_df['capacity'].max():.1f}, mean={capacity_df['capacity'].mean():.1f}")

    # 第二步：使用真实初始库存递推 S_t
    print("  第2步：递推 S_t（使用容量 × init_ratio 作为初始库存）...")
    cap_map = dict(zip(capacity_df["h3_id"], capacity_df["capacity"]))
    avg_cap = capacity_df["capacity"].mean()

    for idx, h3 in enumerate(h3_list, 1):
        if idx % 100 == 0:
            print(f"    处理 H3 进度: {idx}/{total_h3}")
        mask = df["h3_id"] == h3
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
        cap = cap_map.get(h3, avg_cap)
        init_S = cap * INIT_S0_RATIO

        dates = df.loc[indices, "date"].values
        d_in = df.loc[indices, "d_in"].values.astype(np.float64)
        d_out = df.loc[indices, "d_out"].values.astype(np.float64)

        S_arr_full = np.empty(len(indices), dtype=np.float64)
        current_date = dates[0]
        start = 0
        S = init_S
        for i in range(1, len(dates)):
            if dates[i] != current_date:
                day_d_in = d_in[start:i]
                day_d_out = d_out[start:i]
                if HAS_NUMBA:
                    day_S = compute_s_and_peaks(day_d_in, day_d_out, S)
                else:
                    day_S = compute_s_and_peaks(day_d_in, day_d_out, S)
                S_arr_full[start:i] = day_S
                S = day_S[-1]
                current_date = dates[i]
                start = i
        if start < len(dates):
            day_d_in = d_in[start:]
            day_d_out = d_out[start:]
            if HAS_NUMBA:
                day_S = compute_s_and_peaks(day_d_in, day_d_out, S)
            else:
                day_S = compute_s_and_peaks(day_d_in, day_d_out, S)
            S_arr_full[start:] = day_S

        df.loc[indices, "S_t"] = S_arr_full

    print("  递推完成")
    return capacity_df, df


# 按日期输出
def output_by_date(df, warmup_days=WARMUP_DAYS):
    df["date"] = df["time_bin"].dt.date
    dates = sorted(df["date"].unique())
    if len(dates) <= warmup_days:
        print(f"⚠️ 总天数 {len(dates)} ≤ 预热期 {warmup_days}，无输出")
        return

    output_dates = dates[warmup_days:]
    print(f"输出 {len(output_dates)} 天数据（预热期前 {warmup_days} 天已用于稳定）")

    for idx, date in enumerate(output_dates, 1):
        df_day = df[df["date"] == date].copy()
        df_day["tide_index"] = (df_day["d_in"] - df_day["d_out"]) / (df_day["d_in"] + df_day["d_out"] + 1)

        out_file = OUTPUT_DIR / f"{date}.csv"
        cols = ["h3_id", "time_bin", "d_out", "d_in", "slot", "slot_sin", "slot_cos",
                "weekday", "is_weekend", "holiday", "is_preholiday", "S_t", "tide_index"]
        cols_present = [c for c in cols if c in df_day.columns]
        df_day[cols_present].to_csv(out_file, index=False)
        if idx % 20 == 0:
            print(f"  已输出 {idx}/{len(output_dates)} 天")


# 主程序
def main():
    print("🚀 开始预处理（numba 极速版 + 低流量过滤 + 容量下限修正）")
    all_data = load_multiday_data()
    if all_data is None:
        return

    # 过滤低流量格子
    all_data = filter_low_flow_h3(all_data, min_flow=MIN_TOTAL_FLOW)

    dates = sorted(all_data["time_bin"].dt.date.unique())
    print(f"✔ 数据时间范围: {dates[0]} ~ {dates[-1]} ({len(dates)} 天)")

    capacity_df, all_data = process_all_data(all_data)
    output_by_date(all_data)

    capacity_df.to_csv(CAPACITY_FILE, index=False)
    print(f"✔ 容量保存至 {CAPACITY_FILE}")
    print("✅ 完成")


if __name__ == "__main__":
    main()