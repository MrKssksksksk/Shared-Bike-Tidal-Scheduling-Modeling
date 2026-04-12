import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
import chinese_calendar as cc

# 路径
INPUT_DIR = Path("data/h3")
TEMP_DIR = Path("data/flow_raw")
FINAL_DIR = Path("data/flow_final")

TEMP_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 100000

# 时间分箱
def time_to_bin(series):
    return pd.to_datetime(series, errors="coerce").dt.floor("30min")

# chunk处理
def process_chunk(chunk):
    chunk.columns = (
        chunk.columns
        .str.strip()
        .str.lower()
        .str.replace("\ufeff", "", regex=False)
    )

    # 时间过滤
    mask = (
        chunk["start_time"].astype(str).str.len() >= 19
    ) & (
        chunk["end_time"].astype(str).str.len() >= 19
    )

    chunk = chunk[mask]

    if len(chunk) == 0:
        return None

    # 时间转换
    start_dt = pd.to_datetime(chunk["start_time"], errors="coerce")
    end_dt = pd.to_datetime(chunk["end_time"], errors="coerce")

    valid = start_dt.notna() & end_dt.notna()
    chunk = chunk[valid]

    if len(chunk) == 0:
        return None

    chunk["start_bin"] = start_dt[valid].dt.floor("30min")
    chunk["end_bin"] = end_dt[valid].dt.floor("30min")

    # 出流（单独处理）
    out_flow = (
        chunk.groupby(["h3_start", "start_bin"])
        .size()
        .reset_index(name="D_out")
    )

    out_flow = out_flow.rename(columns={
        "h3_start": "h3_id",
        "start_bin": "time_bin"
    })

    out_flow["D_in"] = 0

    # 入流（单独处理）
    in_flow = (
        chunk.groupby(["h3_end", "end_bin"])
        .size()
        .reset_index(name="D_in")
    )

    in_flow = in_flow.rename(columns={
        "h3_end": "h3_id",
        "end_bin": "time_bin"
    })

    in_flow["D_out"] = 0

    # 合并
    flow = pd.concat([out_flow, in_flow], ignore_index=True)

    # 最终聚合
    flow = flow.groupby(
        ["h3_id", "time_bin"],
        as_index=False
    ).sum()

    return flow

# 单文件处理
def process_file(file_path):
    print(f"🚀 处理: {file_path.name}")

    out_file = TEMP_DIR / f"flow_raw_{file_path.name}"

    if out_file.exists():
        print(f"⏭️ 跳过: {file_path.name}")
        return

    # 文件头检查
    df_head = pd.read_csv(file_path, nrows=5, encoding="utf-8-sig")

    cols = (
        df_head.columns
        .str.strip()
        .str.lower()
        .str.replace("\ufeff", "", regex=False)
    )

    required = ["start_time", "end_time", "h3_start", "h3_end"]

    if not all(c in cols for c in required):
        print(f"❌ 跳过异常文件: {file_path.name}")
        print(f"   列名: {cols.tolist()}")
        return

    # chunk读取
    chunks = pd.read_csv(
        file_path,
        chunksize=CHUNK_SIZE,
        encoding="utf-8-sig",
        on_bad_lines="skip"   # 自动跳过异常行
    )

    first = True

    for i, chunk in enumerate(chunks):
        flow = process_chunk(chunk)

        if flow is None or len(flow) == 0:
            continue

        flow.to_csv(
            out_file,
            mode="a",
            index=False,
            header=first
        )

        first = False
        print(f"  chunk {i}")

# 全局聚合
def merge_all_flows():
    print("🚀 开始全局聚合")

    files = list(TEMP_DIR.glob("flow_raw_*.csv"))

    agg_df = None

    for i, f in enumerate(files):
        print(f"  合并: {f.name}")

        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"❌ 读取失败，跳过: {f.name}")
            continue

        # 列名标准化
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace("\ufeff", "", regex=False)
        )

        # 核心检查
        if "h3_id" not in df.columns or "time_bin" not in df.columns:
            print(f"❌ 字段缺失，跳过: {f.name}")
            print(f"   实际列: {df.columns.tolist()}")
            continue

        df = df.groupby(["h3_id", "time_bin"]).sum().reset_index()

        if agg_df is None:
            agg_df = df
        else:
            agg_df = pd.concat([agg_df, df])
            agg_df = agg_df.groupby(["h3_id", "time_bin"]).sum().reset_index()

        if i % 5 == 0:
            agg_df = agg_df.copy()

    print("agg_df 行数:", len(agg_df))

    print("📊 添加时间特征")

    # 时间转换
    agg_df["time_bin"] = pd.to_datetime(
        agg_df["time_bin"],
        errors="coerce"
    )

    # 过滤非法时间
    agg_df = agg_df[
        agg_df["time_bin"].notna()
    ]

    # 过滤异常年份
    agg_df = agg_df[
        (agg_df["time_bin"].dt.year >= 2004) &
        (agg_df["time_bin"].dt.year <= 2026)
    ]

    # slot
    agg_df["slot"] = (
        agg_df["time_bin"].dt.hour * 2 +
        agg_df["time_bin"].dt.minute // 30
    )

    # 周期
    agg_df["slot_sin"] = np.sin(2 * np.pi * agg_df["slot"] / 48)
    agg_df["slot_cos"] = np.cos(2 * np.pi * agg_df["slot"] / 48)

    # 日期特征
    agg_df["weekday"] = agg_df["time_bin"].dt.weekday
    agg_df["is_weekend"] = (agg_df["weekday"] >= 5).astype(int)

    def safe_is_holiday(x):
        try:
            return int(cc.is_holiday(x))
        except:
            return 0

    agg_df["holiday"] = agg_df["time_bin"].map(safe_is_holiday)

    next_day = agg_df["time_bin"] + pd.Timedelta(days=1)
    next_day_is_holiday_or_weekend = next_day.map(
        lambda x: int(cc.is_holiday(x) or x.weekday() >= 5)
    )
    agg_df["is_preholiday"] = (
        (agg_df["weekday"] < 5) &
        (agg_df["holiday"] == 0) &
        (next_day_is_holiday_or_weekend == 1)
    ).astype(int)

    # 按天输出
    agg_df["date"] = agg_df["time_bin"].dt.date

    for d, group in agg_df.groupby("date"):
        group.to_csv(FINAL_DIR / f"{d}.csv", index=False)

    print("🎉 完成")

# 主程序
if __name__ == "__main__":
    files = list(INPUT_DIR.glob("h3_clean_*.csv"))

    print(f"📂 共 {len(files)} 个文件")

    with Pool(max(1, cpu_count() // 2)) as pool:
        pool.map(process_file, files)

    merge_all_flows()