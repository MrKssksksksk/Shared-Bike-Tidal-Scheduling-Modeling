import pandas as pd
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point
import multiprocessing as mp

# 调试开关
DEBUG = False

# 路径配置
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
GEO_PATH = Path("data/geo/shenzhen_boundary.shp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 深圳粗筛范围
LAT_MIN, LAT_MAX = 22.4, 22.9
LNG_MIN, LNG_MAX = 113.7, 114.7

# 加载 shapefile
print("📍 加载深圳边界...")
gdf = gpd.read_file(GEO_PATH)
gdf = gdf.set_crs(epsg=4326)

# 减少复杂度
SZ_POLYGON = gdf.unary_union.simplify(0.0001)

# GCJ → WGS
def _transform_latlng(x, y):
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

def gcj2wgs(lng, lat):
    if 72.004 <= lng <= 137.8347 and 0.8293 <= lat <= 55.8271:
        dlng, dlat = _transform_latlng(lng, lat)
        return lng - dlng, lat - dlat
    return lng, lat

def convert_gcj_to_wgs(df):
    df[["start_lng", "start_lat"]] = df.apply(
        lambda x: gcj2wgs(x["start_lng"], x["start_lat"]),
        axis=1, result_type="expand"
    )
    df[["end_lng", "end_lat"]] = df.apply(
        lambda x: gcj2wgs(x["end_lng"], x["end_lat"]),
        axis=1, result_type="expand"
    )
    return df

# 核心清洗
def process_chunk(df):
    df.columns = df.columns.str.strip().str.lower()

    if DEBUG:
        print("列名:", df.columns.tolist())

    required = ["start_time","end_time","start_lat","start_lng","end_lat","end_lng"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"缺少字段: {col}")

    # 删除空值
    df = df.dropna()

    # 粗筛
    df = df[
        (df["start_lat"].between(LAT_MIN, LAT_MAX)) &
        (df["start_lng"].between(LNG_MIN, LNG_MAX)) &
        (df["end_lat"].between(LAT_MIN, LAT_MAX)) &
        (df["end_lng"].between(LNG_MIN, LNG_MAX))
    ]
    if df.empty:
        return df

    # 坐标转换
    df = convert_gcj_to_wgs(df)

    # 时间处理
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")
    df = df.dropna(subset=["start_time","end_time"])
    df = df[df["end_time"] > df["start_time"]]

    # 时长
    df["duration"] = (df["end_time"] - df["start_time"]).dt.total_seconds()
    df = df[(df["duration"] >= 60) & (df["duration"] <= 7200)]
    if df.empty:
        return df

    # shapefile过滤（起点、终点）
    start_points = gpd.GeoSeries(
        [Point(xy) for xy in zip(df["start_lng"], df["start_lat"])],
        index=df.index, crs="EPSG:4326"
    )
    end_points = gpd.GeoSeries(
        [Point(xy) for xy in zip(df["end_lng"], df["end_lat"])],
        index=df.index, crs="EPSG:4326"
    )

    mask = start_points.within(SZ_POLYGON) & end_points.within(SZ_POLYGON)
    df = df[mask]

    # ===== 添加字段 =====
    df["date"] = df["start_time"].dt.date

    return df

# 单文件处理
def clean_one_file(file_path):
    output_path = OUT_DIR / f"clean_{file_path.name}"

    if output_path.exists():
        print(f"⏭️ 跳过 {file_path.name}")
        return (file_path.name, "Skipped")

    try:
        print(f"\n🚀 处理 {file_path.name}")

        chunks = pd.read_csv(file_path, chunksize=100000)

        total_raw = 0
        total_clean = 0
        first = True

        for i, chunk in enumerate(chunks):
            total_raw += len(chunk)

            cleaned = process_chunk(chunk)
            total_clean += len(cleaned)

            if not cleaned.empty:
                cleaned.to_csv(
                    output_path,
                    mode="a",
                    index=False,
                    header=first
                )
                first = False

            print(f" 📦 chunk {i+1} | 保留 {len(cleaned)}")

        ratio = (total_raw - total_clean) / total_raw if total_raw else 0

        return (file_path.name,
                f"✅ 完成 | 原始:{total_raw}→清洗:{total_clean} | 删除:{ratio:.2%}")

    except Exception as e:
        return (file_path.name, f"❌ 错误: {str(e)}")

# 主程序（多进程）
if __name__ == "__main__":
    files = list(RAW_DIR.glob("*.csv"))

    if not files:
        print("📭 无数据")
        exit()

    print(f"🎯 {len(files)} 个文件")

    # ⭐ 更稳定
    num_proc = max(1, mp.cpu_count() // 2)
    print(f"🚀 使用 {num_proc} 进程")

    with mp.Pool(num_proc) as pool:
        results = pool.imap(clean_one_file, files, chunksize=1)

        for name, status in results:
            print(f"📄 {name}: {status}")

    print("\n🎉 全部完成")