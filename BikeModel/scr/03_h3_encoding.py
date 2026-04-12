import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
import h3

INPUT_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/h3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

H3_RESOLUTION = 8
CHUNK_SIZE = 100000

def process_chunk(chunk):
    chunk.columns = chunk.columns.str.lower()

    # 防止异常数据
    chunk = chunk.dropna(subset=["start_lat","start_lng","end_lat","end_lng"])

    chunk["h3_start"] = [
        h3.latlng_to_cell(lat, lng, H3_RESOLUTION)
        for lat, lng in zip(chunk["start_lat"], chunk["start_lng"])
    ]

    chunk["h3_end"] = [
        h3.latlng_to_cell(lat, lng, H3_RESOLUTION)
        for lat, lng in zip(chunk["end_lat"], chunk["end_lng"])
    ]

    return chunk

def process_file(file_path):
    out_file = OUTPUT_DIR / f"h3_{file_path.name}"

    if out_file.exists():
        print(f"⏭️ 跳过 {file_path.name}")
        return

    print(f"🚀 {file_path.name}")

    chunks = pd.read_csv(file_path, chunksize=CHUNK_SIZE)
    first = True
    tmp_file = out_file.with_suffix(".tmp")

    if tmp_file.exists():
        tmp_file.unlink()

    for i, chunk in enumerate(chunks):
        processed = process_chunk(chunk)

        processed.to_csv(
            tmp_file,
            mode="a" if not first else "w",
            index=False,
            header=first
        )

        first = False
        print(f"  chunk {i}")

    if tmp_file.exists():
        tmp_file.replace(out_file)

if __name__ == "__main__":
    files = list(INPUT_DIR.glob("*.csv"))
    with Pool(max(1, cpu_count() // 2)) as pool:
        pool.map(process_file, files)