import pandas as pd
import numpy as np
import pickle
import json
import h3
from pathlib import Path
from datetime import datetime
import sys
import importlib.util
from collections import Counter

# 动态导入 FlowPredictor，并注册到 __main__ 以便 pickle 识别
spec = importlib.util.spec_from_file_location("prediction_model", "scr/08_prediction_model_with_ablation.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
FlowPredictor = module.FlowPredictor

# 将 FlowPredictor 注入到 __main__ 模块，确保 pickle 能找到
sys.modules['__main__'].FlowPredictor = FlowPredictor

TIME_SLOTS = 48
MODEL_PATH = Path("models/prediction_model_no_cluster.pkl")
TARGET_PATH = Path("data/prediction_target.json")
OUTPUT_PATH = Path("data/prediction_no_cluster.csv")


def load_target():
    with open(TARGET_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    date = datetime.strptime(cfg["target_date"], "%Y-%m-%d")
    weekday = date.weekday()
    holiday = cfg.get("holiday", 0)
    preholiday = cfg.get("is_preholiday", 0)
    h3_list = [str(x).lower() for x in cfg["region_h3_list"]]
    return weekday, holiday, preholiday, h3_list, date


def fill_missing_h3(h3_list, predictor):
    """
    补全目标区域内所有h3的cluster（使用邻居众数填充）
    """
    filled = predictor.cluster_map.copy()
    h3_set = set(h3_list)
    changed = True

    while changed:
        changed = False
        for h in h3_set:
            if h in filled:
                continue
            neighbors = h3.grid_ring(h, 1)
            vals = [
                filled.get(str(n).lower())
                for n in neighbors
                if str(n).lower() in filled
            ]
            if vals:
                # 使用众数填充，避免产生不存在的 cluster 编号
                most_common = Counter(vals).most_common(1)[0][0]
                filled[h] = most_common
                changed = True
    return filled


def main():
    print("🚀 加载预测目标...")
    weekday, holiday, preholiday, h3_list, target_date = load_target()

    print("📦 加载预测模型...")
    with open(MODEL_PATH, "rb") as f:
        predictor = pickle.load(f)

    # 加载聚类结果作为初始 cluster_map
    cluster_df = pd.read_csv("data/clustered.csv")
    cluster_map = dict(
        zip(cluster_df['h3_id'].astype(str).str.lower(),
            cluster_df['cluster'])
    )
    predictor.cluster_map = cluster_map

    # 补全缺失站点的 cluster
    print("🔄 补全缺失站点的 cluster...")
    predictor.cluster_map = fill_missing_h3(h3_list, predictor)

    print("🔮 开始预测...")
    Pred_In, Pred_Out, S0 = predictor.predict_for_region(
        h3_list, weekday, holiday, preholiday
    )

    # 构建输出
    rows = []
    valid_count = 0
    fallback_count = 0

    for j, h in enumerate(h3_list):
        cluster_id = predictor.cluster_map.get(h, -1)
        if cluster_id == -1:
            cluster_id = predictor.find_cluster_of_nearby_h3(h)

        for t in range(TIME_SLOTS):
            try:
                d_in, d_out = Pred_In[t, j], Pred_Out[t, j]
                valid_count += 1
            except Exception:
                d_in, d_out = 0, 0
                fallback_count += 1

            rows.append({
                "h3_id": h,
                "time_slot": t,
                "D_in": round(d_in, 2),
                "D_out": round(d_out, 2),
                "S_t": round(S0[j], 2) if t == 0 else np.nan
            })

    print(f"✅ 有效预测: {valid_count}")
    print(f"⚠️ fallback: {fallback_count}")

    df = pd.DataFrame(rows).sort_values(["h3_id", "time_slot"])
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ 完成: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()