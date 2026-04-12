import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pickle
from pathlib import Path

# 配置
INPUT_DIR = Path("data/second_preprocessed")
OUTPUT_PATH = Path("data/clustered.csv")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(exist_ok=True)

# 初始候选聚类数范围（最终会根据样本数和轮廓系数自动选择）
MIN_CLUSTERS = 2
MAX_CLUSTERS = 8
RANDOM_STATE = 42

# 流式特征构建
def build_features():
    print("🚀 构建特征（流式）...")
    files = sorted(INPUT_DIR.glob("*.csv"))
    print(f"📂 发现 {len(files)} 个输入文件")

    agg_df = None
    for idx, file in enumerate(files, start=1):
        print(f"  [{idx}/{len(files)}] 处理: {file.name}")
        df = pd.read_csv(file)

        # 兼容不同列名大小写
        df.columns = df.columns.str.lower()

        # 确保所需列存在
        required = ["h3_id", "d_in", "d_out", "s_t", "slot_sin", "slot_cos",
                    "tide_index", "is_weekend", "holiday"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"⚠️ 文件 {file.name} 缺少列: {missing}，跳过")
            continue

        df["flow"] = df["d_in"] + df["d_out"]
        df["weighted_sin"] = df["slot_sin"] * (df["flow"] + 1)
        df["weighted_cos"] = df["slot_cos"] * (df["flow"] + 1)

        grouped = df.groupby("h3_id").agg(
            n_records=("flow", "count"),
            sum_in=("d_in", "sum"),
            sum_in2=("d_in", lambda x: np.sum(x.values.astype(float) ** 2)),
            sum_out=("d_out", "sum"),
            sum_out2=("d_out", lambda x: np.sum(x.values.astype(float) ** 2)),
            sum_inventory=("s_t", "sum"),
            sum_inventory2=("s_t", lambda x: np.sum(x.values.astype(float) ** 2)),
            max_inventory=("s_t", "max"),
            sum_tide=("tide_index", "sum"),
            sum_tide2=("tide_index", lambda x: np.sum(x.values.astype(float) ** 2)),
            sum_weekend=("is_weekend", "sum"),
            sum_holiday=("holiday", "sum"),
            total_flow=("flow", "sum"),
            sum_weighted_sin=("weighted_sin", "sum"),
            sum_weighted_cos=("weighted_cos", "sum")
        ).reset_index()

        if agg_df is None:
            agg_df = grouped
        else:
            agg_df = pd.concat([agg_df, grouped], ignore_index=True)
            agg_df = agg_df.groupby("h3_id", as_index=False).sum()

        print(f"    ✅ 累计聚合 {len(agg_df)} 个 H3")

    if agg_df is None:
        raise ValueError("没有读取到任何有效的输入文件")

    # 计算最终特征
    agg_df["avg_in"] = agg_df["sum_in"] / agg_df["n_records"]
    agg_df["std_in"] = np.sqrt(np.maximum(agg_df["sum_in2"] / agg_df["n_records"] - agg_df["avg_in"] ** 2, 0))
    agg_df["avg_out"] = agg_df["sum_out"] / agg_df["n_records"]
    agg_df["std_out"] = np.sqrt(np.maximum(agg_df["sum_out2"] / agg_df["n_records"] - agg_df["avg_out"] ** 2, 0))
    agg_df["avg_inventory"] = agg_df["sum_inventory"] / agg_df["n_records"]
    agg_df["std_inventory"] = np.sqrt(np.maximum(agg_df["sum_inventory2"] / agg_df["n_records"] - agg_df["avg_inventory"] ** 2, 0))
    agg_df["avg_tide"] = agg_df["sum_tide"] / agg_df["n_records"]
    agg_df["std_tide"] = np.sqrt(np.maximum(agg_df["sum_tide2"] / agg_df["n_records"] - agg_df["avg_tide"] ** 2, 0))
    agg_df["weekend_ratio"] = agg_df["sum_weekend"] / agg_df["n_records"]
    agg_df["holiday_ratio"] = agg_df["sum_holiday"] / agg_df["n_records"]
    agg_df["avg_active_slot"] = (
        np.arctan2(agg_df["sum_weighted_sin"], agg_df["sum_weighted_cos"]) * 48 / (2 * np.pi)
    ) % 48

    df_features = agg_df[[
        "h3_id",
        "avg_in", "std_in", "avg_out", "std_out", "total_flow",
        "avg_inventory", "std_inventory", "max_inventory",
        "avg_tide", "std_tide",
        "weekend_ratio", "holiday_ratio",
        "avg_active_slot"
    ]]

    print(f"✅ 特征构建完成: {df_features.shape}")
    return df_features


def find_optimal_k(X_scaled, min_k=2, max_k=8):
    """
    使用轮廓系数自动选择最佳聚类数
    """
    best_k = min_k
    best_score = -1
    n_samples = X_scaled.shape[0]
    upper_k = min(max_k, n_samples - 1)  # 不能超过样本数-1
    if upper_k < min_k:
        print(f"⚠️ 样本数 {n_samples} 过少，强制使用 {min_k} 个簇")
        return min_k

    print(f"🔍 在 {min_k}~{upper_k} 范围内搜索最优聚类数...")
    for k in range(min_k, upper_k + 1):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(X_scaled)
        # 轮廓系数计算需要至少2个簇且每个簇至少2个样本（实际sklearn会自动处理）
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(X_scaled, labels)
        print(f"  k={k}, 轮廓系数={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k

    print(f"✅ 最优聚类数: {best_k} (轮廓系数: {best_score:.4f})")
    return best_k


def clustering(df):
    # 1. 准备特征数据，去除标识列
    X = df.drop(columns=["h3_id"]).copy()

    # 2. 处理缺失值与无穷值
    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isnull().any().any():
        print(f"⚠️ 检测到缺失值，使用中位数填充")
        X = X.fillna(X.median())

    # 3. 去极值（1% ~ 99%）
    X = X.clip(lower=X.quantile(0.01), upper=X.quantile(0.99), axis=1)

    # 4. 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. 自动选择最优聚类数
    optimal_k = find_optimal_k(X_scaled, min_k=MIN_CLUSTERS, max_k=MAX_CLUSTERS)

    # 6. KMeans 最终训练
    model = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(X_scaled)

    # 7. 输出结果
    result_df = pd.DataFrame({
        "h3_id": df["h3_id"],
        "cluster": labels
    })

    # 保存聚类结果
    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f"📄 聚类结果已保存至 {OUTPUT_PATH}")

    # 保存模型
    with open(MODEL_DIR / "kmeans.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODEL_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # 打印分布信息
    cluster_counts = result_df["cluster"].value_counts().sort_index()
    print("🎉 聚类完成，各类样本数分布：")
    for cl, cnt in cluster_counts.items():
        print(f"  Cluster {cl}: {cnt} 个站点")

    return result_df


# 主程序
if __name__ == "__main__":
    df_features = build_features()
    clustering(df_features)