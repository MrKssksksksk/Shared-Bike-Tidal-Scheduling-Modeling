import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import sys

# 配置
TARGET_DATE = "2021-08-21"          # 要比较的日期（与预测目标一致）
FLOW_FINAL_DIR = Path("data/unused dates/flow_final")   # 04输出按日期csv的目录
PREDICTION_PATH = Path("data/prediction.csv")
OUTPUT_DIR = Path("data/comparison")
OUTPUT_DIR.mkdir(exist_ok=True)

TIME_SLOTS = 48

# 加载数据
def load_ground_truth(date_str: str) -> pd.DataFrame:
    """加载原始流量数据（04输出）"""
    file_path = FLOW_FINAL_DIR / f"{date_str}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"未找到原始数据文件: {file_path}")
    
    df = pd.read_csv(file_path)
    # 确保列名小写统一
    df.columns = df.columns.str.lower()
    required_cols = ["h3_id", "slot", "d_in", "d_out"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"原始数据缺少列: {col}")
    
    # 只保留需要的时间槽
    df = df[df["slot"].between(0, TIME_SLOTS-1)]
    return df[["h3_id", "slot", "d_in", "d_out"]].copy()

def load_prediction() -> pd.DataFrame:
    """加载预测数据（09输出）"""
    if not PREDICTION_PATH.exists():
        raise FileNotFoundError(f"未找到预测数据文件: {PREDICTION_PATH}")
    
    df = pd.read_csv(PREDICTION_PATH)
    df.columns = df.columns.str.lower()
    required_cols = ["h3_id", "time_slot", "d_in", "d_out"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"预测数据缺少列: {col}")
    
    # 重命名 time_slot 为 slot 以便合并
    df = df.rename(columns={"time_slot": "slot"})
    df = df[df["slot"].between(0, TIME_SLOTS-1)]
    return df[["h3_id", "slot", "d_in", "d_out"]].copy()

# 计算指标
def compute_metrics(actual: pd.Series, predicted: pd.Series, name: str):
    """计算并打印常用误差指标"""
    actual = actual.values
    predicted = predicted.values
    
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    
    # MAPE: 避免除零，仅对实际值>0的样本计算
    mask = actual > 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    else:
        mape = np.nan
    
    # R²
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    print(f"\n📊 {name} 指标:")
    print(f"   MAE  = {mae:.4f}")
    print(f"   RMSE = {rmse:.4f}")
    print(f"   MAPE = {mape:.2f}%")
    print(f"   R²   = {r2:.4f}")
    
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

# 绘图
def plot_scatter(actual, predicted, name, output_path):
    """散点图 + 理想线"""
    plt.figure(figsize=(6, 6))
    plt.scatter(actual, predicted, alpha=0.3, s=10)
    max_val = max(actual.max(), predicted.max())
    plt.plot([0, max_val], [0, max_val], 'r--', label="Perfect Prediction")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{name} - Scatter Plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  散点图保存至: {output_path}")

def plot_time_series_by_h3(merged_df, h3_sample=None, n_samples=5, output_dir=None):
    # 随机选取若干站点，绘制真实 vs 预测的时间序列
    h3_ids = merged_df["h3_id"].unique()
    if h3_sample is None:
        np.random.seed(42)
        h3_sample = np.random.choice(h3_ids, size=min(n_samples, len(h3_ids)), replace=False)
    
    fig, axes = plt.subplots(len(h3_sample), 2, figsize=(12, 3*len(h3_sample)))
    if len(h3_sample) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, h3 in enumerate(h3_sample):
        sub = merged_df[merged_df["h3_id"] == h3].sort_values("slot")
        slots = sub["slot"].values
        
        # D_in
        axes[idx, 0].plot(slots, sub["d_in_actual"], 'o-', label="Actual", markersize=3)
        axes[idx, 0].plot(slots, sub["d_in_pred"], 's-', label="Predicted", markersize=3)
        axes[idx, 0].set_title(f"{h3} - D_in")
        axes[idx, 0].legend()
        axes[idx, 0].set_xlabel("Time Slot")
        axes[idx, 0].set_ylabel("Flow")
        
        # D_out
        axes[idx, 1].plot(slots, sub["d_out_actual"], 'o-', label="Actual", markersize=3)
        axes[idx, 1].plot(slots, sub["d_out_pred"], 's-', label="Predicted", markersize=3)
        axes[idx, 1].set_title(f"{h3} - D_out")
        axes[idx, 1].legend()
        axes[idx, 1].set_xlabel("Time Slot")
        axes[idx, 1].set_ylabel("Flow")
    
    plt.tight_layout()
    out_path = output_dir / "time_series_samples.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  时间序列样本图保存至: {out_path}")

# 主程序
def main():
    print("🚀 开始比较预测值与真实值")
    print(f"📅 目标日期: {TARGET_DATE}")
    
    # 加载数据
    try:
        actual_df = load_ground_truth(TARGET_DATE)
        print(f"✅ 原始数据加载完成: {len(actual_df)} 条记录")
    except Exception as e:
        print(f"❌ 加载原始数据失败: {e}")
        sys.exit(1)
    
    try:
        pred_df = load_prediction()
        print(f"✅ 预测数据加载完成: {len(pred_df)} 条记录")
    except Exception as e:
        print(f"❌ 加载预测数据失败: {e}")
        sys.exit(1)
    
    # 合并数据
    merged = pd.merge(
        actual_df, pred_df,
        on=["h3_id", "slot"],
        suffixes=("_actual", "_pred"),
        how="inner"
    )
    print(f"🔗 合并后记录数: {len(merged)} (丢失 {len(actual_df)+len(pred_df)-2*len(merged)} 条不匹配记录)")
    
    if len(merged) == 0:
        print("❌ 无匹配数据，请检查h3_id或时段范围是否一致")
        sys.exit(1)
    
    # 计算指标
    metrics_in = compute_metrics(merged["d_in_actual"], merged["d_in_pred"], "D_in")
    metrics_out = compute_metrics(merged["d_out_actual"], merged["d_out_pred"], "D_out")
    
    # 保存指标到文本文件
    metrics_df = pd.DataFrame([metrics_in, metrics_out], index=["D_in", "D_out"])
    metrics_df.to_csv(OUTPUT_DIR / "metrics.csv")
    print(f"\n📈 指标已保存至 {OUTPUT_DIR / 'metrics.csv'}")
    
    # 保存详细差异文件
    merged["diff_in"] = merged["d_in_pred"] - merged["d_in_actual"]
    merged["diff_out"] = merged["d_out_pred"] - merged["d_out_actual"]
    merged["abs_diff_in"] = np.abs(merged["diff_in"])
    merged["abs_diff_out"] = np.abs(merged["diff_out"])
    merged.to_csv(OUTPUT_DIR / "detailed_comparison.csv", index=False)
    print(f"📄 详细差异文件保存至 {OUTPUT_DIR / 'detailed_comparison.csv'}")
    
    # 绘图
    plot_scatter(merged["d_in_actual"], merged["d_in_pred"], "D_in", OUTPUT_DIR / "scatter_in.png")
    plot_scatter(merged["d_out_actual"], merged["d_out_pred"], "D_out", OUTPUT_DIR / "scatter_out.png")
    plot_time_series_by_h3(merged, n_samples=5, output_dir=OUTPUT_DIR)
    
    print("\n🎉 比较完成！")

if __name__ == "__main__":
    main()