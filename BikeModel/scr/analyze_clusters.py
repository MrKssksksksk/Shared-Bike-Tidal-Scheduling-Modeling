import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 配置
CLUSTER_FILE = Path("data/clustered.csv")
DATA_DIR = Path("data/second_preprocessed")
OUTPUT_DIR = Path("outputs/cluster_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 早高峰时段 (7:00-9:00 → slot 14-17)，晚高峰时段 (17:30-19:30 → slot 35-38)
MORNING_PEAK_SLOTS = list(range(14, 18))   # 14,15,16,17
EVENING_PEAK_SLOTS = list(range(35, 39))   # 35,36,37,38

# 中文字体设置
import platform
if platform.system() == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
elif platform.system() == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

# 数据加载
print("📂 加载聚类结果...")
cluster_df = pd.read_csv(CLUSTER_FILE)
cluster_df['h3_id'] = cluster_df['h3_id'].astype(str).str.lower()
cluster_map = dict(zip(cluster_df['h3_id'], cluster_df['cluster']))

print("📂 加载 second_preprocessed 数据...")
files = list(DATA_DIR.glob("*.csv"))
df_list = []
for f in files:
    df = pd.read_csv(f)
    df.columns = df.columns.str.lower()
    required = ['h3_id', 'slot', 'd_in', 'd_out', 'tide_index']
    if not all(c in df.columns for c in required):
        continue
    df = df[required].copy()
    df['h3_id'] = df['h3_id'].astype(str).str.lower()
    df_list.append(df)

df_all = pd.concat(df_list, ignore_index=True)
df_all['cluster'] = df_all['h3_id'].map(cluster_map)
df_all = df_all.dropna(subset=['cluster'])
df_all['cluster'] = df_all['cluster'].astype(int)

print(f"✅ 总记录数: {len(df_all)}，覆盖 {df_all['h3_id'].nunique()} 个站点")

# 站点级特征聚合
print("📊 计算站点级特征...")

# 按站点和cluster汇总
site_stats = df_all.groupby(['h3_id', 'cluster']).agg(
    total_flow=('d_in', lambda x: x.sum() + df_all.loc[x.index, 'd_out'].sum()),
    avg_tide=('tide_index', 'mean'),
    std_tide=('tide_index', 'std')
).reset_index()

# 计算早晚高峰流量占比
def calc_peak_ratio(group, peak_slots, col):
    total = group[col].sum()
    peak = group[group['slot'].isin(peak_slots)][col].sum()
    return peak / total if total > 0 else 0

peak_ratios = df_all.groupby('h3_id').apply(
    lambda g: pd.Series({
        'morning_out_ratio': calc_peak_ratio(g, MORNING_PEAK_SLOTS, 'd_out'),
        'morning_in_ratio': calc_peak_ratio(g, MORNING_PEAK_SLOTS, 'd_in'),
        'evening_out_ratio': calc_peak_ratio(g, EVENING_PEAK_SLOTS, 'd_out'),
        'evening_in_ratio': calc_peak_ratio(g, EVENING_PEAK_SLOTS, 'd_in'),
        'peak_flow_ratio': (g[g['slot'].isin(MORNING_PEAK_SLOTS + EVENING_PEAK_SLOTS)]['d_out'].sum() + 
                            g[g['slot'].isin(MORNING_PEAK_SLOTS + EVENING_PEAK_SLOTS)]['d_in'].sum()) /
                           (g['d_out'].sum() + g['d_in'].sum() + 1e-6)
    })
).reset_index()

site_stats = site_stats.merge(peak_ratios, on='h3_id')

# 潮汐指数符号：正表示净流入（还车>借车），负表示净流出（借车>还车）
site_stats['tide_positive_ratio'] = df_all.groupby('h3_id')['tide_index'].apply(
    lambda x: (x > 0).mean()
).values

# 按cluster汇总
print("📈 按 cluster 汇总统计...")
cluster_summary = site_stats.groupby('cluster').agg({
    'total_flow': ['mean', 'median'],
    'avg_tide': ['mean', 'std'],
    'morning_out_ratio': 'mean',
    'morning_in_ratio': 'mean',
    'evening_out_ratio': 'mean',
    'evening_in_ratio': 'mean',
    'peak_flow_ratio': 'mean',
    'tide_positive_ratio': 'mean'
}).round(4)

# 添加站点数量
cluster_summary['site_count'] = site_stats.groupby('cluster').size()

print("\n" + "="*60)
print("📋 Cluster 特征对比表")
print("="*60)
print(cluster_summary.to_string())

# 保存表格
cluster_summary.to_csv(OUTPUT_DIR / "cluster_features.csv")
print(f"\n✅ 特征表已保存至 {OUTPUT_DIR / 'cluster_features.csv'}")

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 早晚高峰借车占比对比（morning_out_ratio, evening_out_ratio）
ax = axes[0, 0]
metrics = ['morning_out_ratio', 'evening_out_ratio']
x = np.arange(len(metrics))
width = 0.35
for i, cl in enumerate(sorted(site_stats['cluster'].unique())):
    sub = site_stats[site_stats['cluster'] == cl]
    values = [sub[m].mean() for m in metrics]
    ax.bar(x + i*width, values, width, label=f'Cluster {cl}')
ax.set_xticks(x + width/2)
ax.set_xticklabels(['早高峰借车占比', '晚高峰借车占比'])
ax.set_ylabel('占比')
ax.set_title('早晚高峰借车量占比')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)

# 2. 潮汐指数均值对比
ax = axes[0, 1]
clusters = sorted(site_stats['cluster'].unique())
means = [site_stats[site_stats['cluster']==cl]['avg_tide'].mean() for cl in clusters]
std = [site_stats[site_stats['cluster']==cl]['avg_tide'].std() for cl in clusters]
ax.bar(clusters, means, yerr=std, capsize=5, color=['#1f77b4', '#ff7f0e'])
ax.set_xlabel('Cluster')
ax.set_ylabel('平均潮汐指数')
ax.set_title('潮汐指数均值 (±标准差)')
ax.axhline(y=0, color='gray', linestyle='--')
ax.grid(axis='y', linestyle='--', alpha=0.5)

# 3. 总流量对比（箱线图）
ax = axes[1, 0]
data = [site_stats[site_stats['cluster']==cl]['total_flow'].values for cl in clusters]
bp = ax.boxplot(data, labels=[f'Cluster {cl}' for cl in clusters], patch_artist=True)
bp['boxes'][0].set_facecolor('#1f77b4')
bp['boxes'][1].set_facecolor('#ff7f0e')
ax.set_ylabel('总流量 (D_in + D_out)')
ax.set_title('站点总流量分布')
ax.grid(axis='y', linestyle='--', alpha=0.5)

# 4. 潮汐指数正负比例（净流入占比）
ax = axes[1, 1]
pos_ratios = [site_stats[site_stats['cluster']==cl]['tide_positive_ratio'].mean() for cl in clusters]
ax.bar(clusters, pos_ratios, color=['#1f77b4', '#ff7f0e'])
ax.set_xlabel('Cluster')
ax.set_ylabel('潮汐指数为正的时间比例')
ax.set_title('净流入时段占比')
ax.set_ylim(0, 1)
ax.axhline(y=0.5, color='gray', linestyle='--', label='平衡线')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cluster_comparison.png", dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ 对比图已保存至 {OUTPUT_DIR / 'cluster_comparison.png'}")

# 生成论文文字描述
print("\n" + "="*60)
print("📝 论文功能区描述建议")
print("="*60)

for cl in sorted(site_stats['cluster'].unique()):
    sub = site_stats[site_stats['cluster'] == cl]
    count = len(sub)
    avg_tide = sub['avg_tide'].mean()
    morning_out = sub['morning_out_ratio'].mean()
    evening_out = sub['evening_out_ratio'].mean()
    tide_pos = sub['tide_positive_ratio'].mean()
    total_flow_median = sub['total_flow'].median()
    
    # 基于流量规模判定功能区类型
    if total_flow_median > 100000:
        func_type = "高流量核心站点"
        reason = f"总流量中位数达 {total_flow_median:,.0f}，是低流量站点的 {total_flow_median / 1378:.0f} 倍，承载了城市主要的骑行需求"
    else:
        func_type = "低流量背景站点"
        reason = f"总流量中位数仅 {total_flow_median:,.0f}，骑行需求分散且量小，对调度策略的敏感度低"
    
    print(f"\nCluster {cl} ({func_type}，{count} 个站点):")
    print(f"  - 平均潮汐指数: {avg_tide:.3f} (正=净流入)")
    print(f"  - 早高峰借车占比: {morning_out:.2%}")
    print(f"  - 晚高峰借车占比: {evening_out:.2%}")
    print(f"  - 净流入时段比例: {tide_pos:.1%}")
    print(f"  - 总流量中位数: {total_flow_median:,.0f}")
    print(f"  - 解读: {reason}")

print("\n✅ 分析完成！")