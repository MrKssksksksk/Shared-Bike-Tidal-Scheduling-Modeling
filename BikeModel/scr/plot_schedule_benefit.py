"""
plot_schedule_benefit.py - 绘制调度效益对比柱状图（总缺车量、加权失败率等）
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import platform

# 中文字体设置
if platform.system() == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
elif platform.system() == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC']
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
natural = pd.read_csv("data/natural_allocation.csv")
scheduled = pd.read_csv("data/allocation.csv")
capacity_df = pd.read_csv("data/capacity.csv")
cap_dict = dict(zip(capacity_df['h3_id'], capacity_df['capacity']))

# 辅助函数：计算缺车总量 L 和超容量总量 O
def compute_metrics(df):
    df = df.copy()
    df['capacity'] = df['h3_id'].map(cap_dict).fillna(50)
    df['L'] = np.maximum(0, df['D_out'] - df['S_t'])
    df['O'] = np.maximum(0, df['S_t'] - df['capacity'])
    df['denom'] = df['D_in'] + df['D_out']
    df['F'] = (df['L'] + df['O']) / df['denom']
    df.loc[df['denom'] == 0, 'F'] = 0.0
    total_L = df['L'].sum()
    total_O = df['O'].sum()
    weighted_F = np.average(df['F'], weights=df['denom'])
    return total_L, total_O, weighted_F

nat_L, nat_O, nat_F = compute_metrics(natural)
sch_L, sch_O, sch_F = compute_metrics(scheduled)

# 高峰时段
peak_slots = [15,16,17,36,37]
nat_peak = natural[natural['time_slot'].isin(peak_slots)]
sch_peak = scheduled[scheduled['time_slot'].isin(peak_slots)]
nat_peak_L, _, _ = compute_metrics(nat_peak)
sch_peak_L, _, _ = compute_metrics(sch_peak)

# 绘图
categories = ['总缺车量', '高峰缺车量', '加权失败率']
nat_values = [nat_L, nat_peak_L, nat_F * 100]   # 失败率转为百分比
sch_values = [sch_L, sch_peak_L, sch_F * 100]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
bars1 = ax.bar(x - width/2, nat_values, width, label='自然状态', color='#1f77b4')
bars2 = ax.bar(x + width/2, sch_values, width, label='调度后', color='#ff7f0e')

# 添加数值标签
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

ax.set_ylabel('数值', fontsize=12)
ax.set_title('调度效益对比', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)

# 添加减少百分比注释
reduction_L = (nat_L - sch_L) / nat_L * 100
reduction_peak = (nat_peak_L - sch_peak_L) / nat_peak_L * 100
ax.text(0.5, 0.9, f'总缺车减少 {reduction_L:.1f}%\n高峰缺车减少 {reduction_peak:.1f}%',
        transform=ax.transAxes, fontsize=11, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('outputs/schedule_benefit.png', dpi=300, bbox_inches='tight')
plt.show()
print("调度效益对比图已保存至 outputs/schedule_benefit.png")