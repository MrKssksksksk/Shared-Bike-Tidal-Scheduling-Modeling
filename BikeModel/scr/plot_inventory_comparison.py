import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import platform

# 设置中文字体
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
tasks = pd.read_csv("data/schedule_tasks.csv")

# 获取调度时刻（去重）
schedule_times = sorted(tasks['schedule_time'].unique())
print(f"调度时刻: {schedule_times}")

# 选择 4 个有调度量的站点（按净变化量绝对值排序取前四）
top_h3 = tasks.groupby('h3_id')['net_change'].apply(lambda x: x.abs().sum()).nlargest(4).index.tolist()
print(f"展示站点: {top_h3}")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.flatten()

for i, h3 in enumerate(top_h3):
    ax = axes[i]
    # 提取该站点数据并按 time_slot 排序
    nat = natural[natural['h3_id'] == h3].sort_values('time_slot').reset_index(drop=True)
    sch = scheduled[scheduled['h3_id'] == h3].sort_values('time_slot').reset_index(drop=True)
    
    if nat.empty or sch.empty:
        ax.text(0.5, 0.5, '数据缺失', ha='center', va='center', transform=ax.transAxes)
        continue
    
    ax.plot(nat['time_slot'], nat['S_t'], 'b-', label='自然状态', linewidth=2)
    ax.plot(sch['time_slot'], sch['S_t'], 'r--', label='调度后', linewidth=2)
    
    # 标注调度时刻（垂直线）
    for st in schedule_times:
        ax.axvline(x=st, color='green', linestyle=':', alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel('时间槽 (0.5小时/槽)', fontsize=11)
    ax.set_ylabel('车辆数', fontsize=11)
    ax.set_title(f'站点 {h3[:12]}...', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 设置合理的 y 轴下限为 0
    ax.set_ylim(bottom=0)

# 添加图例说明调度线
handles, labels = ax.get_legend_handles_labels()
handles.append(plt.Line2D([0], [0], color='green', linestyle=':', linewidth=1.5))
labels.append('调度时刻')
fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.02))

plt.tight_layout()
plt.savefig('outputs/inventory_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("图表已保存至 outputs/inventory_comparison.png")