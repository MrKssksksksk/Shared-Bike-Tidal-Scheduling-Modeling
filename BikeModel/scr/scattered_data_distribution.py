import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# 配置区
DATA_DIR = "data/second_preprocessed"                # 数据存放路径
OUTPUT_PLOT = "data_volume_scatter.png"  # 保存图表的文件名

# 数据收集
print("🔍 正在扫描数据文件...")
dates = []      # 存储日期
volumes = []    # 存储数据量

# 遍历目录下所有csv文件
path = Path(DATA_DIR)
csv_files = sorted(path.glob("*.csv"))  # 排序保证日期顺序

for file_path in csv_files:
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        row_count = len(df)  # 获取行数
        
        # 提取文件名作为日期 (例如 2021-05-12.csv -> 2021-05-12)
        date_str = file_path.stem
        
        dates.append(date_str)
        volumes.append(row_count)
        
        print(f"📊 {date_str}: {row_count:,} 行")
        
    except Exception as e:
        print(f"❌ 读取 {file_path} 失败: {e}")

if not dates:
    print(f"⚠️ 未找到任何数据文件，请检查 {DATA_DIR} 目录是否正确。")
else:
    # 计算总数据量
    total_rows = sum(volumes)
    avg_rows = total_rows / len(volumes) if volumes else 0
    print(f"\n📈 统计概览:")
    print(f"   文件总数: {len(dates)} 个")
    print(f"   总数据行数: {total_rows:,} 行")
    print(f"   日均数据量: {avg_rows:,.0f} 行")
    
    # 绘图准备
    # 将日期字符串转换为 datetime 对象以便在X轴上正确排序和显示
    date_objs = pd.to_datetime(dates)
    
    # 绘制散点图
    plt.figure(figsize=(14, 7))
    
    # 绘制散点，颜色可以根据数据量深浅变化 cmap='viridis'
    scatter = plt.scatter(date_objs, volumes, c=volumes, cmap='viridis', 
                         s=60, alpha=0.8, edgecolors='w', linewidth=0.5)
    
    # 美化图表
    # 在标题中添加总数据量信息
    plt.title(f'共享单车数据量时间分布\n（总计 {total_rows:,} 条记录，日均 {avg_rows:,.0f} 条）', 
              fontsize=14, pad=20)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('数据行数（条）', fontsize=12)
    
    # 旋转X轴标签，防止重叠
    plt.xticks(rotation=45, ha='right')
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    
    # 添加颜色条，表示数据量大小
    cbar = plt.colorbar(scatter, label='数据量')
    
    # 自动调整布局，防止标签被截断
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
    print(f"\n✅ 图表已生成并保存为: {OUTPUT_PLOT}")
    
    # 显示图表
    plt.show()