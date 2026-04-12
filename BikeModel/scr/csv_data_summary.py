import os
import sys
import csv

def get_csv_info(file_path):
    # 获取 CSV 文件的行数、列数、文件大小（优化版：不加载整个文件到内存）
    try:
        file_size = os.path.getsize(file_path)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            # 读第一行获取列数
            first_row = next(reader, None)
            if first_row is None:
                return 0, 0, file_size
            
            num_cols = len(first_row)
            num_rows = 1  # 已经读了第一行
            
            # 逐行计数，不加载到内存
            for _ in reader:
                num_rows += 1
            
            return num_rows, num_cols, file_size
    except Exception as e:
        print(f"❌ 读取 {file_path} 时出错: {e}")
        return 0, 0, 0

def main(folder_path):
    """
    主函数：扫描文件夹，汇总 CSV 数据量
    """
    if not os.path.isdir(folder_path):
        print("❌ 无效的文件夹路径")
        return

    csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]
    if not csv_files:
        print("📂 文件夹中没有找到 CSV 文件")
        return

    print(f"📂 在 {folder_path} 中找到 {len(csv_files)} 个 CSV 文件\n")

    total_rows = 0
    total_size = 0
    file_summaries = []

    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        rows, cols, size = get_csv_info(file_path)
        total_rows += rows
        total_size += size
        file_summaries.append((csv_file, rows, cols, size))
        print(f"📄 {csv_file}: {rows:,} 行, {cols} 列, {size:,} 字节")

    print("\n" + "="*50)
    print(f"📊 总计: {total_rows:,} 行, {total_size:,} 字节")
    print(f"📂 文件夹: {folder_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python csv_data_summary.py <文件夹路径>")
        print("例如: python csv_data_summary.py data/")
        sys.exit(1)

    main(sys.argv[1])