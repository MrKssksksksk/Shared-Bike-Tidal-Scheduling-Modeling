"""
支持两种使用方式：
  1. 交互模式 (無参数) - 按提示输入
    python find_h3_region_by_radius.py

  2. 命令行模式 - 传入参数
    python scr/find_h3_region_by_radius.py <H3_ID> <半径km> [--date YYYY-MM-DD] [--holiday 0/1] [--is-preholiday 0/1]
  
示例:
  python scr/find_h3_region_by_radius.py 89411c02253ffff 5 --date 2021-05-15 --holiday 1 --is-preholiday 0
"""

import sys
import json
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent))

from h3_utils import h3_disk, save_h3_region_to_json

def show_help():
    # 显示帮助信息
    print("\n" + "="*70)
    print("🚀 H3圆形区域查询工具 - 生成prediction_target.json")
    print("="*70)
    print("\n📌 交互模式 (推荐):")
    print("   python find_h3_region_by_radius.py")
    print("\n📌 命令行模式:")
    print("   python find_h3_region_by_radius.py <H3_ID> <半径km> [options]")
    print("\n参数说明:")
    print("   <H3_ID>              中心H3格子ID (必需)")
    print("   <半径km>             查询半径，单位公里 (必需)")
    print("   --date YYYY-MM-DD    目标日期 (可选，例: 2021-05-15)")
    print("   --holiday 0/1        是否为假期 (可选，0=否 1=是，默认0)")
    print("   --is-preholiday 0/1  是否为前假日 (可选，0=否 1=是，默认0)")
    print("   --output 路径        输出文件路径 (可选，默认: data/prediction_target.json)")
    print("\n💡 示例:")
    print("   python find_h3_region_by_radius.py 89411c02253ffff 5 --date 2021-05-15 --holiday 1")
    print("="*70 + "\n")

def validate_date(date_str):
    # 验证日期格式
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def validate_h3_id(h3_id):
    # 简单验证H3 ID格式
    if isinstance(h3_id, str) and len(h3_id) > 0 and h3_id.isalnum():
        return True
    return False

def interactive_mode():
    # 交互式模式
    print("\n" + "="*70)
    print("🎯 H3圆形区域查询工具 - 交互模式")
    print("="*70)
    
    # 输入中心H3 ID
    while True:
        center_h3 = input("\n📍 请输入中心H3 ID (例: 89411c02253ffff): ").strip()
        if validate_h3_id(center_h3):
            break
        print("   ❌ 无效的H3 ID，请重新输入")
    
    # 输入半径
    while True:
        try:
            radius_km = float(input("📏 请输入查询半径 (单位: 公里, 例: 5): "))
            if radius_km > 0:
                break
            print("   ❌ 半径必须为正数，请重新输入")
        except ValueError:
            print("   ❌ 请输入有效的数字")
    
    # 输入目标日期
    while True:
        target_date = input("📅 请输入目标日期 (格式: YYYY-MM-DD, 例: 2021-05-15) [回车跳过]: ").strip()
        if target_date == "":
            target_date = None
            break
        if validate_date(target_date):
            break
        print("   ❌ 日期格式错误，请使用 YYYY-MM-DD 格式")
    
    # 输入假期标记
    while True:
        try:
            holiday = input("🎉 是否为假期? (0=否, 1=是) [默认0]: ").strip() or "0"
            holiday = int(holiday)
            if holiday in [0, 1]:
                break
            print("   ❌ 请输入 0 或 1")
        except ValueError:
            print("   ❌ 请输入有效的数字")
    
    # 输入前假日标记
    while True:
        try:
            is_preholiday = input("🎊 是否为前假日? (0=否, 1=是) [默认0]: ").strip() or "0"
            is_preholiday = int(is_preholiday)
            if is_preholiday in [0, 1]:
                break
            print("   ❌ 请输入 0 或 1")
        except ValueError:
            print("   ❌ 请输入有效的数字")
    
    output_path = input("💾 输出文件路径 [默认: data/prediction_target.json]: ").strip() or "data/prediction_target.json"
    
    print("\n" + "-"*70)
    process_query(center_h3, radius_km, target_date, holiday, is_preholiday, output_path)

def process_query(center_h3, radius_km, target_date, holiday, is_preholiday, output_path):
    # 执行查询和输出
    print(f"🔍 查询区域: 中心H3={center_h3}, 半径={radius_km}km")
    
    try:
        h3_ids = h3_disk(center_h3, radius_km)
    except Exception as e:
        print(f"❌ 错误: {e}")
        return
    
    print(f"✅ 找到 {len(h3_ids)} 个H3网格")
    print(f"   样本: {h3_ids[:5]}{'...' if len(h3_ids) > 5 else ''}")
    
    # 如果指定了日期，直接生成JSON文件
    if target_date:
        save_h3_region_to_json(h3_ids, target_date, output_path, holiday, is_preholiday)
        print(f"\n✨ 文件已生成: {output_path}")
    else:
        # 输出JSON格式供复制
        config = {
            "target_date": "YYYY-MM-DD (请填入)",
            "holiday": holiday,
            "is_preholiday": is_preholiday,
            "region_h3_list": h3_ids
        }
        print("\n📋 JSON 格式 (可复制到 data/prediction_target.json):")
        print(json.dumps(config, indent=2, ensure_ascii=False))
        
        # 提示用户可以保存
        save_choice = input("\n💾 是否保存为JSON文件? (y/n) [默认n]: ").strip().lower()
        if save_choice == 'y':
            save_h3_region_to_json(h3_ids, "YYYY-MM-DD", output_path, holiday, is_preholiday)

def command_line_mode(argv):
    # 命令行模式
    if len(argv) < 3:
        show_help()
        return
    
    center_h3 = argv[1]
    try:
        radius_km = float(argv[2])
    except ValueError:
        print(f"❌ 错误: 半径必须是数字，收到: {argv[2]}")
        return
    
    target_date = None
    holiday = 0
    is_preholiday = 0
    output_path = "data/prediction_target.json"
    
    # 解析可选参数
    i = 3
    while i < len(argv):
        if argv[i] == "--date" and i + 1 < len(argv):
            target_date = argv[i + 1]
            if not validate_date(target_date):
                print(f"❌ 错误: 日期格式错误 {target_date}，请使用 YYYY-MM-DD")
                return
            i += 2
        elif argv[i] == "--holiday" and i + 1 < len(argv):
            try:
                holiday = int(argv[i + 1])
                if holiday not in [0, 1]:
                    print("❌ 错误: holiday 必须为 0 或 1")
                    return
            except ValueError:
                print(f"❌ 错误: holiday 必须是整数，收到: {argv[i + 1]}")
                return
            i += 2
        elif argv[i] == "--is-preholiday" and i + 1 < len(argv):
            try:
                is_preholiday = int(argv[i + 1])
                if is_preholiday not in [0, 1]:
                    print("❌ 错误: is-preholiday 必须为 0 或 1")
                    return
            except ValueError:
                print(f"❌ 错误: is-preholiday 必须是整数，收到: {argv[i + 1]}")
                return
            i += 2
        elif argv[i] == "--output" and i + 1 < len(argv):
            output_path = argv[i + 1]
            i += 2
        elif argv[i] == "--help" or argv[i] == "-h":
            show_help()
            return
        else:
            print(f"⚠️ 警告: 未知参数 {argv[i]}")
            i += 1
    
    if not validate_h3_id(center_h3):
        print(f"❌ 错误: 无效的H3 ID: {center_h3}")
        return
    
    if radius_km <= 0:
        print(f"❌ 错误: 半径必须为正数，收到: {radius_km}")
        return
    
    process_query(center_h3, radius_km, target_date, holiday, is_preholiday, output_path)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # 交互模式
        interactive_mode()
    elif len(sys.argv) >= 2 and sys.argv[1] in ["--help", "-h"]:
        show_help()
    else:
        # 命令行模式
        command_line_mode(sys.argv)
