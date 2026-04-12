import pandas as pd
import numpy as np
import os
import sys
import argparse
from typing import Dict, List

DEFAULT_CAPACITY = 50

def load_capacity(capacity_path: str) -> Dict[str, float]:
    if not os.path.exists(capacity_path):
        print(f"⚠️ 未找到容量文件 {capacity_path}，使用默认容量 {DEFAULT_CAPACITY}")
        return {}
    cap_df = pd.read_csv(capacity_path)
    cap_dict = dict(zip(cap_df['h3_id'].astype(str).str.lower(), cap_df['capacity']))
    return cap_dict

def compute_detailed_metrics(df: pd.DataFrame, cap_dict: Dict[str, float],
                              peak_slots: List[int] = None) -> Dict:
    
    # 计算详细评价指标
    df = df.copy()
    df['h3_id'] = df['h3_id'].astype(str).str.lower()
    df['capacity'] = df['h3_id'].map(cap_dict).fillna(DEFAULT_CAPACITY)
    df['D_in'] = df['D_in'].astype(float)
    df['D_out'] = df['D_out'].astype(float)
    df['S_t'] = df['S_t'].astype(float)

    df['L'] = np.maximum(0, df['D_out'] - df['S_t'])
    df['O'] = np.maximum(0, df['S_t'] - df['capacity'])
    df['denom'] = df['D_in'] + df['D_out']
    df['F'] = (df['L'] + df['O']) / df['denom']
    df.loc[df['denom'] == 0, 'F'] = 0.0

    # 总流量权重
    df['weight'] = df['denom']

    # 按站点汇总总缺车量（用于调度效益计算）
    total_L = df['L'].sum()
    total_O = df['O'].sum()
    total_flow = df['denom'].sum()

    # 简单平均失败率（按时段）
    simple_avg = df.groupby('time_slot')['F'].mean()

    # 加权平均失败率（按时段）
    weighted_avg = df.groupby('time_slot').apply(
        lambda g: np.average(g['F'], weights=g['weight']) if g['weight'].sum() > 0 else 0
    )

    # 全时段平均（简单/加权）
    overall_simple = simple_avg.mean()
    overall_weighted = weighted_avg.mean()

    # 高峰时段指标
    peak_metrics = {}
    if peak_slots:
        peak_df = df[df['time_slot'].isin(peak_slots)]
        if not peak_df.empty:
            peak_simple = peak_df['F'].mean()
            peak_weighted = np.average(peak_df['F'], weights=peak_df['weight']) if peak_df['weight'].sum() > 0 else 0
            peak_total_L = peak_df['L'].sum()
            peak_metrics = {
                'peak_slots': peak_slots,
                'peak_simple_failure': peak_simple,
                'peak_weighted_failure': peak_weighted,
                'peak_total_shortage': peak_total_L
            }

    return {
        'total_shortage': total_L,
        'total_overflow': total_O,
        'total_flow': total_flow,
        'overall_simple_failure': overall_simple,
        'overall_weighted_failure': overall_weighted,
        'per_slot_simple': simple_avg,
        'per_slot_weighted': weighted_avg,
        'peak_metrics': peak_metrics
    }

def main():
    parser = argparse.ArgumentParser(description='增强版失败率评价')
    parser.add_argument('--input', type=str, required=True,
                        help='输入CSV文件路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出CSV文件路径（按时段失败率）')
    parser.add_argument('--capacity', type=str, default='data/capacity.csv',
                        help='容量文件路径')
    parser.add_argument('--peak-slots', type=str, default='15,16,17,36,37',
                        help='高峰时段slot列表，逗号分隔，例如 15,16,17,36,37')
    parser.add_argument('--compare-with', type=str, default=None,
                        help='对比的自然状态文件路径，用于计算调度效益')
    args = parser.parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_failure{ext}"

    peak_slots = [int(x.strip()) for x in args.peak_slots.split(',')] if args.peak_slots else None

    print(f"读取输入文件: {args.input}")
    df = pd.read_csv(args.input)
    required_cols = ['h3_id', 'time_slot', 'D_in', 'D_out', 'S_t']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"输入文件缺少列: {col}")

    print(f"加载容量文件: {args.capacity}")
    cap_dict = load_capacity(args.capacity)

    print("计算详细指标...")
    metrics = compute_detailed_metrics(df, cap_dict, peak_slots)

    # 保存按时段失败率（保持原输出格式）
    slot_result = pd.DataFrame({
        'time_slot': metrics['per_slot_simple'].index,
        'avg_failure_rate': metrics['per_slot_simple'].values
    })
    slot_result.to_csv(args.output, index=False)
    print(f"按时段失败率已保存至: {args.output}")

    # 打印详细报告
    print("\n" + "="*50)
    print(f"文件: {args.input}")
    print("-"*50)
    print(f"总缺车量 (L):          {metrics['total_shortage']:,.0f}")
    print(f"总超容量量 (O):        {metrics['total_overflow']:,.0f}")
    print(f"总流量 (D_in+D_out):   {metrics['total_flow']:,.0f}")
    print(f"全天简单平均失败率:    {metrics['overall_simple_failure']:.4%}")
    print(f"全天加权平均失败率:    {metrics['overall_weighted_failure']:.4%}")

    if peak_slots:
        pm = metrics['peak_metrics']
        print("-"*50)
        print(f"高峰时段 {pm['peak_slots']}:")
        print(f"  简单平均失败率:      {pm['peak_simple_failure']:.4%}")
        print(f"  加权平均失败率:      {pm['peak_weighted_failure']:.4%}")
        print(f"  高峰总缺车量:        {pm['peak_total_shortage']:,.0f}")
    print("="*50)

    # 如果提供了对比文件，计算调度效益
    if args.compare_with:
        print(f"\n对比基准文件: {args.compare_with}")
        df_base = pd.read_csv(args.compare_with)
        metrics_base = compute_detailed_metrics(df_base, cap_dict, peak_slots)

        shortage_reduction = metrics_base['total_shortage'] - metrics['total_shortage']
        reduction_rate = (shortage_reduction / metrics_base['total_shortage']) * 100 if metrics_base['total_shortage'] > 0 else 0

        print("\n调度效益分析:")
        print(f"  缺车量减少:          {shortage_reduction:,.0f} ({reduction_rate:.1f}%)")
        print(f"  简单失败率变化:      {metrics['overall_simple_failure'] - metrics_base['overall_simple_failure']:+.4%}")
        print(f"  加权失败率变化:      {metrics['overall_weighted_failure'] - metrics_base['overall_weighted_failure']:+.4%}")
        if peak_slots:
            peak_reduction = metrics_base['peak_metrics']['peak_total_shortage'] - metrics['peak_metrics']['peak_total_shortage']
            peak_rate = (peak_reduction / metrics_base['peak_metrics']['peak_total_shortage']) * 100 if metrics_base['peak_metrics']['peak_total_shortage'] > 0 else 0
            print(f"  高峰缺车量减少:      {peak_reduction:,.0f} ({peak_rate:.1f}%)")

if __name__ == "__main__":
    main()