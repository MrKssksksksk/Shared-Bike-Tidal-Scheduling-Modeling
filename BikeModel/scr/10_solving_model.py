import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, List, Tuple, Optional

# 配置
PRED_PATH = "data/prediction.csv"
CAPACITY_PATH = "data/capacity.csv"
OUTPUT_PATH = "data/allocation.csv"
NATURAL_INDICATORS_PATH = "data/natural_indicators.csv"
NATURAL_ALLOCATION_PATH = "data/natural_allocation.csv"
SCHEDULE_TASKS_PATH = "data/schedule_tasks.csv"      # 新增：调度任务输出路径

# 评价模型权重
ALPHA = 1.0          # 缺车惩罚权重（越大对缺车越敏感）
BETA = 0.3           # 超容量惩罚权重（越大对拥挤越敏感，0 则忽略容量）

# 高峰识别方法：'tide', 'satisfaction', 'demand', 'fixed'
PEAK_ID_METHOD = 'demand'

# 潮汐指数阈值（用于 tide 方法）
TIDE_THRESHOLD = 0.1
# 满意度阈值（用于 satisfaction 方法，低于此值视为高峰）
SATISFACTION_THRESHOLD = 0.6
# 需求方法参数（用于 demand 方法）
DEMAND_ALPHA = 1.5    # 均值 + alpha * 标准差 作为阈值
DEMAND_TYPE = 'out'   # 'out' 或 'total'
# 固定高峰时段（用于 fixed 方法，格式：[[start_slot, end_slot], ...]）
FIXED_PEAK_PERIODS = [[14, 17], [35, 37]]

MIN_PEAK_DURATION = 2          # 最少连续时段数
DEFAULT_CAPACITY = 50          # 当 capacity.csv 缺失时的默认容量
MAX_X_CHANGE = 200             # 每个站点最大调度变化量
MIN_SCHEDULE_QUANTITY = 5      # 新增：最小调度量，低于此值的站点不输出到任务文件
VERBOSE = True

SCHEDULE_ADVANCE_SLOTS = 2  # 新增：调度提前量（时段数），例如 2 表示在高峰开始前2个时段进行调度

# 辅助函数

def load_prediction(pred_path: str) -> pd.DataFrame:
    df = pd.read_csv(pred_path)
    required_cols = ['h3_id', 'time_slot', 'D_in', 'D_out', 'S_t']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"prediction.csv 缺少列: {col}")
    df['time_slot'] = df['time_slot'].astype(int)
    return df

def compute_natural_inventory(pred_df: pd.DataFrame) -> pd.DataFrame:
    """递推自然状态车辆数（全时段，非负，取整）"""
    h3_ids = pred_df['h3_id'].unique()
    time_slots = sorted(pred_df['time_slot'].unique())
    n_slots = len(time_slots)
    
    D_in_dict, D_out_dict = {}, {}
    for h3 in h3_ids:
        sub = pred_df[pred_df['h3_id'] == h3].sort_values('time_slot')
        D_in_dict[h3] = sub['D_in'].values
        D_out_dict[h3] = sub['D_out'].values
    
    S0_df = pred_df[pred_df['time_slot'] == 0][['h3_id', 'S_t']]
    S_initial = dict(zip(S0_df['h3_id'], S0_df['S_t']))
    
    records = []
    for h3 in h3_ids:
        S = np.zeros(n_slots)
        S[0] = max(0.0, S_initial.get(h3, 0.0))
        for t in range(n_slots - 1):
            net = D_in_dict[h3][t] - D_out_dict[h3][t]
            S[t+1] = max(0.0, S[t] + net)
        S = np.round(S).astype(int)
        for t_idx, t in enumerate(time_slots):
            records.append({'h3_id': h3, 'time_slot': t, 'S_natural': S[t_idx]})
    return pd.DataFrame(records)

def load_or_estimate_capacity(pred_df: pd.DataFrame, capacity_path: str) -> Dict[str, float]:
    """加载容量文件，若缺失则使用默认值"""
    if os.path.exists(capacity_path):
        cap_df = pd.read_csv(capacity_path)
        if 'h3_id' not in cap_df.columns or 'capacity' not in cap_df.columns:
            raise ValueError("capacity.csv 需要包含 h3_id 和 capacity 列")
        cap_dict = dict(zip(cap_df['h3_id'], cap_df['capacity']))
        if VERBOSE:
            print(f"从 {capacity_path} 加载容量，共 {len(cap_dict)} 个站点")
        return cap_dict
    
    if VERBOSE:
        print(f"⚠️ 未找到容量文件 {capacity_path}，使用默认容量 {DEFAULT_CAPACITY}")
    h3_ids = pred_df['h3_id'].unique()
    return {h3: DEFAULT_CAPACITY for h3 in h3_ids}

# 自然状态指标（含容量惩罚）
def compute_and_save_natural_indicators(pred_df: pd.DataFrame,
                                        capacity_dict: Dict[str, float],
                                        alpha: float,
                                        beta: float,
                                        output_path: str):
    """计算自然状态下的逐时段指标（潮汐指数、满意度、总需求）"""
    if VERBOSE:
        print("计算自然状态指标...")
    h3_ids = pred_df['h3_id'].unique()
    time_slots = sorted(pred_df['time_slot'].unique())
    n_slots = len(time_slots)

    D_in_dict, D_out_dict = {}, {}
    for h3 in h3_ids:
        sub = pred_df[pred_df['h3_id'] == h3].sort_values('time_slot')
        D_in_dict[h3] = sub['D_in'].values
        D_out_dict[h3] = sub['D_out'].values

    S_natural_df = compute_natural_inventory(pred_df)
    S_natural = {}
    for h3 in h3_ids:
        sub = S_natural_df[S_natural_df['h3_id'] == h3].sort_values('time_slot')
        S_natural[h3] = sub['S_natural'].values

    indicators = []
    for t_idx in range(n_slots):
        tide_sum = 0.0
        u_sum = 0.0
        demand_sum = 0.0
        count = 0
        for h3 in h3_ids:
            d_in = D_in_dict[h3][t_idx]
            d_out = D_out_dict[h3][t_idx]
            S = S_natural[h3][t_idx]
            C = capacity_dict.get(h3, DEFAULT_CAPACITY)

            tide = (d_in - d_out) / (d_in + d_out + 1e-6)
            tide_sum += abs(tide)

            # 缺车惩罚
            L = max(0.0, d_out - S)
            # 超容量惩罚（考虑还车后可能超容）
            S_after_return = max(0.0, S - d_out + d_in)
            O = max(0.0, S_after_return - C)
            u = np.exp(-alpha * L - beta * O)
            u_sum += u

            demand_sum += d_out
            count += 1

        avg_tide = tide_sum / count if count > 0 else 0.0
        avg_satisfaction = u_sum / count if count > 0 else 1.0
        indicators.append([t_idx, avg_tide, avg_satisfaction, demand_sum])

    df_ind = pd.DataFrame(indicators,
                          columns=['time_slot', 'avg_tide_abs', 'avg_satisfaction', 'total_demand'])
    df_ind.to_csv(output_path, index=False)
    if VERBOSE:
        print(f"自然状态指标已保存至 {output_path}")
    return df_ind

# ---------- 高峰识别函数 ----------
def find_peak_periods_by_tide(pred_df: pd.DataFrame,
                              threshold: float = 0.3,
                              min_duration: int = 2) -> List[List[int]]:
    h3_ids = pred_df['h3_id'].unique()
    time_slots = sorted(pred_df['time_slot'].unique())
    n_slots = len(time_slots)
    D_in_dict, D_out_dict = {}, {}
    for h3 in h3_ids:
        sub = pred_df[pred_df['h3_id'] == h3].sort_values('time_slot')
        D_in_dict[h3] = sub['D_in'].values
        D_out_dict[h3] = sub['D_out'].values
    tide_abs = []
    for t_idx in range(n_slots):
        tide_sum = 0.0
        count = 0
        for h3 in h3_ids:
            d_in = D_in_dict[h3][t_idx]
            d_out = D_out_dict[h3][t_idx]
            tide = (d_in - d_out) / (d_in + d_out + 1e-6)
            tide_sum += abs(tide)
            count += 1
        tide_abs.append(tide_sum / count if count > 0 else 0.0)
    peak_mask = np.array(tide_abs) > threshold
    periods = []
    i = 0
    while i < len(peak_mask):
        if peak_mask[i]:
            start = i
            while i < len(peak_mask) and peak_mask[i]:
                i += 1
            end = i - 1
            if end - start + 1 >= min_duration:
                periods.append([start, end])
        else:
            i += 1
    return periods

def find_peak_periods_by_satisfaction(S_natural_df: pd.DataFrame,
                                      D_in_dict, D_out_dict,
                                      capacity_dict: Dict[str, float],
                                      alpha: float, beta: float,
                                      threshold: float = 0.6,
                                      min_duration: int = 2) -> Tuple[List[List[int]], List[float]]:
    h3_ids = S_natural_df['h3_id'].unique()
    time_slots = sorted(S_natural_df['time_slot'].unique())
    n_slots = len(time_slots)
    S_natural = {}
    for h3 in h3_ids:
        sub = S_natural_df[S_natural_df['h3_id'] == h3].sort_values('time_slot')
        S_natural[h3] = sub['S_natural'].values
    U_t = []
    for t_idx in range(n_slots):
        u_sum = 0.0
        count = 0
        for h3 in h3_ids:
            S = S_natural[h3][t_idx]
            d_out = D_out_dict[h3][t_idx]
            d_in = D_in_dict[h3][t_idx]
            C = capacity_dict.get(h3, DEFAULT_CAPACITY)
            L = max(0.0, d_out - S)
            S_after = max(0.0, S - d_out + d_in)
            O = max(0.0, S_after - C)
            u = np.exp(-alpha * L - beta * O)
            u_sum += u
            count += 1
        U_t.append(u_sum / count if count > 0 else 1.0)
    low_mask = np.array(U_t) < threshold
    periods = []
    i = 0
    while i < len(low_mask):
        if low_mask[i]:
            start = i
            while i < len(low_mask) and low_mask[i]:
                i += 1
            end = i - 1
            if end - start + 1 >= min_duration:
                periods.append([start, end])
        else:
            i += 1
    return periods, U_t

def find_peak_periods_by_demand(pred_df: pd.DataFrame,
                                demand_type: str = 'out',
                                alpha_factor: float = 1.5,
                                min_duration: int = 2) -> List[List[int]]:
    h3_ids = pred_df['h3_id'].unique()
    time_slots = sorted(pred_df['time_slot'].unique())
    n_slots = len(time_slots)
    D_in_dict, D_out_dict = {}, {}
    for h3 in h3_ids:
        sub = pred_df[pred_df['h3_id'] == h3].sort_values('time_slot')
        D_in_dict[h3] = sub['D_in'].values
        D_out_dict[h3] = sub['D_out'].values
    total_demand = []
    for t_idx in range(n_slots):
        if demand_type == 'out':
            demand = sum(D_out_dict[h3][t_idx] for h3 in h3_ids)
        else:
            demand = sum(D_in_dict[h3][t_idx] + D_out_dict[h3][t_idx] for h3 in h3_ids)
        total_demand.append(demand)
    mean_val = np.mean(total_demand)
    std_val = np.std(total_demand)
    threshold = mean_val + alpha_factor * std_val
    peak_mask = np.array(total_demand) > threshold
    periods = []
    i = 0
    while i < len(peak_mask):
        if peak_mask[i]:
            start = i
            while i < len(peak_mask) and peak_mask[i]:
                i += 1
            end = i - 1
            if end - start + 1 >= min_duration:
                periods.append([start, end])
        else:
            i += 1
    return periods

# 核心优化函数（含容量惩罚）
def site_objective(x: float,
                   D_out_seq: List[float],
                   D_in_seq: List[float],
                   C: float,
                   alpha: float,
                   beta: float) -> float:
    """
    目标函数：对一个站点，给定高峰初始车辆数 x，
    模拟高峰时段内各时段的满意度并求和。
    D_out_seq, D_in_seq 是高峰时段内各时段的借出/还入量（长度相同）
    """
    total_u = 0.0
    S = x
    for d_out, d_in in zip(D_out_seq, D_in_seq):
        # 缺车量
        L = max(0.0, d_out - S)
        # 模拟还车后的库存
        S_after_return = max(0.0, S - d_out + d_in)
        # 超容量量
        O = max(0.0, S_after_return - C)
        u = np.exp(-alpha * L - beta * O)
        total_u += u
        S = S_after_return  # 更新库存用于下一时段
    return total_u

def optimal_x_for_site(S_orig: float,
                       D_out_seq: List[float],
                       D_in_seq: List[float],
                       C: float,
                       alpha: float,
                       beta: float,
                       max_change: int = 200) -> Tuple[int, float]:
    low = max(0, int(S_orig) - max_change)
    high = int(S_orig) + max_change
    high = min(high, 5000)
    best_x = int(round(S_orig))
    best_val = site_objective(best_x, D_out_seq, D_in_seq, C, alpha, beta)
    for x in range(low, high+1):
        val = site_objective(x, D_out_seq, D_in_seq, C, alpha, beta)
        if val > best_val:
            best_val = val
            best_x = x
    return best_x, best_val

def solve_allocation(pred_df: pd.DataFrame,
                     capacity_dict: Dict[str, float],
                     alpha: float = 1.0,
                     beta: float = 0.3,
                     peak_id_method: str = 'tide',
                     tide_threshold: float = 0.3,
                     satisfaction_threshold: float = 0.6,
                     demand_alpha: float = 1.5,
                     demand_type: str = 'out',
                     fixed_peak_periods: Optional[List[List[int]]] = None,
                     min_peak_duration: int = 2,
                     max_x_change: int = 200) -> pd.DataFrame:
    
    h3_ids = pred_df['h3_id'].unique()
    time_slots = sorted(pred_df['time_slot'].unique())
    n_slots = len(time_slots)
    
    D_in_dict, D_out_dict = {}, {}
    for h3 in h3_ids:
        sub = pred_df[pred_df['h3_id'] == h3].sort_values('time_slot')
        D_in_dict[h3] = sub['D_in'].values
        D_out_dict[h3] = sub['D_out'].values
    
    S_natural_df = compute_natural_inventory(pred_df)
    S_natural = {}
    for h3 in h3_ids:
        sub = S_natural_df[S_natural_df['h3_id'] == h3].sort_values('time_slot')
        S_natural[h3] = sub['S_natural'].values
    
    # 保存自然状态全时段数据
    natural_rows = []
    for h3 in h3_ids:
        for t_idx, t in enumerate(time_slots):
            natural_rows.append({
                'h3_id': h3,
                'time_slot': t,
                'D_in': D_in_dict[h3][t_idx],
                'D_out': D_out_dict[h3][t_idx],
                'S_t': S_natural[h3][t_idx]
            })
    natural_df = pd.DataFrame(natural_rows)
    natural_df = natural_df[['h3_id', 'time_slot', 'D_in', 'D_out', 'S_t']]
    os.makedirs(os.path.dirname(NATURAL_ALLOCATION_PATH), exist_ok=True)
    natural_df.to_csv(NATURAL_ALLOCATION_PATH, index=False)
    if VERBOSE:
        print(f"自然状态全时段数据已保存至 {NATURAL_ALLOCATION_PATH}")
    
    # 自然指标文件
    if not os.path.exists(NATURAL_INDICATORS_PATH):
        compute_and_save_natural_indicators(pred_df, capacity_dict, alpha, beta, NATURAL_INDICATORS_PATH)
    else:
        if VERBOSE:
            print(f"自然指标文件已存在，直接读取: {NATURAL_INDICATORS_PATH}")
        pd.read_csv(NATURAL_INDICATORS_PATH)
    
    S_matrix = {h3: S_natural[h3].copy() for h3 in h3_ids}
    
    # 识别高峰时段
    if peak_id_method == 'tide':
        peak_periods = find_peak_periods_by_tide(pred_df, threshold=tide_threshold, min_duration=min_peak_duration)
        if VERBOSE:
            print(f"基于潮汐指数识别的高峰时段: {peak_periods}")
    elif peak_id_method == 'satisfaction':
        peak_periods, _ = find_peak_periods_by_satisfaction(
            S_natural_df, D_in_dict, D_out_dict, capacity_dict,
            alpha, beta, threshold=satisfaction_threshold, min_duration=min_peak_duration
        )
        if VERBOSE:
            print(f"基于满意度识别的高峰时段: {peak_periods}")
    elif peak_id_method == 'demand':
        peak_periods = find_peak_periods_by_demand(pred_df, demand_type, demand_alpha, min_peak_duration)
        if VERBOSE:
            print(f"基于需求识别的高峰时段: {peak_periods}")
    elif peak_id_method == 'fixed':
        if fixed_peak_periods is None:
            raise ValueError("使用固定高峰时段时，必须提供 fixed_peak_periods 参数")
        peak_periods = fixed_peak_periods
        if VERBOSE:
            print(f"使用固定高峰时段: {peak_periods}")
    else:
        raise ValueError(f"未知的 peak_id_method: {peak_id_method}")
    
    # 收集调度任务记录
    schedule_records = []
    
    if not peak_periods:
        if VERBOSE:
            print("未识别出任何高峰时段，直接输出自然状态数据（无调度）")
        rows = []
        for h3 in h3_ids:
            for t_idx, t in enumerate(time_slots):
                rows.append({'h3_id': h3, 'time_slot': t,
                             'D_in': D_in_dict[h3][t_idx],
                             'D_out': D_out_dict[h3][t_idx],
                             'S_t': S_matrix[h3][t_idx]})
        out_df = pd.DataFrame(rows)
        return out_df[['h3_id', 'time_slot', 'D_in', 'D_out', 'S_t']]
    
    for start, end in peak_periods:
        if VERBOSE:
            print(f"\n优化高峰时段 [{start}, {end}]")
        S_orig_dict = {h3: S_matrix[h3][start] for h3 in h3_ids}
        
        # 准备每个站点在高峰时段内的 D_out 和 D_in 序列
        D_out_peak_dict = {}
        D_in_peak_dict = {}
        for h3 in h3_ids:
            D_out_peak_dict[h3] = D_out_dict[h3][start:end+1]
            D_in_peak_dict[h3] = D_in_dict[h3][start:end+1]
        
        opt_x_dict = {}
        for h3 in h3_ids:
            C = capacity_dict.get(h3, DEFAULT_CAPACITY)
            best_x, _ = optimal_x_for_site(S_orig_dict[h3],
                                           D_out_peak_dict[h3],
                                           D_in_peak_dict[h3],
                                           C, alpha, beta,
                                           max_change=max_x_change)
            opt_x_dict[h3] = best_x
        
        # 记录本次调度的净变化量（调度时刻为 start-SCHEDULE_ADVANCE_SLOTS，若 start=0 则取 0）
        schedule_time = max(0, start - SCHEDULE_ADVANCE_SLOTS)
        for h3 in h3_ids:
            delta = opt_x_dict[h3] - S_orig_dict[h3]
            if abs(delta) >= MIN_SCHEDULE_QUANTITY:
                schedule_records.append({
                    'schedule_time': schedule_time,
                    'peak_start': start,
                    'peak_end': end,
                    'h3_id': h3,
                    'net_change': round(delta, 2)
                })
        
        # 应用调度
        for h3 in h3_ids:
            S_matrix[h3][start] = opt_x_dict[h3]
        
        # 递推后续库存（高峰时段内）
        for t_idx in range(start, end):
            for h3 in h3_ids:
                net = D_in_dict[h3][t_idx] - D_out_dict[h3][t_idx]
                S_matrix[h3][t_idx+1] = max(0.0, S_matrix[h3][t_idx] + net)
                S_matrix[h3][t_idx+1] = round(S_matrix[h3][t_idx+1])
        
        total_before = sum(S_orig_dict.values())
        total_after = sum(opt_x_dict.values())
        change_pct = (total_after - total_before) / total_before * 100 if total_before > 0 else 0
        if VERBOSE:
            print(f"  调度前总车数: {total_before:.0f}, 调度后总车数: {total_after:.0f}, 变化: {change_pct:+.1f}%")
            if abs(change_pct) > 20:
                print(f"  ⚠️ 总车数变化超过20%，请检查调度车容量是否足够或调整 max_x_change")
    
    # 最终取整
    for h3 in h3_ids:
        S_matrix[h3] = np.round(S_matrix[h3]).astype(int)
    
    # 保存调度任务文件
    if schedule_records:
        tasks_df = pd.DataFrame(schedule_records)
        tasks_df.to_csv(SCHEDULE_TASKS_PATH, index=False)
        if VERBOSE:
            print(f"调度任务已保存至 {SCHEDULE_TASKS_PATH}，共 {len(tasks_df)} 条记录")
    else:
        if VERBOSE:
            print("无有效调度任务（净变化量均小于阈值），未生成 schedule_tasks.csv")
    
    rows = []
    for h3 in h3_ids:
        for t_idx, t in enumerate(time_slots):
            rows.append({'h3_id': h3, 'time_slot': t,
                         'D_in': D_in_dict[h3][t_idx],
                         'D_out': D_out_dict[h3][t_idx],
                         'S_t': S_matrix[h3][t_idx]})
    out_df = pd.DataFrame(rows)
    return out_df[['h3_id', 'time_slot', 'D_in', 'D_out', 'S_t']]

# 主程序
if __name__ == "__main__":
    try:
        if VERBOSE:
            print("正在加载预测数据...")
        pred_df = load_prediction(PRED_PATH)
        if VERBOSE:
            print(f"加载完成，共 {pred_df['h3_id'].nunique()} 个站点，{pred_df['time_slot'].nunique()} 个时段")
        
        capacity_dict = load_or_estimate_capacity(pred_df, CAPACITY_PATH)
        
        if VERBOSE:
            print("开始求解最优分配方案（评价模型已启用容量惩罚）...")
        allocation_df = solve_allocation(
            pred_df, capacity_dict,
            alpha=ALPHA, beta=BETA,
            peak_id_method=PEAK_ID_METHOD,
            tide_threshold=TIDE_THRESHOLD,
            satisfaction_threshold=SATISFACTION_THRESHOLD,
            demand_alpha=DEMAND_ALPHA,
            demand_type=DEMAND_TYPE,
            fixed_peak_periods=FIXED_PEAK_PERIODS if PEAK_ID_METHOD == 'fixed' else None,
            min_peak_duration=MIN_PEAK_DURATION,
            max_x_change=MAX_X_CHANGE
        )
        
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        allocation_df.to_csv(OUTPUT_PATH, index=False)
        if VERBOSE:
            print(f"调度结果已保存至 {OUTPUT_PATH}")
    
    except Exception as e:
        print(f"运行出错: {e}", file=sys.stderr)
        sys.exit(1)