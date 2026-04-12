import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import h3
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 消融实验配置
# 修改以下开关即可生成不同消融版本的模型
ABLATION = {
    'use_cluster': False,          # 是否使用聚类标签特征
    'use_lag_features': True,     # 是否使用滞后特征（1/7/14天）
    'use_site_avg': True,         # 是否使用站点历史平均流量特征
}

DATA_DIR = Path("data/second_preprocessed")
CLUSTER_FILE = Path("data/clustered.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# 根据消融配置生成模型文件名
ablation_suffix = ""
if not ABLATION['use_cluster']:
    ablation_suffix += "_no_cluster"
if not ABLATION['use_lag_features']:
    ablation_suffix += "_no_lag"
if not ABLATION['use_site_avg']:
    ablation_suffix += "_no_siteavg"
if ablation_suffix == "":
    ablation_suffix = "_full"
MODEL_PATH = MODEL_DIR / f"prediction_model{ablation_suffix}.pkl"

TIME_SLOTS = 48
MIN_TRAIN_SAMPLES = 100
LAG_DAYS = [1, 7, 14] if ABLATION['use_lag_features'] else []


def prepare_training_data(df, cluster_id=None):
    df = df.copy()
    df['date'] = pd.to_datetime(df['time_bin']).dt.date
    # df['day_of_month'] = pd.to_datetime(df['time_bin']).dt.day
    # df['month'] = pd.to_datetime(df['time_bin']).dt.month
    df['slot'] = df['slot'].astype(int)
    df['h3_id'] = df['h3_id'].astype(str)
    df = df.sort_values(['h3_id', 'date', 'slot']).reset_index(drop=True)

    # 滞后特征（若启用）
    if ABLATION['use_lag_features']:
        for lag_day in LAG_DAYS:
            lag_slots = lag_day * TIME_SLOTS
            df[f'lag_D_in_{lag_day}d'] = df.groupby('h3_id')['D_in'].shift(lag_slots)
            df[f'lag_D_out_{lag_day}d'] = df.groupby('h3_id')['D_out'].shift(lag_slots)

    # 站点历史平均（若启用）
    if ABLATION['use_site_avg']:
        site_avg = df.groupby('h3_id')[['D_in', 'D_out']].transform('mean')
        df['site_avg_in'] = site_avg['D_in']
        df['site_avg_out'] = site_avg['D_out']

    df['hour'] = df['slot'] // 2
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)

    # 删除含 NaN 的行（仅当有滞后特征时才需检查）
    if ABLATION['use_lag_features'] and LAG_DAYS:
        df = df.dropna(subset=[f'lag_D_in_{LAG_DAYS[-1]}d'])
    return df


def build_features(df, is_train=True):
    feature_cols = [
        'slot', 'weekday', 'is_weekend', 'holiday', 'is_preholiday',
        'sin_hour', 'cos_hour'
    ]
    if ABLATION['use_site_avg']:
        feature_cols.extend(['site_avg_in', 'site_avg_out'])
    if ABLATION['use_lag_features']:
        for lag in LAG_DAYS:
            feature_cols.extend([f'lag_D_in_{lag}d', f'lag_D_out_{lag}d'])
    if ABLATION['use_cluster'] and 'cluster' in df.columns:
        feature_cols.append('cluster')
    X = df[feature_cols].copy()
    y_in = df['D_in'].values
    y_out = df['D_out'].values
    return X, y_in, y_out


class FlowPredictor:
    def __init__(self):
        self.models_in = {}
        self.models_out = {}
        self.global_model_in = None
        self.global_model_out = None
        self.cluster_map = {}
        self.feature_columns = None
        self.default_in = 0.0
        self.default_out = 0.0
        self.slot_avg_in = {}
        self.slot_avg_out = {}
        self.site_avg_in = {}
        self.site_avg_out = {}
        self.site_s0 = {}

    def load_and_build_patterns(self):
        print(f"🚀 加载数据并构建 LightGBM 模型 (消融配置: {ablation_suffix[1:]})")
        files = list(DATA_DIR.glob("*.csv"))
        df_list = []
        for f in files:
            df = pd.read_csv(f)
            df.columns = df.columns.str.lower()
            df = df.rename(columns={'d_in': 'D_in', 'd_out': 'D_out', 's_t': 'S_t'})
            df['h3_id'] = df['h3_id'].astype(str).str.lower()
            df_list.append(df)
        df_raw = pd.concat(df_list, ignore_index=True)
        df_raw['time_bin'] = pd.to_datetime(df_raw['time_bin'])

        # 聚类特征（若启用）
        if ABLATION['use_cluster']:
            cluster_df = pd.read_csv(CLUSTER_FILE)
            cluster_df['h3_id'] = cluster_df['h3_id'].astype(str).str.lower()
            self.cluster_map = dict(zip(cluster_df['h3_id'], cluster_df['cluster']))
            df_raw['cluster'] = df_raw['h3_id'].map(self.cluster_map).fillna(-1).astype(int)
        else:
            df_raw['cluster'] = -1  # 占位，实际不会被使用

        # 全局时段平均
        slot_avg = df_raw.groupby('slot')[['D_in', 'D_out']].mean()
        self.slot_avg_in = slot_avg['D_in'].to_dict()
        self.slot_avg_out = slot_avg['D_out'].to_dict()
        self.default_in = df_raw['D_in'].mean()
        self.default_out = df_raw['D_out'].mean()

        # 站点个体统计量（仅当启用站点均值特征或预测时需要）
        print("计算站点个体平均特征...")
        site_stats = df_raw.groupby('h3_id').agg(
            avg_in=('D_in', 'mean'),
            avg_out=('D_out', 'mean')
        )
        self.site_avg_in = site_stats['avg_in'].to_dict()
        self.site_avg_out = site_stats['avg_out'].to_dict()

        s0_df = df_raw[df_raw['slot'] == 0].groupby('h3_id')['S_t'].mean()
        self.site_s0 = s0_df.fillna(10).to_dict()

        df = prepare_training_data(df_raw)
        X_sample, _, _ = build_features(df)
        self.feature_columns = X_sample.columns.tolist()
        print(f"   使用特征 ({len(self.feature_columns)} 个): {self.feature_columns}")

        if ABLATION['use_cluster']:
            clusters = df['cluster'].unique()
        else:
            clusters = [-1]  # 单一伪 cluster

        for cl in clusters:
            df_cl = df[df['cluster'] == cl] if ABLATION['use_cluster'] else df
            if len(df_cl) < MIN_TRAIN_SAMPLES:
                print(f"⚠️ Cluster {cl} 样本不足 ({len(df_cl)}), 跳过")
                continue
            X, y_in, y_out = build_features(df_cl)
            model_in = lgb.LGBMRegressor(
                n_estimators=150,
                max_depth=8,
                num_leaves=31,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                verbose=-1
            )
            model_out = lgb.LGBMRegressor(n_estimators=80, max_depth=6, verbose=-1)
            model_in.fit(X, y_in)
            model_out.fit(X, y_out)
            self.models_in[cl] = model_in
            self.models_out[cl] = model_out
            print(f"✅ Cluster {cl} 训练完成，样本数: {len(df_cl)}")

        # 全局备用模型
        print("训练全局备用模型...")
        X_all, y_in_all, y_out_all = build_features(df)
        self.global_model_in = lgb.LGBMRegressor(n_estimators=100, max_depth=7, verbose=-1)
        self.global_model_out = lgb.LGBMRegressor(n_estimators=100, max_depth=7, verbose=-1)
        self.global_model_in.fit(X_all, y_in_all)
        self.global_model_out.fit(X_all, y_out_all)

        print("✅ 模型训练完成")

    def find_cluster_of_nearby_h3(self, h3_id):
        if not ABLATION['use_cluster']:
            return -1
        neighbors = h3.grid_ring(h3_id, 1)
        clusters = [
            self.cluster_map.get(str(n).lower())
            for n in neighbors
            if str(n).lower() in self.cluster_map
        ]
        if clusters:
            from collections import Counter
            return Counter(clusters).most_common(1)[0][0]
        return -1

    def predict_for_region(self, h3_list, weekday, holiday, preholiday):
        N = len(h3_list)
        Pred_In = np.zeros((TIME_SLOTS, N))
        Pred_Out = np.zeros((TIME_SLOTS, N))
        S0 = np.zeros(N)

        for j, h in enumerate(h3_list):
            h = str(h).lower()
            cluster = self.cluster_map.get(h, -1) if ABLATION['use_cluster'] else -1
            if cluster == -1 and ABLATION['use_cluster']:
                cluster = self.find_cluster_of_nearby_h3(h)

            site_avg_in = self.site_avg_in.get(h, self.default_in)
            site_avg_out = self.site_avg_out.get(h, self.default_out)

            X_pred = pd.DataFrame({
                'slot': np.arange(TIME_SLOTS),
                'weekday': weekday,
                'is_weekend': 1 if weekday >= 5 else 0,
                'holiday': holiday,
                'is_preholiday': preholiday,
                'sin_hour': np.sin(2 * np.pi * (np.arange(TIME_SLOTS) // 2) / 24),
                'cos_hour': np.cos(2 * np.pi * (np.arange(TIME_SLOTS) // 2) / 24),
            })
            if ABLATION['use_site_avg']:
                X_pred['site_avg_in'] = site_avg_in
                X_pred['site_avg_out'] = site_avg_out
            if ABLATION['use_lag_features']:
                for lag in LAG_DAYS:
                    X_pred[f'lag_D_in_{lag}d'] = site_avg_in
                    X_pred[f'lag_D_out_{lag}d'] = site_avg_out
            if ABLATION['use_cluster']:
                X_pred['cluster'] = cluster

            # 确保列顺序与训练时一致
            X_pred = X_pred[self.feature_columns]

            model_in = self.models_in.get(cluster, self.global_model_in)
            model_out = self.models_out.get(cluster, self.global_model_out)

            if model_in is not None and model_out is not None:
                pred_in = model_in.predict(X_pred)
                pred_out = model_out.predict(X_pred)
                Pred_In[:, j] = np.maximum(pred_in, 0)
                Pred_Out[:, j] = np.maximum(pred_out, 0)
            else:
                for t in range(TIME_SLOTS):
                    Pred_In[t, j] = self.slot_avg_in.get(t, self.default_in)
                    Pred_Out[t, j] = self.slot_avg_out.get(t, self.default_out)

            S0[j] = self.site_s0.get(h, 20.0)

        return Pred_In, Pred_Out, S0


def main():
    predictor = FlowPredictor()
    predictor.load_and_build_patterns()
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(predictor, f)
    print(f"✅ 模型已保存至 {MODEL_PATH}")


if __name__ == "__main__":
    main()