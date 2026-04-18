# 数据驱动的共享单车潮汐调度闭环建模与动态优化

[![Release](https://img.shields.io/badge/release-v1.0.0-blue)](https://github.com/MrKssksksksk/Shared-Bike-Tidal-Scheduling-Modeling/releases/tag/1.0.0)
[![Python](https://img.shields.io/badge/python-3.8%2B-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-orange)](LICENSE)

## 📖 项目简介

本项目是数学建模论文 **《数据驱动的共享单车潮汐调度闭环建模与动态优化》** 的完整代码实现。基于深圳市2021年1月至8月约2.4亿条真实骑行数据，构建了从数据清洗、空间聚类、流量预测到调度优化与路径规划的端到端解决方案。

> 📌 **论文对应版本**：本仓库持续更新，论文引用的代码版本为 **[v1.0.0](https://github.com/MrKssksksksk/Shared-Bike-Tidal-Scheduling-Modeling/releases/tag/1.0.0)**。如需复现论文结果，请下载该 Release 的源代码。

**核心亮点**：
- 完整的工程化闭环：13个模块无缝衔接，可一键复现
- 高效的数据处理：Numba JIT加速库存递推，亿级数据单机可处理
- 消融实验支持：预测模型支持特征开关，便于验证各组件贡献
- 交互式可视化：Leaflet + Folium 动态地图展示潮汐分布与调度路线

## 🚀 快速开始

### 环境要求
- Python 3.8 或更高版本
- 依赖库安装：`pip install -r requirements.txt`

### 数据准备
由于原始数据受深圳市开放数据平台使用协议限制，本仓库**不包含原始数据文件及行政区划边界文件**。请按以下步骤获取数据：

1. **骑行订单数据**  
   访问 [深圳市政府数据开放平台 - 共享单车企业每日订单表](https://opendata.sz.gov.cn/data/dataSet/toDataDetails/29200_00403627)，申请 API 密钥（AppKey）。  
   该数据集包含 2021 年 1 月至 8 月约 2.44 亿条记录，原始坐标系为 **GCJ-02**（火星坐标系），包含 8 个字段：`USER_ID`、`COM_ID`、`START_TIME`、`START_LNG`、`START_LAT`、`END_TIME`、`END_LNG`、`END_LAT`。

2. **深圳行政区划边界 Shapefile**  
   从 [POI86](https://www.poi86.com/poi/amap/city/440300.html) 下载深圳市边界 Shapefile 文件（含 `.shp`、`.shx`、`.dbf` 等），放置于 `data/geo/` 目录下，用于数据清洗时的空间范围过滤。

3. **配置 API 密钥**  
   修改 `scr/01_fetch_data.py` 中的 `APP_KEY` 变量为您的密钥。

4. **运行数据获取脚本**（详见下方运行流程）

### 运行流程
按顺序执行以下脚本，即可复现论文全部结果：

| 步骤 | 脚本 | 功能 |
|------|------|------|
| 1 | `scr/01_fetch_data.py` | 多线程抓取原始订单数据 |
| 2 | `scr/02_preprocess.py` | 数据清洗（GCJ-02→WGS-84、空间过滤） |
| 3 | `scr/03_h3_encoding.py` | H3六边形网格编码 |
| 4 | `scr/04_flow_generation.py` | 30分钟流量聚合与时间特征构造 |
| 5 | `scr/05_preprocess_for_clustering.py` | 库存递推、容量估计、潮汐指数计算 |
| 6 | `scr/06_visualization.py` | 交互式潮汐动态地图（可选） |
| 7 | `scr/07_functional_clustering.py` | K-Means站点功能区聚类 |
| 8 | `scr/08_prediction_model_with_ablation.py` | 训练LightGBM预测模型 |
| 9 | `scr/09_prediction.py` | 执行预测，输出 `prediction.csv` |
| 10 | `scr/10_solving_model.py` | 调度优化求解，生成调度任务 |
| 11 | `scr/12_transport_solution.py` | 多车辆路径规划（CVRP-TW） |
| 12 | `scr/11_evaluation.py` | 调度效果评价与指标对比 |
| 13 | `scr/13_visualize_routes.py` | 调度路径交互式地图（可选） |

> 详细运行说明请参见各脚本头部注释。

## 📁 目录结构
      ├── scr/ # 源代码
      ├── data/ # 数据目录（需自行准备部分数据）
      │ ├── raw/ # 原始CSV文件
      │ ├── processed/ # 清洗后数据
      │ ├── h3/ # H3编码后数据
      │ ├── flow_final/ # 流量聚合数据
      │ ├── second_preprocessed/ # 预处理后数据（含库存与潮汐指数）
      │ └── geo/ # 深圳行政区划shapefile（需自行下载）
      ├── models/ # 训练好的模型文件
      ├── outputs/ # 输出图表与交互式地图
      ├── requirements.txt # Python依赖列表
      ├── README.md # 项目说明文档
      └── LICENSE # MIT许可证

## 📊 主要结果

以2021年8月21日为例，调度前后指标对比：

|       指标       | 自然状态 | 执行调度 | 减少率|
|------------------|---------|---------|--------|
|      总缺车量     |  3,979  |  3,287  | 17.39% |
|    高峰总缺车量   |   692   |    0     |  100%  |
| 全天加权平均失败率 | 127.18% |  126.57% | 0.48%  |

## 📄 论文引用

如果您使用了本项目的代码或数据，请引用：

> 黄梓涵, 杨臻璟, 麦展源. 数据驱动的共享单车潮汐调度闭环建模与动态优化[EB/OL]. GitHub, (2026). https://github.com/MrKssksksksk/Shared-Bike-Tidal-Scheduling-Modeling/releases/tag/1.0.0

## ⚠️ 数据与版权说明

- **原始骑行订单数据**受深圳市开放数据平台使用协议限制，本仓库**不包含**该数据。使用者需自行通过 API 获取，并遵守平台相关规定。
- **深圳行政区划边界 Shapefile** 可从 [POI86](https://www.poi86.com/poi/amap/city/440300.html) 免费下载，本仓库**不包含**该文件。请在下载后放置于 `data/geo/` 目录。
- 项目中使用的 **H3 地理索引系统** 为 Uber 开源技术，坐标系转换算法参考公开文献实现。

## 📝 许可证

本项目采用 [MIT License](LICENSE)。
