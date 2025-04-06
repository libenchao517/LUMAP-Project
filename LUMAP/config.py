################################################################################
# 本文件用于存储Linear UMAP算法的关键参数
################################################################################
import numpy as np
################################################################################
class LUMAP_config:
    def __init__(self):
        self.split_state = np.random.randint(2024) # 随机种子
        self.n_components = 32      # 一维方法的目标维度
        self.n_components_2d = 1    # 二维方法的目标维度
        self.none_data = []         #
        self.train_size = 0.20      # 训练数据的比例
        self.transform_mode = "projection" # 模式：分别获取训练数据和测试数据的低维投影
        self.LUMAP_1D_init = 'random'      # 随机初始化LUMAP的映射矩阵
        self.LUMAP_2D_init = 'random'      # 随机初始化2DUMAP的映射矩阵
        self.MFD_dict = {    # 故障诊断任务中使用的数据集
            "B1": "Ottawa",
            "B2": "Polito",
            "B3": "Sebear",
            "G1" : "Connectiect",
            "G2": "Segear",
            "MB" : "Mix-Bear",
            "MG" : "Mix-Gear"
        }
        self.LUMAP_data = {  # 数据集的维度
            "B1": [32, 32],
            "B2": [32, 32],
            "B3": [32, 32],
            "G1": [32, 32],
            "G2": [32, 32],
            "MB" : [32, 32],
            "MG" : [32, 32],
        }
        self.LUMAP_select_target = { # 选择数据集使用的故障类型
            "B1": None,
            "B2": (1, 2, 3, 5, 6),
            "B3": None,
            "G1": None,
            "G2": None,
            "MB" : None,
            "MG" : None
        }
        self.basic_data = ['B1', 'G1'] # 参数实验使用的数据集
        self.cut_dimension = 32        # AIC测试中使用的观测维度
        self.visualization = True      # 是否可视化实验结果
        self.classification = True     # 是否进行分类实验
        self.clustering = True         # 是否进行聚类实验
