################################################################################
# 本文件用于存储Deep Net算法的关键参数
################################################################################
import numpy as np
################################################################################
class Deep_Net_config:
    def __init__(self):
        self.split_state = np.random.randint(2024) # 随机种子
        self.none_data = []                        #
        self.MFD_dict = {  # 故障诊断任务中使用的数据集
            "B1": "Ottawa",
            "B2": "Polito",
            "B3": "Sebear",
            "G1": "Connectiect",
            "G2": "Segear",
            "MB": "Mix-Bear",
            "MG": "Mix-Gear"
        }
        self.data_2D = ["B1", "B2", "B3", "G1", "G2", "MB", "MG"]
        self.Net_data = {   # 故障诊断任务中使用的数据集
            "B1": [32, 32],
            "B2": [32, 32],
            "B3": [32, 32],
            "G1": [32, 32],
            "G2": [32, 32],
            "MB": [32, 32],
            "MG": [32, 32]
        }
        self.train_size = {  # 训练数据的比例
            "B1": 0.20,
            "B2": 0.20,
            "B3": 0.20,
            "G1": 0.20,
            "G2": 0.20,
            "MB": 0.20,
            "MG": 0.20,
        }
        self.LUMAP_select_target = {  # 选择数据集使用的故障类型
            "B2": (1, 2, 3, 5, 6),
        }
        self.epoch=100   # 网络的迭代次数
        self.visualization = True      # 是否可视化实验结果
        self.classification = True     # 是否进行分类实验
        self.clustering = True         # 是否进行聚类实验
        self.device = "cpu"  # 运行网络的设备
