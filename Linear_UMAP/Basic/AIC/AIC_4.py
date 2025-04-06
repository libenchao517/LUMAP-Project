################################################################################
# 基本实验1-4：2DSUMAP最优降维维度测算实验
#
# 实验目的：测试以下数据集在Supervised Two Dimensional UMAP上的效果
# 评价指标：AIC
################################################################################
# 添加路径和消除警告
import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.append("/".join(Path(__file__).parts[0:Path(__file__).parts.index('REUMAP') + 1]))
################################################################################
# 导入实验模块
import numpy as np
from Assess import K_Nearest_Neighbors
from Assess import Akaike_Information_Criterion
from Factory import factory
from LUMAP import LUMAP_config
################################################################################
# 导入关键变量
config = LUMAP_config()
AI = Akaike_Information_Criterion(filename='AIC-4-2DSUMAP', func_name='2DSLUMAP')
knn = K_Nearest_Neighbors()
################################################################################
# 定义实验变量
KNN = np.zeros((len(config.basic_data), config.cut_dimension))
################################################################################
# 运行试验
for d, ds in enumerate(config.basic_data):
    model = factory(
        func_name="2DSUMAP", data_name=ds, train_size=0.1,
        random_state=config.split_state, return_time=True,
        sec_part="AIC", sec_num=4)
    AI.max_dimension.append(config.LUMAP_data[ds][1])
    AI.cut_dimension.append(config.cut_dimension)
    AI.data_names.append(ds)
    for i in range(1,config.cut_dimension+1):
        model.Product_Linear_UMAP_Object(
            type="2D", n_components=i, n_neighbors=15,
            is_Supervised=True, init=config.LUMAP_2D_init,
            transform_mode=config.transform_mode)
        model.Linear_2DUMAP_Object.fit_transform(
            X_train=model.X_train, X_test=model.X_test,
            T_train=model.T_train, T_test=model.T_test, label=model.label,
            sample_height=config.LUMAP_data[ds][0],
            sample_weight=config.LUMAP_data[ds][1],
            force_all_finite=True)
        knn.KNN_predict_odds_splited(
            model.Linear_2DUMAP_Object.Y_train,
            model.Linear_2DUMAP_Object.Y_test,
            model.Linear_2DUMAP_Object.T_train,
            model.Linear_2DUMAP_Object.T_test, name=None)
        KNN[d][i-1] = knn.accuracy
AI.odds=KNN
AI.AIC_Run()