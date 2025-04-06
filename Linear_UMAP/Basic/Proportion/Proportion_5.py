################################################################################
# 基础实验2-5：2DUMAP与2DSUMAP算法训练集比例划分
#
# 实验目的：测试以下数据集在Two Dimensional Linear UMAP上的效果
# 评价指标：KNN、NMI
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
from Factory import factory
from Assess import K_Nearest_Neighbors
from Assess import Normalized_Mutual_Info_Score
################################################################################
# 导入关键变量
from LUMAP import LUMAP_config
from Draw import Draw_Line_Chart
config = LUMAP_config()
knn = K_Nearest_Neighbors(neighbors=5)
nmi = Normalized_Mutual_Info_Score()
################################################################################
# 定义实验变量
train_size = [i*0.05 for i in range(1, 20)]
lts = len(train_size)
################################################################################
# 运行试验
for ds in config.basic_data:
    KNN_2DUMAP = np.zeros(lts)
    KNN_2DSUMAP = np.zeros(lts)
    for i, ts in enumerate(train_size):
        model_2DUMAP = factory(
            func_name="2DUMAP", data_name=ds, train_size=ts,
            random_state=config.split_state, return_time=True,
            sec_part="Proportion", sec_num=5)
        model_2DUMAP.Product_Linear_UMAP_Object(
            type="2D", n_components=config.n_components_2d,
            n_neighbors=15, is_Supervised=False,
            init=config.LUMAP_2D_init,
            transform_mode=config.transform_mode)
        model_2DUMAP.Linear_2DUMAP_Object.fit_transform(
            X_train=model_2DUMAP.X_train, X_test=model_2DUMAP.X_test,
            T_train=model_2DUMAP.T_train, T_test=model_2DUMAP.T_test,
            label=None,
            sample_height=config.LUMAP_data[ds][0],
            sample_weight=config.LUMAP_data[ds][1],
            force_all_finite=True)
        knn.KNN_predict_odds_splited(
            model_2DUMAP.Linear_2DUMAP_Object.Y_train,
            model_2DUMAP.Linear_2DUMAP_Object.Y_test,
            model_2DUMAP.Linear_2DUMAP_Object.T_train,
            model_2DUMAP.Linear_2DUMAP_Object.T_test, name=None)
        KNN_2DUMAP[i] = knn.accuracy
        model_2DSUMAP = factory(
            func_name="2DSUMAP", data_name=ds, train_size=ts,
            random_state=config.split_state, return_time=True,
            sec_part="Proportion", sec_num=5)
        model_2DSUMAP.Product_Linear_UMAP_Object(
            type="2D", n_components=config.n_components_2d,
            n_neighbors=15, is_Supervised=True,
            init=config.LUMAP_2D_init,
            transform_mode=config.transform_mode)
        model_2DSUMAP.Linear_2DUMAP_Object.fit_transform(
            X_train=model_2DSUMAP.X_train, X_test=model_2DSUMAP.X_test,
            T_train=model_2DSUMAP.T_train, T_test=model_2DSUMAP.T_test,
            label=model_2DSUMAP.label,
            sample_height=config.LUMAP_data[ds][0],
            sample_weight=config.LUMAP_data[ds][1],
            force_all_finite=True)
        knn.KNN_predict_odds_splited(
            model_2DSUMAP.Linear_2DUMAP_Object.Y_train,
            model_2DSUMAP.Linear_2DUMAP_Object.Y_test,
            model_2DSUMAP.Linear_2DUMAP_Object.T_train,
            model_2DSUMAP.Linear_2DUMAP_Object.T_test, name=None)
        KNN_2DSUMAP[i] = knn.accuracy
    Draw_Line_Chart(
        filename=model_2DUMAP.sec_part + "-" + str(model_2DUMAP.sec_num) + '-' + model_2DUMAP.data_name,
        column=train_size,
        left=[KNN_2DUMAP, KNN_2DSUMAP],
        ylim_left=(0, 1.02),
        left_marker=(["s", "^"]),
        left_color=["#9AB5F3", "#7323AB"],
        left_markeredgecolor=["#658DED", "#4F1773"],
        left_markerfacecolor=["#9AB5F3", "#7323AB"],
        ylabel_left="classification accuracy",
        xlabel="% size of train set",
        fontsize=18,
        titlefontsize=20,
        left_label=["2DUMAP", "2DSUMAP"]).Draw_simple_line()
