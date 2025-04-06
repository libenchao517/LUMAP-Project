################################################################################
# 实验三：2DUMAP分类和聚类实验
#
# 实验目的：测试2DUMAP的效果
# 评价指标：KNN、SVM、time
################################################################################
# 添加路径和消除警告
import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.append("/".join(Path(__file__).parts[0:Path(__file__).parts.index('REUMAP') + 1]))
################################################################################
# 导入实验模块
from Assess import Analysis_Linear_DR
from Factory import factory
################################################################################
# 导入关键变量
from LUMAP import LUMAP_config
config= LUMAP_config()
################################################################################
# 运行实验
for ds in config.LUMAP_data.keys():
    model = factory(
        func_name="2DLUMAP",
        data_name=ds,
        train_size=config.train_size,
        random_state=config.split_state,
        return_time=True,
        sec_part="Experiment",
        sec_num=3)
    model.Product_Linear_UMAP_Object(
        type="2D",
        n_components=config.n_components_2d,
        n_neighbors=15,
        is_Supervised=False,
        init=config.LUMAP_2D_init,
        transform_mode=config.transform_mode)
    model.Linear_2DUMAP_Object.fit_transform(
        X_train=model.X_train, X_test=model.X_test,
        T_train=model.T_train, T_test=model.T_test,
        label=None,
        sample_height=config.LUMAP_data[ds][0],
        sample_weight=config.LUMAP_data[ds][1],
        force_all_finite=True)
    Analysis_Linear_DR(model.Linear_2DUMAP_Object).Analysis(
        classification=config.classification,
        cluster=config.clustering,
        visualization=config.visualization)
