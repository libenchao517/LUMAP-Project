################################################################################
# 对比实验七：UMAP分类和聚类实验
#
# 实验目的：测试UMAP的效果
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
config = LUMAP_config()
################################################################################
# 运行实验
for ds in config.LUMAP_data.keys():
    cmodel = factory(
        func_name="UMAP",
        data_name=ds,
        train_size=config.train_size,
        random_state=config.split_state,
        return_time=True,
        sec_part="Comparatation",
        sec_num=7)
    cmodel.Product_Linear_Contrast_Object(n_components=config.n_components)
    cmodel.Linear_Contrast_Object.Linear_MAP(cmodel.X_test, cmodel.T_test)
    Analysis_Linear_DR(cmodel.Linear_Contrast_Object).Analysis(
        classification=config.classification,
        cluster=config.clustering,
        visualization=config.visualization)
