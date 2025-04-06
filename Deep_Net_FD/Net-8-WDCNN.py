################################################################################
# 对比实验八：WDCNN
################################################################################
# 添加路径和消除警告
import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.append("/".join(Path(__file__).parts[0:Path(__file__).parts.index('REUMAP') + 1]))
################################################################################
# 导入实验模块
from Assess import Analysis_Deep_Net_FD
from Factory import factory
################################################################################
# 导入关键变量
from LUMAP import Deep_Net_config
config = Deep_Net_config()
################################################################################
# 运行实验
for ds in config.Net_data.keys():
    net_model = factory(
        func_name="WDCNN",
        data_name=ds,
        train_size=config.train_size.get(ds),
        random_state=config.split_state,
        return_time=True,
        sec_part="Deep-Net",
        sec_num=8)
    net_model.Product_Deep_Net_FD_Object()
    net_model.Linear_Contrast_Net_Object.Run_Deep_Net()
    Analysis_Deep_Net_FD(net_model.Linear_Contrast_Net_Object).Analysis(
        classification=config.classification,
        cluster=config.clustering,
        visualization=config.visualization)
