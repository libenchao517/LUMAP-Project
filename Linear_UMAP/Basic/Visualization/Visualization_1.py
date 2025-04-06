################################################################################
# 可视化实验1：可视化数据集
#
# 实验目的：可视化同类样本间相似性
################################################################################
# 添加路径和消除警告
import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.append("/".join(Path(__file__).parts[0:Path(__file__).parts.index('REUMAP') + 1]))
################################################################################
# 导入实验模块
import os
from DATA import Load_Data
from DATA import abbre_labels
from Draw import Visual_Pixes
from LUMAP import LUMAP_config
config = LUMAP_config()
os.makedirs("Figure", exist_ok=True)
################################################################################
# 可视化数据集
for dn in config.LUMAP_data.keys():
    if dn in config.MFD_dict.keys():
        dn = config.MFD_dict.get(dn)
    data, target = Load_Data(dn).Loading()
    legd = abbre_labels.get(dn)
    Visual_Pixes(lgd=legd, filename="Visualization-Datasets-"+dn).Drawing(data, target)
