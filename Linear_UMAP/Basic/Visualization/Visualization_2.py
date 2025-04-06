################################################################################
# 可视化实验2：可视化数据集分布
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
from umap import UMAP
from Draw import Draw_Embedding
from LUMAP import LUMAP_config
config= LUMAP_config()
from Factory import factory
################################################################################
# 可视化
for ds in config.LUMAP_data.keys():
    model = factory(
        func_name="2DUMAP",
        data_name=ds,
        train_size=config.train_size,
        random_state=config.split_state,
        return_time=True,
        sec_part="Visualization",
        sec_num=2)
    model.Product_Linear_UMAP_Object(
        type="2D",
        n_components=config.n_components_2d,
        n_neighbors=15,
        is_Supervised=False,
        init=config.LUMAP_2D_init,
        transform_mode=config.transform_mode)
    DE = Draw_Embedding(
        dota_size=6,
        fontsize=18,
        titlefontsize=20,
        lgd=model.Linear_2DUMAP_Object.legd,
        cmap="viridis"
    )
    embedding = UMAP().fit_transform(model.Linear_2DUMAP_Object.data)
    DE.Draw_embedding(embedding, model.Linear_2DUMAP_Object.label, name=model.Linear_2DUMAP_Object.para)
