################################################################################
# 本代码用于自动化运行LUMAP项目实验
################################################################################
# 添加路径和消除警告
import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.append("/".join(Path(__file__).parts[0:Path(__file__).parts.index('REUMAP') + 1]))
################################################################################
# 导入模块
from Utils import Auto_Run
################################################################################
# 运行项目
AR = Auto_Run(
    Project="LUMAP",
    MRPY=None,
    content="Linear_UMAP/Basic/Visualization", lock=True)
AR.Run()

AR = Auto_Run(
    Project="LUMAP",
    MRPY=None,
    content="Linear_UMAP/Basic/Proportion",
    lock=True)
AR.Run()

AR = Auto_Run(
    Project="LUMAP",
    MRPY="Make_Pre_Results_LUMAP.py",
    content="Linear_UMAP/Basic/AIC",
    lock=True
)
AR.Run()

for i in range(10):
    AR = Auto_Run(
        Project="LUMAP",
        MRPY="Make_Results_LUMAP.py",
        content="Linear_UMAP/Experiments",
        lock=True
    )
    AR.Run()

AR = Auto_Run(
    Project="LUMAP",
    MRPY=None,
    content="Results/LUMAP-Project",
    run_file="Total_LUMAP.py",
    is_parallel=False,
    lock=True
)
AR.Run()
