################################################################################
# 本代码用于自动化运行LUMAP-Deep-Net-FD项目实验
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
    MRPY="Make_Results_DNetFD.py",
    content="Deep_Net_FD",
    lock=True
)
AR.Run()

for i in range(10):
    AR = Auto_Run(
        Project="LUMAP",
        MRPY=None,
        content="Deep_Net_FD",
        run_file="Net-1-CNN_2560_768.py",
        is_parallel=False,
        lock=True
    )
    AR.Run()

    AR = Auto_Run(
        Project="LUMAP",
        MRPY=None,
        content="Deep_Net_FD",
        run_file="Net-4-LeNet_5.py",
        is_parallel=False,
        lock=True
    )
    AR.Run()

    AR = Auto_Run(
        Project="LUMAP",
        MRPY=None,
        content="Deep_Net_FD",
        run_file="Net-5-LiNet.py",
        is_parallel=False,
        lock=True
    )
    AR.Run()

    AR = Auto_Run(
        Project="LUMAP",
        MRPY=None,
        content="Deep_Net_FD",
        run_file="Net-7-TICNN.py",
        is_parallel=False,
        lock=True
    )
    AR.Run()

    AR = Auto_Run(
        Project="LUMAP",
        MRPY="Make_Results_DNetFD.py",
        content="Deep_Net_FD",
        run_file="Net-8-WDCNN.py",
        is_parallel=False,
        lock=True
    )
    AR.Run()
