################################################################################
# 本代码用于整理实验结果
################################################################################
# 添加路径和消除警告
import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.append("/".join(Path(__file__).parts[0:Path(__file__).parts.index('REUMAP') + 1]))
################################################################################
# 导入必要模块
import os
import shutil
import datetime as dt
################################################################################
# 整理文件和文件夹
fromdir = os.getcwd()
todir = "LUMAP-Pre-Test-"+str(dt.date.today()) + "-" + dt.datetime.now().time().strftime("%H-%M")
Path("./" + todir).mkdir(exist_ok=True)
shutil.move(fromdir + "/AIC-Best-Dimensionality.txt", todir)
shutil.move(fromdir + "/Figure", todir)
