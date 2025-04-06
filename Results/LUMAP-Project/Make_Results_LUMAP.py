################################################################################
# 本代码用于整理LUMAP项目的实验结果
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
import numpy as np
import pandas as pd
from Utils import Make_Table
from LUMAP import LUMAP_config
config = LUMAP_config()
################################################################################
# 定义必要变量
Root = "/".join(Path(__file__).parts[:-1])
Analysis = '/Analysis/'
Temp_1_File= '/Temp-1-Files/'
Temp_2_File='/Temp-2-Files/'
Result_File= '/Result-Files/'
Path("./Temp-1-Files").mkdir(exist_ok=True)
Path("./Temp-2-Files").mkdir(exist_ok=True)
Path("./Result-Files").mkdir(exist_ok=True)
methods = ['PCA', 'TDPCA','ICA','LPP', 'PNNLPP','NPE',
           'UMAP', 'PUMAP','2DUMAP','2DSUMAP']
MT = Make_Table(methods)
################################################################################
# 整理到临时汇总文件
for d in config.LUMAP_data:
    temp = pd.DataFrame()
    path = Root + Analysis + d
    xlsx_list = list(map(str, list(Path(path).rglob("*.xlsx"))))
    xlsx_list = [xlsx for xlsx in xlsx_list if "Experiment" in xlsx or "Comparatation" in xlsx]
    for xlsx in xlsx_list:
        df = pd.read_excel(xlsx, header=0, index_col=0)
        temp = pd.concat([temp, df])
    xlsx_path = d + '.xlsx'
    temp.to_excel(Root + Temp_1_File + xlsx_path)
for d in config.LUMAP_data:
    temp = pd.DataFrame()
    path = Root + Analysis + d
    xlsx_list = list(map(str, list(Path(path).rglob("*.xlsx"))))
    xlsx_list = [xlsx for xlsx in xlsx_list if "Experiment" in xlsx or "Compara-Best" in xlsx]
    for xlsx in xlsx_list:
        df = pd.read_excel(xlsx, header=0, index_col=0)
        temp = pd.concat([temp, df])
    xlsx_path = d + '.xlsx'
    temp.to_excel(Root + Temp_2_File + xlsx_path)
################################################################################
# 汇总相同纬度分析的所有结果
Total_Results = pd.DataFrame()
path = Root + Temp_1_File
xlsx_list = list(map(str, list(Path(path).rglob("*.xlsx"))))
for xlsx in xlsx_list:
    df = pd.read_excel(xlsx, header=0, index_col=0)
    Total_Results = pd.concat([Total_Results, df])
LUMAP_method = list(set(list(Total_Results.Method)))
if np.nan in LUMAP_method:
    LUMAP_method.remove(np.nan)
LUMAP_method.sort()
LUMAP_index = ['KNN', 'SVM', 'FMS', 'NMI', 'SHS', 'time']
except_index= ['KNN', 'SVM']
################################################################################
# 整理相同维度分析的各个指标
for idx in LUMAP_index:
    Results = Total_Results[['Method', 'Datasets', idx]].copy()
    Result_train = Results[Total_Results.index == 'Train']
    Result_test = Results[Total_Results.index == 'Test']
    Result_total = Results[Total_Results.index == 'Total']
    Result_train.set_index(['Method', 'Datasets'], inplace=True)
    Result_test.set_index(['Method', 'Datasets'], inplace=True)
    Result_total.set_index(['Method', 'Datasets'], inplace=True)
    Result_1 = pd.DataFrame(index=LUMAP_method, columns=config.LUMAP_data)
    Result_2 = pd.DataFrame(index=LUMAP_method, columns=config.LUMAP_data)
    Result_3 = pd.DataFrame(index=LUMAP_method, columns=config.LUMAP_data)
    if idx in except_index:
        Result_apply = Results[Total_Results.index == 'Apply']
        Result_apply.set_index(['Method', 'Datasets'], inplace=True)
        Result_4 = pd.DataFrame(index=LUMAP_method, columns=config.LUMAP_data)
    for m in LUMAP_method:
        for d in config.LUMAP_data:
            try:
                Result_1.loc[m, d] = Result_train.loc[m, d][0]
            except:
                Result_1.loc[m, d] = None
            try:
                Result_2.loc[m, d] = Result_test.loc[m, d][0]
            except:
                Result_2.loc[m, d] = None
            try:
                Result_3.loc[m, d] = Result_total.loc[m, d][0]
            except:
                Result_3.loc[m, d] = None
            if idx in except_index:
                try:
                    Result_4.loc[m, d] = Result_apply.loc[m, d][0]
                except:
                    Result_4.loc[m, d] = None
    with pd.ExcelWriter(Root + Result_File + idx + '.xlsx') as writer:
        Result_1.to_excel(writer, sheet_name='train')
        Result_2.to_excel(writer, sheet_name='test')
        Result_3.to_excel(writer, sheet_name='total')
        if idx in except_index:
            Result_4.to_excel(writer, sheet_name='apply')
################################################################################
# 汇总相同纬度分析的所有结果
Total_Results = pd.DataFrame()
path = Root + Temp_2_File
xlsx_list = list(map(str, list(Path(path).rglob("*.xlsx"))))
for xlsx in xlsx_list:
    df = pd.read_excel(xlsx, header=0, index_col=0)
    Total_Results = pd.concat([Total_Results, df])
LUMAP_method = list(set(list(Total_Results.Method)))
if np.nan in LUMAP_method:
    LUMAP_method.remove(np.nan)
LUMAP_method.sort()
LUMAP_index = ['KNN', 'SVM', 'FMS', 'NMI', 'SHS', 'time']
except_index= ['KNN', 'SVM']
################################################################################
# 整理相同维度分析的各个指标
for idx in LUMAP_index:
    Results = Total_Results[['Method', 'Datasets', idx]].copy()
    Result_train = Results[Total_Results.index == 'Train']
    Result_test = Results[Total_Results.index == 'Test']
    Result_total = Results[Total_Results.index == 'Total']
    Result_train.set_index(['Method', 'Datasets'], inplace=True)
    Result_test.set_index(['Method', 'Datasets'], inplace=True)
    Result_total.set_index(['Method', 'Datasets'], inplace=True)
    Result_1 = pd.DataFrame(index=LUMAP_method, columns=config.LUMAP_data)
    Result_2 = pd.DataFrame(index=LUMAP_method, columns=config.LUMAP_data)
    Result_3 = pd.DataFrame(index=LUMAP_method, columns=config.LUMAP_data)
    if idx in except_index:
        Result_apply = Results[Total_Results.index == 'Apply']
        Result_apply.set_index(['Method', 'Datasets'], inplace=True)
        Result_4 = pd.DataFrame(index=LUMAP_method, columns=config.LUMAP_data)
    for m in LUMAP_method:
        for d in config.LUMAP_data:
            try:
                Result_1.loc[m, d] = Result_train.loc[m, d][0]
            except:
                Result_1.loc[m, d] = None
            try:
                Result_2.loc[m, d] = Result_test.loc[m, d][0]
            except:
                Result_2.loc[m, d] = None
            try:
                Result_3.loc[m, d] = Result_total.loc[m, d][0]
            except:
                Result_3.loc[m, d] = None
            if idx in except_index:
                try:
                    Result_4.loc[m, d] = Result_apply.loc[m, d][0]
                except:
                    Result_4.loc[m, d] = None
    with pd.ExcelWriter(Root + Result_File + idx + 'Best' + '.xlsx') as writer:
        Result_1.to_excel(writer, sheet_name='train')
        Result_2.to_excel(writer, sheet_name='test')
        Result_3.to_excel(writer, sheet_name='total')
        if idx in except_index:
            Result_4.to_excel(writer, sheet_name='apply')
################################################################################
# 整理文件和文件夹
fromdir = os.getcwd()
todir = "LUMAP-"+str(dt.date.today()) + "-" + dt.datetime.now().time().strftime("%H-%M")
Path("./" + todir).mkdir(exist_ok=True)
shutil.move(fromdir + "/Analysis", todir)
shutil.move(fromdir + "/Result-Files", todir)
shutil.move(fromdir + "/Temp-1-Files", todir)
shutil.move(fromdir + "/Temp-2-Files", todir)
shutil.move(fromdir + "/Figure", todir)
if os.path.exists(fromdir + "/AIC-Best-Dimensionality.txt"):
    shutil.move(fromdir + "/AIC-Best-Dimensionality.txt", todir)
################################################################################
# 整理结果文件格式
xlsx_list = list(map(str, list(Path(todir).rglob("*.xlsx"))))
for xlsx in xlsx_list:
    MT.Make(xlsx)
