################################################################################
# 本文件用于对Linear UMAP算法的评价进行标准化
################################################################################
# 导入模块
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from .Assessment import K_Nearest_Neighbors
from .Assessment import Support_Vector_Machine
from .Assessment import Normalized_Mutual_Info_Score
from .Assessment import Fowlkes_Mallows_Score
from .Assessment import Silhouette_Score
from Draw import Color_Mapping
from Draw import Confusion_Matrix
from Draw import Draw_Embedding
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
################################################################################
class Analysis_Linear_DR:
    """
    线性降维算法标准化分析过程
    """
    def __init__(self, object):
        """
        初始化函数
        :param object: 算法实例化对象
        """
        self.object = object
        # 初始化评价指标
        self.knn = K_Nearest_Neighbors(neighbors=5)
        self.svm = Support_Vector_Machine(gamma=0.01)
        self.shs = Silhouette_Score()
        self.fms = Fowlkes_Mallows_Score()
        self.nmi = Normalized_Mutual_Info_Score()
        self.kms = KMeans(n_clusters=len(set(list(self.object.T_train))))
        # 初始化存储结果的表格
        self.result = pd.DataFrame(
            columns=['Method', 'Datasets', 'KNN', 'SVM', 'FMS', 'NMI', 'SHS', 'time'],
            index=['Total', 'Train', 'Test', 'Apply']
        )
        # 文件存储路径
        self.xlsx_path = "-".join(self.object.para[0:4]) + '.xlsx'
        self.true_label_path = "-".join(self.object.para[0:4]) + '-troe.csv'
        self.pred_label_path = "-".join(self.object.para[0:4]) + '-pred.csv'
        self.pred_data_path = "-".join(self.object.para[0:4]) + '-datas.csv'
        self.cmap_path = "-".join(self.object.para[0:4]) + '-Color-Map'
        self.cmat_path = "-".join(self.object.para[0:4]) + '-Confusion-Matrix'
        self.embed_path = "-".join(self.object.para[0:4]) + '-Embedding-3D'
        self.cmap = Color_Mapping(filename=self.cmap_path, titlefontsize=20, fontsize=16, lgd=object.legd)
        self.cmat = Confusion_Matrix(
            filename=self.cmat_path,
            fontsize=18,
            titlefontsize=20,
            lgd=object.legd)
        self.xn = 80
        Path("./Analysis").mkdir(exist_ok=True)

    def Analysis(self, classification=True, cluster=True, visualization=True):
        """
        分析模型的主函数
        :param classification:
        :param cluster:
        :param visualization:
        :return:
        """
        Path("./Analysis/" + self.object.data_name).mkdir(exist_ok=True)
        print("*" * self.xn)
        print(self.object.para[2] + "算法在" + self.object.para[3] + "数据集上的降维效果定量评价报告")
        print("*" * self.xn)
        Y = np.concatenate((self.object.Y_train, self.object.Y_test))
        T = np.concatenate((self.object.T_train, self.object.T_test))
        测试解决局外样本点嵌入问题的能力
        self.knn.KNN_predict_odds(Y, T, name=None)
        self.svm.SVM_predict_odds(Y, T, name=None)
        self.kms.fit(Y)
        self.fms.fowlkes_mallows_score_(T, self.kms.labels_, name=None)
        self.nmi.normalized_mutual_info_score_(T, self.kms.labels_, name=None)
        self.result['Method'].Total = self.object.func_name
        self.result['Datasets'].Total = self.object.data_name
        self.result['KNN'].Total = self.knn.accuracy
        self.result['SVM'].Total = self.svm.accuracy
        self.result['FMS'].Total = self.fms.score
        self.result['NMI'].Total = self.nmi.score
        self.result['time'].Total = self.object.time
        self.shs.silhouette_score_(Y, T, name=None)
        self.result['SHS'].Total = self.shs.score

        self.kms.fit(self.object.Y_train)
        self.knn.KNN_predict_odds(self.object.Y_train,self.object.T_train, name=None)
        self.svm.SVM_predict_odds(self.object.Y_train,self.object.T_train, name=None)
        self.fms.fowlkes_mallows_score_(self.object.T_train, self.kms.labels_, name=None)
        self.nmi.normalized_mutual_info_score_(self.object.T_train, self.kms.labels_, name=None)
        self.result['Method'].Train = self.object.func_name
        self.result['Datasets'].Train = self.object.data_name
        self.result['KNN'].Train = self.knn.accuracy
        self.result['SVM'].Train = self.svm.accuracy
        self.result['FMS'].Train = self.fms.score
        self.result['NMI'].Train = self.nmi.score
        self.result['time'].Train = self.object.time
        self.shs.silhouette_score_(self.object.Y_train, self.object.T_train, name=None)
        self.result['SHS'].Train = self.shs.score

        self.kms.fit(self.object.Y_test)
        self.knn.KNN_predict_odds(self.object.Y_test, self.object.T_test, name=None)
        self.svm.SVM_predict_odds(self.object.Y_test, self.object.T_test, name=None)
        self.fms.fowlkes_mallows_score_(self.object.T_test, self.kms.labels_, name=None)
        self.nmi.normalized_mutual_info_score_(self.object.T_test, self.kms.labels_, name=None)
        self.result['Method'].Test = self.object.func_name
        self.result['Datasets'].Test = self.object.data_name
        self.result['KNN'].Test = self.knn.accuracy
        self.result['SVM'].Test = self.svm.accuracy
        self.result['FMS'].Test = self.fms.score
        self.result['NMI'].Test = self.nmi.score
        self.result['time'].Test = self.object.time
        self.shs.silhouette_score_(self.object.Y_test, self.object.T_test, name=None)
        self.result['SHS'].Test = self.shs.score
        # 测试模型分类能力
        self.knn.KNN_predict_odds_splited(self.object.Y_train, self.object.Y_test, self.object.T_train, self.object.T_test, name=None)
        self.svm.SVM_predict_odds_splited(self.object.Y_train, self.object.Y_test, self.object.T_train, self.object.T_test, name=None)
        save_t_pred = self.knn.t_pred
        self.result['Method'].Apply = self.object.func_name
        self.result['Datasets'].Apply = self.object.data_name
        self.result['KNN'].Apply = self.knn.accuracy
        self.result['SVM'].Apply = self.svm.accuracy
        self.result['FMS'].Apply = None
        self.result['NMI'].Apply = None
        self.result['time'].Apply = self.object.time
        self.result['SHS'].Apply = None
        # 可视化结果
        if visualization:
            self.cmat.Drawing(self.object.T_test, self.knn.t_pred)
            self.cmap.Mapping(self.object.T_test, self.knn.t_pred)
        print(self.result)
        print("*" * self.xn)
        # 存储结果
        self.result.to_excel("./Analysis/"+ self.object.data_name + "/" + self.xlsx_path)
        pd.DataFrame(self.object.T_test).to_csv("./Analysis/"+ self.true_label_path, header=False, index=False)
        pd.DataFrame(save_t_pred).to_csv("./Analysis/"+ self.pred_label_path, header=False, index=False)
        pd.DataFrame(self.object.X_test).to_csv("./Analysis/"+ self.pred_data_path, header=False, index=False)
################################################################################
class Analysis_Deep_Net_FD:
    def __init__(self, object):
        """
        深度网络的分类分析
        :param object: 网络的实例化对象
        """
        self.object = object
        # 初始化存储结果的表格
        self.result = pd.DataFrame(
            columns=['Method', 'Datasets', 'ACC', 'PRE', 'REC', 'F1', 'time'],
            index=['Total']
        )
        # 存储结果的路径
        self.xlsx_path = "-".join(self.object.para[0:4]) + '.xlsx'
        self.xn = 80
        Path("./Analysis").mkdir(exist_ok=True)

    def Analysis(self, classification=True, cluster=True, visualization=True):
        """
        分析模型的主函数
        :param classification:
        :param cluster:
        :param visualization:
        :return:
        """
        Path("./Analysis/" + self.object.data_name).mkdir(exist_ok=True)
        print("*" * self.xn)
        print(self.object.para[2] + "算法在" + self.object.para[3] + "数据集上的降维效果定量评价报告")
        print("*" * self.xn)
        self.result['Method'].Total = self.object.func_name
        self.result['Datasets'].Total = self.object.data_name
        if classification:
            self.result['ACC'].Total = accuracy_score(self.object.t_test, self.object.y_pred)
            self.result['PRE'].Total = precision_score(self.object.t_test, self.object.y_pred, average="macro")
            self.result['REC'].Total = recall_score(self.object.t_test, self.object.y_pred, average="macro")
            self.result['F1'].Total = f1_score(self.object.t_test, self.object.y_pred, average="macro")
        self.result['time'].Total = self.object.time
        if cluster:
            pass
        if visualization:
            pass
        print(self.result)
        print("*" * self.xn)
        self.result.to_excel("./Analysis/"+ self.object.data_name + "/" + self.xlsx_path)
