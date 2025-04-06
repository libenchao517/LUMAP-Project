################################################################################
# 本文件用于对流形学习的对比算法进行封装和标准化
################################################################################
# 导入模块
import io
import sys
import warnings
warnings.filterwarnings("ignore")
from time import perf_counter
import numpy as np
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import DictionaryLearning
from sklearn.decomposition import IncrementalPCA
from DR import LocalityPreservingProjection
from DR import PNN_LPP
from DR import Neighborhood_Preserving_Embedding
from DR import TD_PCA
from DeepNet import CNN_2560_768
from DeepNet import TDCNN_GCFOREST
from DeepNet import LeNet_5
from DeepNet import LiNet
from DeepNet import One_Dcnn_Softmax
from DeepNet import TICNN
from DeepNet import WDCNN
from DeepNet import GC_Forest
from DeepNet import MA_1DCNN
from DeepNet import MIX_CNN
from sklearn.model_selection import train_test_split
################################################################################
class Contrast_Method_LUMAP:
    """
    对LUMAP项目中的对比方法进行统一封装
    """
    def __init__(
        self,
        data,
        target,
        n_components,
        sample_height=32,
        sample_weight=32,
        func_name='UMAP',
        data_name='USPS',
        return_time=True,
        sec_part='Comparatation',
        sec_num=0
    ):
        """
        初始化函数
        :param data:   训练数据
        :param target: 训练标签
        :param n_components: 目标维度
        :param sample_height: 2D矩阵的高度
        :param sample_weight: 2D矩阵的宽度
        :param func_name:   算法名称
        :param data_name:   数据名称
        :param return_time: 是否返回时间
        :param sec_part:    项目名称
        :param sec_num:     实验序号
        """
        self.X_train = data
        self.T_train = target
        self.n_components = n_components
        self.func_name = func_name
        self.data_name = data_name
        self.para = [sec_part, str(sec_num), func_name, data_name, 'total']
        self.return_time = return_time
        self.time = None
        self.start_text = "当前正在" + data_name + "数据集上运行" + func_name + "算法......"
        self.Y_train = None
        self.sample_height = sample_height
        self.sample_weight = sample_weight

    def Print_time(self):
        """
        输出时间
        :return:
        """
        print("\r",
              "{:8s}".format(self.para[2]),
              "{:8s}".format(self.para[3]),
              "{:8s}".format("time"),
              "{:.6F}".format(self.time_end - self.time_start) + " " * 20)
        if self.return_time:
            self.time = self.time_end - self.time_start

    def Linear_MAP(self, oos_data, oos_target):
        """
        统一的调用函数
        :param oos_data:   局外样本数据
        :param oos_target: 局外样本标签
        :return:
        """
        self.X_test = oos_data
        self.T_test = oos_target
        print(self.start_text, end="")
        self.time_start = perf_counter()  # 开始计时
        eval("self." + self.func_name.upper() + "_MAP()")
        self.time_end = perf_counter()    # 结束计时
        if self.Y_train is None:
            self.Y_train = self.X_train @ self.components_
            self.Y_test = self.X_test @ self.components_
        if np.allclose(np.imag(self.Y_train), 0):
            self.Y_train = self.Y_train.real
        if np.allclose(np.imag(self.Y_test), 0):
            self.Y_test = self.Y_test.real
        self.Print_time()

    def PCA_MAP(self):
        """
        使用PCA分别对训练数据和局外样本进行降维
        :return: None
        """
        model = PCA(n_components=self.n_components)
        model.fit(self.X_train)
        self.components_ = model.components_.T

    def TDPCA_MAP(self):
        """
        使用2DPCA分别对训练数据和局外样本进行降维
        :return: None
        """
        self.Y_train, self.Y_test = TD_PCA(
            mode="dimension", transform_mode="projection",
            n_components=self.n_components).fit_transform(
            self.X_train, oos_data=self.X_test,
            sample_height=self.sample_height,
            sample_weight=self.sample_weight)

    def UMAP_MAP(self):
        """
        使用UMAP分别对训练数据和局外样本进行降维
        :return: None
        """
        X = np.concatenate((self.X_train, self.X_test))
        embedding = umap.UMAP(n_components=self.n_components).fit_transform(X)
        self.Y_train = embedding[:len(self.X_train)]
        self.Y_test = embedding[len(self.X_train):]

    def PUMAP_MAP(self):
        """
        使用PUMAP分别对训练数据和局外样本进行降维
        :return: None
        """
        original = sys.stdout
        output = io.StringIO()
        sys.stdout = output
        PUMAP = umap.ParametricUMAP(n_components=self.n_components).fit(self.X_train)
        self.Y_train = PUMAP.transform(self.X_train)
        self.Y_test = PUMAP.transform(self.X_test)
        sys.stdout = original
        del output

    def IPCA_MAP(self):
        """
        使用Incremental PCA分别对训练数据和局外样本进行降维
        :return: None
        """
        model = IncrementalPCA(n_components=self.n_components)
        model.fit(self.X_train)
        self.components_ = model.components_.T

    def ICA_MAP(self):
        """
        使用ICA分别对训练数据和局外样本进行降维
        :return: None
        """
        model = FastICA(n_components=self.n_components)
        model.fit(self.X_train)
        self.components_ = model.components_.T

    def FA_MAP(self):
        """
        使用因子分析分别对训练数据和局外样本进行降维
        :return: None
        """
        model = FactorAnalysis(n_components=self.n_components)
        model.fit(self.X_train)
        self.components_ = model.components_.T

    def DL_MAP(self):
        """
        使用字典学习分别对训练数据和局外样本进行降维
        :return: None
        """
        model = DictionaryLearning(n_components=self.n_components)
        model.fit(self.X_train)
        self.components_ = model.components_.T

    def LPP_MAP(self):
        """
        使用LPP分别对训练数据和局外样本进行降维
        :return: None
        """
        if self.X_train.shape[0] < self.X_train.shape[1]:
            model = PCA(n_components=self.X_train.shape[0]-1)
            model.fit(self.X_train)
            PCA_components_ = model.components_.T
            temp_train = self.X_train @ PCA_components_
            temp_test = self.X_test @ PCA_components_
        else:
            temp_train = self.X_train
            temp_test = self.X_test
        self.components_ = LocalityPreservingProjection(
            n_components=self.n_components,
            transform_mode="mapping"
        ).fit_transform(temp_train)
        self.Y_train = temp_train @ self.components_
        self.Y_test = temp_test @ self.components_

    def PNNLPP_MAP(self):
        """
        使用PNNLPP分别对训练数据和局外样本进行降维
        :return: None
        """
        self.components_ = PNN_LPP(
            n_components=self.n_components,
            transform_mode="mapping").fit_transform(self.X_train)

    def NPE_MAP(self):
        """
        使用NPE分别对训练数据和局外样本进行降维
        :return: None
        """
        if self.X_train.shape[0] < self.X_train.shape[1]:
            model = PCA(n_components=self.X_train.shape[0]-1)
            model.fit(self.X_train)
            PCA_components_ = model.components_.T
            temp_train = self.X_train @ PCA_components_
            temp_test = self.X_test @ PCA_components_
        else:
            temp_train = self.X_train
            temp_test = self.X_test
        self.components_ = Neighborhood_Preserving_Embedding(
            n_components=self.n_components,
            transform_mode="mapping").fit_transform(temp_train)
        self.Y_train = temp_train @ self.components_
        self.Y_test = temp_test @ self.components_

class Contrans_Method_Deep_Net:
    """
    对故障诊断任务中的网络方法进行封装
    """
    def __init__(
        self,
        x_train,
        x_test,
        t_train,
        t_test,
        sample_height=32,
        sample_weight=32,
        class_num=10,
        epoch=100,
        device="cpu",
        func_name='UMAP',
        data_name='USPS',
        return_time=True,
        sec_part='Comparatation',
        sec_num=0
    ):
        """
        初始化函数
        :param x_train: 训练数据
        :param x_test:  测试数据
        :param t_train: 训练标签
        :param t_test:  测试标签
        :param sample_height: 2D矩阵高度
        :param sample_weight: 2D矩阵宽度
        :param class_num: 类别数量
        :param epoch:     网络迭代次数
        :param device:    运行网络的设备
        :param func_name: 网络名称
        :param data_name: 数据名称
        :param return_time: 是否返回时间
        :param sec_part:    项目名称
        :param sec_num:     实验编号
        """
        self.x_train = x_train
        self.x_test = x_test
        self.t_train = t_train
        self.t_test = t_test
        self.sample_height = sample_height
        self.sample_weight = sample_weight
        self.class_num = class_num
        self.epoch = epoch
        self.device = device
        self.high_dimension = sample_weight*sample_height
        self.func_name = func_name
        self.data_name = data_name
        self.para = [sec_part, str(sec_num), func_name, data_name, 'total']
        self.return_time = return_time
        self.time = None
        self.start_text = "当前正在" + data_name + "数据集上运行" + func_name + "算法......"

    def Print_time(self):
        """
        格式输出时间
        :return:
        """
        print(
            "\r",
            "{:8s}".format(self.para[2]),
            "{:8s}".format(self.para[3]),
            "{:8s}".format("time"),
            "{:.6F}".format(self.time_end - self.time_start) + " " * 20)
        if self.return_time:
            self.time = self.time_end - self.time_start

    def Run_Deep_Net(self):
        """
        统一的调用函数
        :return:
        """
        print(self.start_text, end="")
        self.time_start = perf_counter()       # 开始计时
        eval("self." + self.func_name + "_()") # 调用网络
        self.time_end = perf_counter()         # 结束计时
        self.Print_time()

    def CNN_2560_768_(self):
        """
        CNN-2560-768 网络
        :return:
        """
        model = CNN_2560_768(
            sample_height=self.sample_height,
            sample_weight=self.sample_weight,
            num_classes=self.class_num,
            epoch=self.epoch)
        self.y_pred = model.fit_transform(
            self.x_train, self.x_test, self.t_train, self.t_test)

    def TDCNN_GCFOREST_(self):
        """
        2DCNN + GCForest
        :return:
        """
        model = TDCNN_GCFOREST(
            sample_height=self.sample_height,
            sample_weight=self.sample_weight,
            num_classes=self.class_num,
            epoch=self.epoch)
        self.y_pred = model.fit_transform(
            self.x_train, self.x_test, self.t_train, self.t_test)

    def LeNet_5_(self):
        """
        LeNet-5
        :return:
        """
        model = LeNet_5(
            sample_height=self.sample_height,
            sample_weight=self.sample_weight,
            num_classes=self.class_num,
            epoch=self.epoch)
        self.y_pred = model.fit_transform(
            self.x_train, self.x_test, self.t_train, self.t_test)

    def LiNet_(self):
        """
        LiNet
        :return:
        """
        model = LiNet(
            sample_height=self.sample_height,
            sample_weight=self.sample_weight,
            num_classes=self.class_num,
            epoch=self.epoch)
        self.y_pred = model.fit_transform(
            self.x_train, self.x_test, self.t_train, self.t_test)

    def One_Dcnn_Softmax_(self):
        """
        1DCNN + Softmax
        :return:
        """
        model = One_Dcnn_Softmax(
            sample_height=1,
            sample_weight=self.high_dimension,
            num_classes=self.class_num,
            epoch=self.epoch)
        self.y_pred = model.fit_transform(
            self.x_train, self.x_test, self.t_train, self.t_test)

    def TICNN_(self):
        """
        TICNN
        :return:
        """
        model = TICNN(
            sample_height=1,
            sample_weight=self.high_dimension,
            num_classes=self.class_num,
            epoch=self.epoch)
        self.y_pred = model.fit_transform(
            self.x_train, self.x_test, self.t_train, self.t_test)

    def WDCNN_(self):
        """
        WDCNN
        :return:
        """
        model = WDCNN(
            sample_height=1,
            sample_weight=self.high_dimension,
            num_classes=self.class_num,
            epoch=self.epoch)
        self.y_pred = model.fit_transform(
            self.x_train, self.x_test, self.t_train, self.t_test)

    def GC_Forest_(self):
        """
        GCForest
        :return:
        """
        model = GC_Forest(
            sample_height=self.sample_height,
            sample_weight=self.sample_weight,
            num_classes=self.class_num,
            epoch=self.epoch)
        self.y_pred = model.fit_transform(
            self.x_train, self.x_test, self.t_train, self.t_test)

    def MA_1DCNN_(self):
        """
        MA1DCNN
        :return:
        """
        model = MA_1DCNN(
            sample_height=1,
            sample_weight=self.high_dimension,
            num_classes=self.class_num,
            epoch=self.epoch,
            device=self.device)
        self.y_pred = model.fit_transform(
            self.x_train, self.x_test, self.t_train, self.t_test)

    def MIX_CNN_(self):
        """
        MIXCNN
        :return:
        """
        model = MIX_CNN(
            sample_height=1,
            sample_weight=self.high_dimension,
            num_classes=self.class_num,
            epoch=self.epoch,
            device=self.device)
        self.y_pred = model.fit_transform(
            self.x_train, self.x_test, self.t_train, self.t_test)
