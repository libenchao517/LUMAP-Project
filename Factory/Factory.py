################################################################################
# 本文件用于实现流形学习算法模型工厂
################################################################################
# 导入模块
import numpy as np
import random
import datetime
import platform
from sklearn.decomposition import PCA
from Contrast import Contrast_Method_LUMAP
from Contrast import Contrans_Method_Deep_Net
from DATA import Load_Data
from DATA import datas
from DATA import abbre_labels
from DATA import Pre_Procession as PP
from LUMAP import LUMAP
from LUMAP import L_2D_UMAP
from LUMAP import LUMAP_Supervised_Labels
from LUMAP import LUMAP_config
from LUMAP import Deep_Net_config
################################################################################
class factory:
    """
    模型工厂
    """
    def __init__(
        self,
        func_name='UMAP',
        data_name='USPS',
        return_time=True,
        train_size=0.1,
        split_type=None,
        random_state=None,
        is_noisy=False,
        sigma=0.01,
        is_clip=False,
        clip_num=0,
        is_select_target = False,
        target_num = 0,
        sec_part='Comparatation',
        sec_num=0,
    ):
        """
        初始化函数
        :param func_name:   算法名称
        :param data_name:   数据名称
        :param return_time: 是否返回时间
        :param train_size:  训练比例
        :param split_type:  训练集测试集划分方法
        :param random_state: 随机种子
        :param is_noisy:     是否添加噪声
        :param sigma:        高斯噪声强度
        :param is_clip:      是否切割数据
        :param clip_num:     子集的规模
        :param is_select_target: 是否根据类别进行采样
        :param target_num:   选择的类别的数量
        :param sec_part:     项目名称
        :param sec_num:      实验序号
        """
        self.xn = 80
        print("#" * self.xn)
        print(func_name + "算法性能测试")
        print("*" * self.xn)
        print("性能指标：")
        print("*" * self.xn)
        print("测试日期：", datetime.date.today())
        print("测试时间：", datetime.datetime.now().time().strftime("%H:%M:%S"))
        print("计算机名：", platform.node())
        print("操作系统：", platform.system())
        print("解 释 器：", platform.python_version())
        print("数 据 集：", data_name)
        print("算法名称：", func_name)
        print("*" * self.xn)
        self.data_name = data_name
        self.func_name = func_name
        self.random_state = random_state
        self.return_time = return_time
        self.train_size = train_size
        self.split_type = split_type
        self.is_noisy = is_noisy
        self.sigma =sigma
        self.is_clip = is_clip
        self.clip_num = clip_num
        self.is_select_target = is_select_target
        self.target_num = target_num
        self.sec_part = sec_part
        self.sec_num = sec_num

    def Product_Linear_UMAP_Object(
            self,
            type="1D",
            n_components=20,
            n_neighbors=15,
            init="Random",
            label_num=15,
            is_Supervised = False,
            Supervised_mode = "NPE",
            pca_rate=0.95,
            transform_mode="Projection"
    ):
        """
        生产线性UMAP算法对象
        :param type:             线性方法的类型
        :param n_components:     目标维度
        :param n_neighbors:      近邻数
        :param init:             初始化映射矩阵的方法
        :param label_num:        标签向量的维度
        :param is_Supervised:    是否进行监督
        :param Supervised_mode:  监督方式
        :param pca_rate:         PCA的累积贡献率
        :param transform_mode:   降维模式
        :return:
        """
        config = LUMAP_config()
        # 加载数据集
        if self.data_name in config.MFD_dict.keys():
            dn = config.MFD_dict.get(self.data_name)
        else:
            dn = self.data_name
        data, target = Load_Data(dn).Loading()
        legd = abbre_labels.get(dn)
        # 对数据集的类别的进行随机采样
        if self.is_select_target:
            index_list = [i for i in range(len(legd))]
            index_select = random.sample(index_list, k=self.target_num)
            index_select.sort()
            config.LUMAP_select_target[self.data_name] = index_select

        if config.LUMAP_select_target.get(self.data_name) is not None:
            data, target = PP().select_target(data, target, config.LUMAP_select_target.get(self.data_name))

        target, mapping = PP().target_mirror(target)
        if config.LUMAP_select_target.get(self.data_name) is not None and legd is not None:
            temp = [int(mapping.get(i)) for i in config.LUMAP_select_target.get(self.data_name)]
            legd = [legd[i] for i in temp]
        # 采用数据集的子集
        if self.is_clip:
            _, data, _, target = PP().sub_one_sampling(data, target, train_size=self.clip_num, random_state=self.random_state)
        # 为数据集添加高斯噪声
        if self.is_noisy:
            data = PP().add_gaussian_noise(data, sigma=self.sigma)
        # 划分数据集
        self.X_train, self.X_test, self.T_train, self.T_test = PP().uniform_sampling(data, target, train_size=self.train_size, random_state=self.random_state)
        # 制作新的标签向量
        if is_Supervised:
            self.label = LUMAP_Supervised_Labels(model=Supervised_mode).Make_label(self.X_train, self.T_train, label_num, len(set(target)))
        # 初始化对象
        if type=="1D":
            self.Linear_UMAP_Object = LUMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                init=init,
                transform_mode=transform_mode,
                return_time=self.return_time,
                data_name=self.data_name,
                func_name=self.func_name,
                sec_part=self.sec_part,
                sec_num=self.sec_num)
            self.Linear_UMAP_Object.legd = legd
            return self.Linear_UMAP_Object
        elif type=="2D":
            self.Linear_2DUMAP_Object = L_2D_UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                transform_mode=transform_mode,
                return_time=self.return_time,
                data_name=self.data_name,
                func_name=self.func_name,
                sec_part=self.sec_part,
                sec_num=self.sec_num)
            self.Linear_2DUMAP_Object.legd = legd
            return self.Linear_2DUMAP_Object

    def Product_Linear_Contrast_Object(self, n_components=20):
        """
        生产线性降维算法对象
        :param n_components: 目标维度
        :return:
        """
        config = LUMAP_config()
        # 加载数据集
        if self.data_name in config.MFD_dict.keys():
            dn = config.MFD_dict.get(self.data_name)
        else:
            dn = self.data_name
        data, target = Load_Data(dn).Loading()
        legd = abbre_labels.get(dn)
        # 对数据集的类别的进行随机采样
        if self.is_select_target:
            index_list = [i for i in range(len(legd))]
            index_select = random.sample(index_list, k=self.target_num)
            index_select.sort()
            config.LUMAP_select_target[self.data_name] = index_select

        if config.LUMAP_select_target.get(self.data_name) is not None:
            data, target = PP().select_target(data, target, config.LUMAP_select_target.get(self.data_name))

        target, mapping = PP().target_mirror(target)
        if config.LUMAP_select_target.get(self.data_name) is not None and legd is not None:
            temp = [int(mapping.get(i)) for i in config.LUMAP_select_target.get(self.data_name)]
            legd = [legd[i] for i in temp]
        # 采用数据集的子集
        if self.is_clip:
            _, data, _, target = PP().sub_one_sampling(data, target, train_size=self.clip_num, random_state=self.random_state)
        # 为数据集添加高斯噪声
        if self.is_noisy:
            data = PP().add_gaussian_noise(data, sigma=self.sigma)
        # 划分训练集和测试集
        self.X_train, self.X_test, self.T_train, self.T_test = PP().uniform_sampling(data, target, train_size=self.train_size, random_state=self.random_state)
        # 初始化对象
        self.Linear_Contrast_Object = Contrast_Method_LUMAP(
            data=self.X_train,
            target=self.T_train,
            n_components=n_components,
            data_name=self.data_name,
            func_name=self.func_name,
            return_time=self.return_time,
            sec_part=self.sec_part,
            sec_num=self.sec_num)
        self.Linear_Contrast_Object.legd = legd
        return self.Linear_Contrast_Object

    def Product_Deep_Net_FD_Object(self):
        """
        生成深度网络模型的工厂
        :return:
        """
        config = Deep_Net_config()
        # 加载数据集
        if self.data_name in config.MFD_dict.keys():
            dn = config.MFD_dict.get(self.data_name)
        else:
            dn = self.data_name
        data, target = Load_Data(dn).Loading()
        legd = abbre_labels.get(dn)
        # 对数据集的类别的进行随机采样
        if self.is_select_target:
            index_list = [i for i in range(len(legd))]
            index_select = random.sample(index_list, k=self.target_num)
            index_select.sort()
            config.LUMAP_select_target[self.data_name] = index_select

        if config.LUMAP_select_target.get(self.data_name) is not None:
            data, target = PP().select_target(data, target, config.LUMAP_select_target.get(self.data_name))

        target, mapping = PP().target_mirror(target)
        if config.LUMAP_select_target.get(self.data_name) is not None and legd is not None:
            temp = [int(mapping.get(i)) for i in config.LUMAP_select_target.get(self.data_name)]
            legd = [legd[i] for i in temp]
        # 采用数据集的子集
        if self.is_clip:
            _, data, _, target = PP().sub_one_sampling(data, target, train_size=self.clip_num, random_state=self.random_state)
        # 为数据集添加高斯噪声
        if self.is_noisy:
            data = PP().add_gaussian_noise(data, sigma=self.sigma)
        # 划分训练集和测试集
        X_train, X_test, T_train, T_test = PP().uniform_sampling(
            data, target, train_size=self.train_size, random_state=self.random_state)
        # 初始化对象
        self.Linear_Contrast_Net_Object = Contrans_Method_Deep_Net(
            x_train=X_train, x_test=X_test, t_train=T_train, t_test=T_test,
            sample_height=config.Net_data[self.data_name][0],
            sample_weight=config.Net_data[self.data_name][1],
            class_num=len(np.unique(target)), epoch=config.epoch,
            device=config.device,
            func_name=self.func_name, data_name=self.data_name,
            return_time=self.return_time,
            sec_part=self.sec_part, sec_num=self.sec_num
        )
        self.Linear_Contrast_Net_Object.legd = legd
        return self.Linear_Contrast_Net_Object
