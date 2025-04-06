################################################################################
# 本文件用于封装GCForest分类器
################################################################################
# 导入模块
import numpy as np
from .GCForest import gcForest
################################################################################
# 定义分类器
class GC_Forest:
    def __init__(self, sample_height=10, sample_weight=10, num_classes=10, epoch=500):
        self.sample_height = sample_height
        self.sample_weight = sample_weight
        self.num_classes = num_classes
        self.epoch = epoch

    def fit(self, X_train, X_test, T_train, T_test):
        # 使用gcForest进行后处理
        gc = gcForest(shape_1X=[self.sample_weight, self.sample_height], n_mgsRFtree=30, window=30, stride=1, cascade_test_size=0.2, n_cascadeRF=2,
                      n_cascadeRFtree=101, cascade_layer=np.inf, min_samples_mgs=2, min_samples_cascade=2,
                      tolerance=0.0, n_jobs=1)
        gc.fit(X_train, T_train)

        # 输出最终预测结果
        final_predictions = gc.predict(X_test)
        return final_predictions

    def fit_transform(self, X_train, X_test, T_train, T_test):
        T_pred = self.fit(X_train, X_test, T_train, T_test)
        return T_pred
