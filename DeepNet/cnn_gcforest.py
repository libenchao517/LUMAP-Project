################################################################################
# 本文件用于实现2DCNN + GCForest分类器
################################################################################
# 导入模块
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from keras import backend as K
import numpy as np
from .GCForest import gcForest
################################################################################
# 定义网络
class TDCNN_GCFOREST:
    """
    Xu Y, Li Z, Wang S, et al.
    A hybrid deep-learning model for fault diagnosis of rolling bearings[J].
    Measurement, 2021, 169: 108502.
    """
    def __init__(
            self,
            sample_height=10,
            sample_weight=10,
            num_classes=10,
            epoch=500
    ):
        self.sample_height = sample_height
        self.sample_weight = sample_weight
        self.num_classes = num_classes
        self.epoch = epoch

    def def_model(self):
        model = Sequential()
        model.add(Conv2D(32, (5, 5), input_shape=(self.sample_weight, self.sample_height, 1), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(self.num_classes, activation='sigmoid'))
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def get_layer_output(self, model, x, index):
        layer=K.function([model.input],[model.layers[index].output])
        return layer([x])[0]

    def fit(self, X_train, X_test, T_train, T_test, T_train_copy):
        model = self.def_model()
        # 训练模型
        model.fit(X_train, T_train, batch_size=128, epochs=self.epoch)
        # 评估模型
        loss, accuracy = model.evaluate(X_test, T_test)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        layer_train = self.get_layer_output(model, X_train, index=-3);
        layer_test = self.get_layer_output(model, X_test, index=-3);
        # 使用gcForest进行后处理
        gc = gcForest(shape_1X=[16, 16], n_mgsRFtree=30, window=14, stride=1, cascade_test_size=0.2, n_cascadeRF=2,
                      n_cascadeRFtree=101, cascade_layer=np.inf, min_samples_mgs=2, min_samples_cascade=2,
                      tolerance=0.0, n_jobs=1)
        gc.fit(layer_train, T_train_copy)
        # 输出最终预测结果
        final_predictions = gc.predict(layer_test)
        return final_predictions

    def fit_transform(self, X_train, X_test, T_train, T_test):
        T_train = T_train.astype(int)
        T_test = T_test.astype(int)
        T_train_copy = T_train
        T_train = np.eye(self.num_classes)[T_train]
        T_test = np.eye(self.num_classes)[T_test]
        X_train = X_train.reshape(-1, self.sample_weight, self.sample_height, 1)
        X_test = X_test.reshape(-1, self.sample_weight, self.sample_height, 1)
        T_pred = self.fit(X_train, X_test, T_train, T_test, T_train_copy)
        return T_pred
