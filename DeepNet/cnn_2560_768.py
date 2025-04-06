################################################################################
# 本文件用于实现CNN-2560-768网络
################################################################################
# 导入模块
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
################################################################################
# 定义网络
class CNN_2560_768:
    """
    Wen L, Li X, Gao L, et al.
    A new convolutional neural network-based data-driven fault diagnosis method[J].
    IEEE transactions on industrial electronics, 2017, 65(7): 5990-5998.
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
        # 创建一个Sequential模型
        model = Sequential()
        # 添加第一层卷积层：Conv(5x5x32)
        model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(self.sample_weight, self.sample_height, 1), padding='same', name='L1'))
        # 添加第二层最大池化层：Maxpool(2x2)
        model.add(MaxPooling2D((2, 2), name='L2'))
        # 添加第三层卷积层：Conv(3x3x64)
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='L3'))
        # 添加第四层最大池化层：Maxpool(2x2)
        model.add(MaxPooling2D((2, 2), name='L4'))
        # 添加第五层卷积层：Conv(3x3x128)
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='L5'))
        # 添加第六层最大池化层：Maxpool(2x2)
        model.add(MaxPooling2D((2, 2), name='L6'))
        # 添加第七层卷积层：Conv(3x3x256)
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='L7'))
        # 添加第八层最大池化层：Maxpool(2x2)
        model.add(MaxPooling2D((2, 2), name='L8'))
        # 添加一个Flatten层，将卷积层输出展平
        model.add(Flatten())
        # 添加第一个全连接层：FC1=2560
        model.add(Dense(2560, activation='relu', name='FC1'))
        # 添加第二个全连接层：FC2=768
        model.add(Dense(768, activation='relu', name='FC2'))
        # 添加用于分类的Softmax输出层
        model.add(Dense(self.num_classes, activation='softmax'))
        # 编译模型
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # 打印模型结构
        model.summary()
        return model

    def fit(
            self,
            X_train,
            X_test,
            T_train,
            T_test
    ):
        model = self.def_model()
        model.fit(X_train, T_train, batch_size=128, epochs=self.epoch, verbose=1)
        # 输出最终预测结果
        y_predict = model.predict(X_test)
        final_predictions = np.argmax(y_predict, axis=1)
        return final_predictions

    def fit_transform(
            self,
            X_train,
            X_test,
            T_train,
            T_test
    ):
        T_train = T_train.astype(int)
        T_test = T_test.astype(int)
        T_train = np.eye(self.num_classes)[T_train]
        T_test = np.eye(self.num_classes)[T_test]
        X_train = X_train.reshape(-1, self.sample_weight, self.sample_height, 1)
        X_test = X_test.reshape(-1, self.sample_weight, self.sample_height, 1)
        T_pred = self.fit(X_train, X_test, T_train, T_test)
        return T_pred
