################################################################################
# 本文件用于实现LeNet-5网络
################################################################################
# 导入模块
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow as tf
import numpy as np
################################################################################
# 定义网络
class LeNet_5:
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
        model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(self.sample_weight, self.sample_height, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(120, activation='relu'))
        model.add(Dense(84, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def fit(
            self,
            X_train,
            X_test,
            T_train,
            T_test
    ):
        model = self.def_model()
        # 训练模型
        model.fit(X_train, T_train, batch_size=128, epochs=self.epoch)
        # 提取CNN模型中间层特征
        intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
        x_test_features = intermediate_layer_model.predict(X_test)
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
