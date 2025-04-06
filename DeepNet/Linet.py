################################################################################
# 本文件用于实现LiNet网络
################################################################################
# 导入模块
from tensorflow import keras
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow as tf
import numpy as np
################################################################################
# 定义网络
class LiNet:
    """
    Jin T, Yan C, Chen C, et al.
    Light neural network with fewer parameters based on CNN for fault diagnosis of rotating machinery[J].
    Measurement, 2021, 181: 109639.
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
        inputs = keras.Input(shape=(self.sample_weight, self.sample_height, 1))
        h1 = layers.Conv2D(filters=16, kernel_size=(3,1), padding='same', activation='relu')(inputs)
        h1 = layers.BatchNormalization()(h1)
        h1 = layers.MaxPool2D(pool_size=(2,1), padding='same')(h1)

        # Light module(a)
        ha1 = layers.Conv2D(filters=8, kernel_size=(1,1), padding='same', activation='relu')(h1)
        ha1 = layers.BatchNormalization()(ha1)
        ha1 = layers.Conv2D(filters=16, kernel_size=(1,3), padding='same', activation='relu')(ha1)
        ha1 = layers.BatchNormalization()(ha1)
        ha1 = layers.Conv2D(filters=16, kernel_size=(1,1), padding='same', activation='relu')(ha1)
        ha1 = layers.BatchNormalization()(ha1)

        ha2 = layers.Conv2D(filters=8, kernel_size=(1,1), padding='same', activation='relu')(h1)
        ha2 = layers.BatchNormalization()(ha2)
        ha2 = layers.Conv2D(filters=16, kernel_size=(1,5), padding='same', activation='relu')(ha2)
        ha2 = layers.BatchNormalization()(ha2)
        ha2 = layers.Conv2D(filters=16, kernel_size=(1,1), padding='same', activation='relu')(ha2)
        ha2 = layers.BatchNormalization()(ha2)

        # 连接
        h1 = layers.concatenate([ha1, ha2], axis=-1);
        h1 = layers.MaxPool2D(pool_size=(2,1), padding='same')(h1)

        # Light module(b)
        hb1 = layers.Conv2D(filters=32, kernel_size=(1,1), padding='same', activation='relu')(h1)
        hb1 = layers.BatchNormalization()(hb1)
        hb1 = layers.Conv2D(filters=16, kernel_size=(1,3), padding='same', activation='relu')(hb1)
        hb1 = layers.BatchNormalization()(hb1)
        hb1 = layers.Conv2D(filters=16, kernel_size=(1,1), padding='same', activation='relu')(hb1)
        hb1 = layers.BatchNormalization()(hb1)

        hb2 = layers.Conv2D(filters=32, kernel_size=(1,1), padding='same', activation='relu')(h1)
        hb2 = layers.BatchNormalization()(hb2)
        hb2 = layers.Conv2D(filters=16, kernel_size=(1,5), padding='same', activation='relu')(hb2)
        hb2 = layers.BatchNormalization()(hb2)
        hb2 = layers.Conv2D(filters=16, kernel_size=(1,1), padding='same', activation='relu')(hb2)
        hb2 = layers.BatchNormalization()(hb2)

        # 连接
        h1 = layers.concatenate([hb1, hb2], axis=-1);
        h1 = layers.MaxPool2D(pool_size=(2,1), padding='same')(h1)
        h1 = layers.Conv2D(filters=16, kernel_size=(1,1), padding='same', activation='relu')(h1)
        h1 = layers.GlobalAveragePooling2D()(h1)
        h1 = layers.Dense(self.num_classes, activation='softmax')(h1)

        LiNet = keras.Model(inputs, h1, name="LiNet")
        return LiNet

    def fit(
            self,
            X_train,
            X_test,
            T_train,
            T_test
    ):
        model = self.def_model()
        model.summary()
        # 编译模型
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
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
        X_train = X_train.reshape(-1, self.sample_weight, self.sample_height, 1)
        X_test = X_test.reshape(-1, self.sample_weight, self.sample_height, 1)
        T_pred = self.fit(X_train, X_test, T_train, T_test)
        return T_pred
