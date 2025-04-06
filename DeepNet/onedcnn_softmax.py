################################################################################
# 本文件用于实现CNN-2560-768网络
################################################################################
# 导入模块
from tensorflow import keras
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow as tf
import numpy as np
################################################################################
# 定义网络
class One_Dcnn_Softmax:
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
        inputs = keras.Input(shape=(self.sample_weight, self.sample_height))
        h1 = layers.Conv1D(filters=8, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)
        h1 = layers.MaxPool1D(pool_size=2, strides=2, padding='same')(h1)

        h1 = layers.Conv1D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu')(h1)
        h1 = layers.MaxPool1D(pool_size=2, strides=2, padding='same')(h1)

        h1 = layers.Flatten()(h1)
        h1 = layers.Dropout(0.6)(h1)
        h1 = layers.Dense(32, activation='relu')(h1)
        h1 = layers.Dense(self.num_classes, activation='softmax')(h1)
        deep_model = keras.Model(inputs, h1, name="cnn")
        return deep_model

    def fit(self, X_train, X_test, T_train, T_test):
        model = self.def_model()
        model.summary()
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

        # 提取CNN模型中间层特征
        intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
        x_test_features = intermediate_layer_model.predict(X_test)

        model.fit(X_train, T_train,
                batch_size=128, epochs=self.epoch, verbose=1,
                validation_data=(X_test, T_test))

        # 输出最终预测结果
        y_predict = model.predict(X_test)
        final_predictions = np.argmax(y_predict, axis=1)
        return final_predictions

    def fit_transform(self, X_train, X_test, T_train, T_test):
        X_train = tf.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = tf.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        T_train = T_train.astype(int)
        T_test = T_test.astype(int)
        T_pred = self.fit(X_train, X_test, T_train, T_test)
        return T_pred
