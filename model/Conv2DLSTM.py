import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.models import Model




def convls(n_hidden, train1_x, train2_x, train_y):
    sequence_len = np.array(train1_x)[1]
    vales_1 = np.array(train1_x)[2]
    value_2 = np.array(train2_x)[2]

    # 入力を定義
    input1 = Input(shape=(sequence_len,vales_1))
    input2 = Input(shape=(sequence_len,value_2))

    # 入力1から結合前まで(Conv1D)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, dilation_rate=1, activation='relu')(input1)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, dilation_rate=1, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, dilation_rate=2, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, dilation_rate=3, activation='relu')(x)

    x = Model(inputs=input1, outputs=x)

    # 入力2から結合前まで(Conv1D)
    y = tf.keras.layers.Conv1D(filters=64, kernel_size=3, dilation_rate=1, activation='relu')(input2)
    y = tf.keras.layers.Conv1D(filters=64, kernel_size=3, dilation_rate=1, activation='relu')(y)
    y = tf.keras.layers.Conv1D(filters=64, kernel_size=3, dilation_rate=2, activation='relu')(y)
    y = tf.keras.layers.Conv1D(filters=64, kernel_size=3, dilation_rate=3, activation='relu')(y)

    y = Model(inputs=input2, outputs=y)

    # 結合
    combined = concatenate([x.output, y.output])

    # 密結合
    z = tf.keras.layers.LSTM(units=n_hidden, return_sequences=True)(combined)
    z = tf.keras.layers.LSTM(units=n_hidden, return_sequences=False)(z)


    z = tf.keras.layers.Dense(units=50, activation='relu')(z)
    z = tf.keras.layers.Dense(class_num, activation='softmax')(z)

    # モデル定義とコンパイル
    model = tf.keras.Model(inputs=[x.input, y.input], outputs=z)


    model.compile(loss='sparse_categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
    model.fit([train1_x, train2_x], train_y,
            batch_size=100,
            epochs=100,
            validation_split=0.1,
            callbacks=[callback]
            )

    return model