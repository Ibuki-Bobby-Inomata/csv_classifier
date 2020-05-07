import tensorflow as tf
import numpy as np

def conv2d(filters, kernel_size, label_num, train_x, train_y):
    is_ = np.sarray(train_x.shape)

    '''LSTMモデル定義'''
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, input_shape=(is_[1], is_[2], is_[3]),
                                     dilation_rate=1, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Conv2D(filters, kernel_size, dilation_rate=1, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Conv2D(filters, kernel_size, dilation_rate=2, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Conv2D(filters, kernel_size, dilation_rate=3, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.GlobalMaxPooling2D())

    model.add(tf.keras.layers.Dense(units=100, activation='relu'))

    model.add(tf.keras.layers.Dense(units=label_num, activation='softmax'))

    model.summary()

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
    model_history = model.fit(train_x, train_y,
                            batch_size=100,
                            epochs=100,
                            validation_split=0.1,
                            callbacks=[callback]
                            )


    return model