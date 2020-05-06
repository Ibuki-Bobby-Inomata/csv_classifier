import tensorflow as tf


def lstm(n_hidden, sequence_len, in_out, class_num, train_x, train_y):
    '''LSTMモデル定義'''
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.LSTM(units=n_hidden, return_sequences=False, input_shape=(sequence_len, in_out)))

    model.add(tf.keras.layers.Dense(units=100, activation='relu'))
    model.add(tf.keras.layers.Dense(class_num, activation='softmax'))

    model.sammary()

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