import warnings
import sys
import tensorflow as tf
keras = tf.keras

from model_module import data_converter as dc
from model_module import plot_result
from model import LSTM

warnings.filterwarnings('ignore')



'''train, testデータの作成関数'''
if __name__ == '__main__':
    print('type used column')
    column = input()

    sequence_len = 10
    csv_columns = [column]
    train_x, train_y = dc.pick_filer(sequence_len, 'train', csv_columns)
    test_x, test_y = dc.pick_filer(sequence_len, 'test', csv_columns)

    in_out_neurons = 27
    class_num = 7
    n_hidden = 128
    model = LSTM.lstm(n_hidden, sequence_len, in_out_neurons, class_num, train_x, train_y)

    '''結果の表示'''
    target_names = []
    title = 'result'

    plot_result.plot_result(model, test_x, test_y, title, target_names)


    sys.exit()