import numpy as np
import pandas as pd
import glob


from sklearn.preprocessing import MinMaxScaler



def pick_filer(sequence_len, trte, csv_columns):
    def normalize(df, name):
        # MinMaxScalerモジュールのインスタンス生成
        mmsc = MinMaxScaler()

        # dfデータを引っ張り出す
        df_data = df[[name]]

        # 標準化
        df_data = abs(df_data)
        normed_data = mmsc.fit_transform(df_data)
        df_data_0 = pd.DataFrame(data=normed_data)
        df_data_fin = df_data_0.rename(columns={0: name})

        return df_data_fin

    def append_for(df_all, data):
        for i in range(len(data)):
            df_all.append(data[i])
        return df_all

    # fileの作成
    files = glob.glob('data/' + trte + '/*.csv')
    # データフレームへの変換
    df_all_values = []
    df_all_label = []

    for t in range(len(files)):
        data_file = pd.read_csv(files[t])
        '''使うデータのみ取り出し'''
        df_use = data_file[csv_columns]


        '''データの前処理'''
        df_other = df_use[:, ['label']]
        df_other_fin = df_other.reset_index(drop=True)

        """csv_"""
        df_loss = normalize(df_use, csv_columns[0])
        df_data = df_loss.join([normalize(df_use, csv_columns[1]),
                                normalize(df_use, csv_columns[2]),
                                normalize(df_use, csv_columns[3]),
                                df_other_fin
                                ])

        df_loss = normalize(df_use, csv_columns[0], '0')
        num = len(csv_columns) - 1
        for s in range(num):
            if s == 0:
                df_data = df_loss.join([normalize(df_use, csv_columns[s + 1], '0')])
            else:
                df_data = df_data.join([normalize(df_use, csv_columns[s + 1], '0')])

        '''学習・テストデータの作成'''
        values = np.array(df_data)
        label = np.array(df_other_fin)

        # sequence_lenごとに分ける
        sliced_valies = [values[i:i + sequence_len] for i in range(values.shape[0] - sequence_len)]
        sliced_valies = np.array(sliced_valies)

        sliced_label = [label[i + sequence_len] for i in range(len(values) - sequence_len)]
        sliced_label = np.array(sliced_label)

        df_all_values = append_for(df_all_values, sliced_valies)
        df_all_label = append_for(df_all_label, sliced_label)

    return np.array(df_all_values), np.array(df_all_label)