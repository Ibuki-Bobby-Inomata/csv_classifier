import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd

from sklearn.metrics import confusion_matrix




def plot_result(model, test_x, test_y, title, target_names):
    def print_cmx(y_true, y_pred, labels, title):
       cmx_data = confusion_matrix(y_true, y_pred)

       cmx_data = cmx_data.astype('float') / cmx_data.sum(axis=1)[:, np.newaxis]

       cmx_data = cmx_data.round(2)

       df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

       plt.figure(figsize=(15, 15))
       sn.heatmap(df_cmx, annot=True, fmt='g', square=True, cmap='Reds', vmin=0.0, vmax=1.0)
       sn.set_context('poster')

       plt.ylim(len(target_names), 0)
       plt.xlabel("Predict-labels")
       plt.ylabel("True-labels")
       plt.title(title)

       plt.savefig('image/'+title+'.png')

       return

    predict_classes = model.predict(test_x)
    true_classes = test_y

    pre_cla = []

    for i in range(len(predict_classes)):
       max_ = np.amax(predict_classes[i])
       for j in range(len(predict_classes[i])):
            if max_==predict_classes[i][j]:
                pre_cla.append(j)

    predict_classes = np.array(pre_cla)

    print_cmx(true_classes, predict_classes, target_names, title)

    return