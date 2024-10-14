import time
import numpy as np

# Random seed
seed = 50
np.random.seed(seed)
import tensorflow

tensorflow.random.set_seed(seed)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def dataset(n_features=6, n_classes=4):
    dataset = pd.read_csv('feature8(160).csv')

    def process(data):
        columns = data.columns
        label = data[columns[6]]
        label = label.tolist()
        data = data.values.tolist()
        X, Y = [], []
        for i in range(len(data)):
            train_seq = []
            for c in range(0, 6):
                train_seq.append(data[i][c])
            X.append(train_seq)
            Y.append(label[i])
        X, Y = np.array(X), np.array(Y)
        length = int(len(X))
        X, Y = X[:length], Y[:length]
        return X, Y

    x, y = process(dataset)
    y = np.ravel(y)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False, random_state=1)
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, y_train, X_test, y_test


def plot_confusion_matrix(cm, labels_name, is_norm=True, colorbar=True, cmap=plt.cm.Blues):
    if is_norm:
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name)
    plt.yticks(num_local, labels_name)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if is_norm:
        plt.savefig(r'.\cm_norm_' + '.png', format='png')
    else:
        plt.savefig(r'.\cm_' + '.png', format='png')
    plt.show()
    plt.close()


n_features = 6
n_classes = 4
X_train, y_train, X_test, y_test = dataset(n_features=n_features, n_classes=n_classes)

# SVM classifier
svm_model = SVC(kernel='linear', random_state=seed)
svm_model.fit(X_train, y_train)

# Training accuracy
y_train_pred = svm_model.predict(X_train)
print('\n Training performance of SVM: {}'.format(accuracy_score(y_train_pred, y_train)))

# Testing accuracy
start_time = time.time()
y_test_pred = svm_model.predict(X_test)
end_time = time.time()
testing_time = end_time - start_time
print('\n Testing performance of SVM: {}'.format(accuracy_score(y_test_pred, y_test)))
print("Testing time: ", testing_time, "seconds")

# Confusion matrix
label_name = ['ds2', 'ds3', 'ds4', 'ds6']
cm = confusion_matrix(y_test, y_test_pred)
plot_confusion_matrix(cm, label_name, is_norm=False)
plot_confusion_matrix(cm, label_name, is_norm=True)
