import time
import numpy

#############randome seed:
# seed = 100
seed = 50
numpy.random.seed(seed)
# TensorFlow has its own random number generator
# from tensorflow import set_random_seed
# set_random_seed(seed)
import tensorflow

tensorflow.random.set_seed(seed)
####################
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import AdaBoostClassifier
# from multi_AdaBoost import AdaBoostClassifier
from Improved_AdaBoost_CNN import AdaBoost_CNN

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
#####deep lCNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Activation
# from keras.layers.convolutional import Conv1D, MaxPooling1D, AveragePooling1D
# from tensorflow.keras.layers import Conv1D, MaxPooling1D

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split  # LabelBinarizer


# theano doesn't need any seed because it uses numpy.random.seed
#######function def:
def train_DNN(X_train=None, y_train=None, epochs=None, batch_size=None, X_test=None, y_test=None, n_features=6,
              seed=100):
    ######ranome seed
    numpy.random.seed(seed)
    # set_random_seed(seed)
    tensorflow.random.set_seed(seed)

    model = baseline_model(n_features, seed)
    # reshape imput matrig to be compatibel to DNN
    # newshape = X_train.shape
    # newshape = list(newshape)
    # newshape.append(1)
    # newshape = tuple(newshape)
    # X_train_r = numpy.reshape(X_train, newshape)  # reshat the trainig data to (2300, 10, 1) for CNN
    # binarize labes:
    # lb=LabelBinarizer()
    # y_train_b = lb.fit_transform(y_train)

    lb = OneHotEncoder(sparse=False)
    y_train_b = y_train.reshape(len(y_train), 1)
    y_train_b = lb.fit_transform(y_train_b)
    # train CNN
    numpy.random.seed(seed)
    tensorflow.random.set_seed(seed)
    # set_random_seed(seed)
    model.fit(X_train, y_train_b, epochs=epochs, batch_size=batch_size)

    #####################reshap test data and evaluate:
    # newshape = X_test.shape
    # newshape = list(newshape)
    # newshape.append(1)
    # newshape = tuple(newshape)
    # X_test_r = numpy.reshape(X_test, newshape)
    # bibarize lables:
    # lb=LabelBinarizer()
    # y_test_b = lb.fit_transform(y_test)
    lb = OneHotEncoder(sparse=False)
    y_test_b = y_test.reshape(len(y_test), 1)
    y_test_b = lb.fit_transform(y_test_b)

    yp = model.evaluate(X_train, y_train_b)
    print('\nSingle DNN evaluation on training data, [loss, test_accuracy]:')
    print(yp)

    start_time = time.time()
    yp = model.evaluate(X_test, y_test_b)
    end_time = time.time()
    testing_time = end_time - start_time
    print('\nSingle DNN evaluation on testing data, [loss, test_accuracy]:')
    print(yp)
    print("测试时间为: ", testing_time, "秒")
    ########################


#####deep CNN
def baseline_model(n_features=6, seed=100, n_classes=8):
    numpy.random.seed(seed)
    # set_random_seed(seed)
    tensorflow.random.set_seed(seed)
    # create model
    # model.add(Conv1D(16, 3, strides=2,padding="same", input_shape=(n_features, 1),kernel_initializer=tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)))
    # model.add(Activation('relu'))
    # model.add(Conv1D(16, 3, strides=1, padding="same", input_shape=(n_features, 1),kernel_initializer=tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)))
    # model.add(Activation('relu'))
    # model.add(Conv1D(32, 3,padding="same", input_shape=(n_features, 1),kernel_initializer=tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)))
    # model.add(Activation('relu'))
    # model.add(Conv1D(32, 3, strides=1, padding="same", input_shape=(n_features, 1),kernel_initializer=tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Conv1D(48, 3, strides=1, padding="same", input_shape=(n_features, 1),kernel_initializer=tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)))
    # model.add(Activation('relu'))
    # # model.add(MaxPooling1D(pool_size=2))
    # model.add(Conv1D(64, 3, strides=1, padding="same", input_shape=(n_features, 1),kernel_initializer=tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)))
    # model.add(Activation('relu'))
    # model.add(Conv1D(96, 3, strides=1, padding="same", input_shape=(n_features, 1),kernel_initializer=tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling1D(pool_size=2))
    #    model.add(Conv1D(32, 3, border_mode='valid',  activation='relu'))
    #    model.add(MaxPooling1D(pool_size=(2, 2)))
    #    model.add(Dropout(0.2))#
    model = Sequential()
    # model.add(Flatten())
    model.add(Dense(128, activation='relu', input_shape=(n_features,), kernel_initializer=tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)))  #
    model.add(Dense(144, activation='relu',kernel_initializer=tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)))
    # model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu',kernel_initializer=tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu',kernel_initializer=tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)))
    # model.add(Dense(256, activation='relu',kernel_initializer=tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)))
    # model.add(Dense(256, activation='relu',kernel_initializer=tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)))
    # model.add(Dense(416, activation='relu',kernel_initializer=tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)))
    # model.add(Dense(416, activation='relu',kernel_initializer=tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)))
    # model.add(Dense(672, activation='relu',  kernel_initializer=tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)))
    # model.add(AveragePooling1D(pool_size=7, strides=1))
    model.add(Dense(n_classes, kernel_initializer=tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)))
    model.add(Activation('softmax'))
    # Compile model
    numpy.random.seed(seed)
    # set_random_seed(seed)
    tensorflow.random.set_seed(seed)
    #    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())
    return model


# from keras import backend as K
# K.set_image_dim_ordering('th')


# X, y = make_gaussian_quantiles(n_samples=13000, n_features=10,
#                               n_classes=3, random_state=1)
def dataset(n_features=6, n_classes=8):
    # generat randon synethetic data
    # from sklearn.datasets import make_gaussian_quantiles
    #########################plot the hist and pie
    import matplotlib as mpl
    #     def plot_hist(y_test, oName0):
    #
    #         mpl.rc('font', family = 'Times New Roman')
    #     #        (n, bins, patches)=plt.hist(y_train, bins=[0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25])
    #         (n, bins, patches)=plt.hist(y_test+1, bins=[0.75, 1.25, 1.75, 2.25, 2.75, 3.25])#, 3.75, 4.25, 4.75, 5.25])
    #
    #         #    (n, bins, patches)=plt.hist(y_train+1, bins=[0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25])
    #         plt.xlabel('Class')
    #         plt.ylabel('# Samples')
    #         oName = oName0+ '_hist.png'
    #         plt.savefig(oName,dpi=200)
    #         plt.show()
    #         print_t = 'The Histogram of the data is saved as: ' + oName
    #         print(print_t)
    #         print (n)
    #         print(n/len(y_test))
    #
    #        ########################Basci pie chart:
    #         labels = 'C1', 'C2', 'C3'#, 'C4', 'C5'
    #         sizes = [v for i, v in enumerate(n) if (i%2)==0]
    #     #      sizes = [15, 30, 45, 10]
    #         explode = (0, 0, 0.1)#, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
    #
    #         fig1, ax1 = plt.subplots()
    #         ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
    #                     shadow=True, startangle=90)
    #         ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    #         oName = oName0 +'_pie.png'
    #         plt.savefig(oName,dpi=200)
    #
    #         plt.show()
    #         print_t = 'The Histogram of the data is saved as: ' + oName
    #         print(print_t)
    #
    #     #############################################
    #     #epochs=6
    #
    #     X, y = make_gaussian_quantiles(n_samples=13000, n_features=n_features,
    #                                    n_classes=n_classes, random_state=1)
    #
    #     n_split = 3000
    #
    #     X_train, X_test = X[:n_split], X[n_split:]
    #     y_train, y_test = y[:n_split], y[n_split:]
    #
    #     #    df=pd.DataFrame({'a':y_train})
    #     N_re=[200, 500]
    # #    N_re=[200, 0]
    #
    #
    #     a=[index for index, v in enumerate(y_train) if v == 0]
    #     y_train=numpy.delete(y_train, a[0:N_re[0]])
    #     X_train=numpy.delete(X_train, a[0:N_re[0]], axis=0)
    #     a=[index for index, v in enumerate(y_train) if v == 1]
    #     y_train=numpy.delete(y_train, a[0:N_re[1]])
    #     X_train=numpy.delete(X_train, a[0:N_re[1]], axis=0)
    #     ########plot hist and pie
    #     plot_hist(y_train, oName0 = 'synethetic_train')
    #
    #     plot_hist(y_test, oName0 = 'synethetic_test')
    ###################
    dataset = pd.read_csv('feature8(160).csv')

    def process(data):
        columns = data.columns
        label = data[columns[6]]
        label = label.tolist()  # 转换成列表 https://vimsky.com/examples/usage/python-pandas-series-tolist.html
        data = data.values.tolist()
        X, Y = [], []
        for i in range(len(data)):
            train_seq = []
            train_label = []
            # for j in range(i, i + 24):  #24小时
            #     train_seq.append(wind[j])

            for c in range(0, 6):
                train_seq.append(data[i][c])
            train_label.append(label[i])
            X.append(train_seq)
            Y.append(train_label)

        X, Y = numpy.array(X), numpy.array(Y)

        length = int(len(X))
        X, Y = X[:length], Y[:length]
        return X, Y

    x, y = process(dataset)
    y = numpy.ravel(y)
    # numpy.random.seed(20)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False, random_state=1)
    t = MinMaxScaler()
    t.fit(X_train)
    X_train = t.transform(X_train)
    X_test = t.transform(X_test)
    return X_train, y_train, X_test, y_test


# def reshape_for_DNN(X):
#     ###########reshape input mak it to be compatibel to CNN
#     newshape = X.shape
#     newshape = list(newshape)
#     newshape.append(1)
#     newshape = tuple(newshape)
#     X_r = numpy.reshape(X, newshape)  # reshat the trainig data to (2300, 10, 1) for CNN
#
#     return X_r

def onehot_for_DNN(y):
    lb = OneHotEncoder(sparse=False)
    y_b = y.reshape(len(y), 1)
    y_b = lb.fit_transform(y_b)

    return y_b


n_features = 6
n_classes = 8
# n_classes=2


X_train, y_train, X_test, y_test = dataset(n_features=n_features, n_classes=n_classes)
batch_size = 10
# X_train_r, X_test_r = reshape_for_DNN()
# X_train_r = reshape_for_DNN(X_train)
# X_test_r = reshape_for_DNN(X_test)
# X_train_r = reshape_for_DNN(X_train)
# X_test_r = reshape_for_DNN(X_test)
y_test_b = onehot_for_DNN(y_test)
y_train_b = onehot_for_DNN(y_train)

###########################################Adaboost+CNN:

# from multi_adaboost_CNN import AdaBoostClassifier as Ada_CNN
#
# n_estimators = 2
# epochs = 6
# bdt_real_test_CNN = Ada_CNN(
#     base_estimator=baseline_model(n_features=n_features, n_classes=n_classes),
#     n_estimators=n_estimators,
#     learning_rate=1,
#     epochs=epochs)
# #######discreat:
#
# bdt_real_test_CNN.fit(X_train_r, y_train, batch_size)
# test_real_errors_CNN = bdt_real_test_CNN.estimator_errors_[:]
#
# y_pred_CNN = bdt_real_test_CNN.predict(X_train_r)
# print('\n Training performance of bdt_real_test_CNN (AdaBoost+CNN): {}'.format(
#     accuracy_score(bdt_real_test_CNN.predict(X_train_r), y_train)))
#
# start_time = time.time()
# y_pred_CNN = bdt_real_test_CNN.predict(X_test_r)
# end_time = time.time()
# testing_time = end_time - start_time
# print('\n Testing performance of bdt_real_test_CNN (AdaBoost+CNN): {}'.format(
#     (accuracy_score(bdt_real_test_CNN.predict(X_test_r), y_test))))
# print("测试时间为: ", testing_time, "秒")
##########################################single CNN:

# train_DNN(X_train=X_train, y_train=y_train, epochs=20,
#           batch_size=batch_size, X_test=X_test, y_test=y_test,
#           n_features=n_features, seed=seed)
model = baseline_model(n_features=6, seed=100, n_classes=8)
model.fit(X_train, y_train_b, epochs=20, batch_size=batch_size)
y_pred = model.predict(X_test)
y_pred_test = numpy.argmax(y_pred, axis=1)

# def plot_confusion_matrix(cm, labels_name, is_norm=True,  colorbar=True, cmap=plt.cm.Blues):
#     if is_norm==True:
#         cm = numpy.around(cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis],2)  # 横轴归一化并保留2位小数
#
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)  # 在特定的窗口上显示图像
#     for i in range(len(cm)):
#         for j in range(len(cm)):
#             plt.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center') # 默认所有值均为黑色
#             # plt.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', color="white" if i==j else "black", verticalalignment='center') # 将对角线值设为白色
#
#     num_local = numpy.array(range(len(labels_name)))
#     plt.xticks(num_local, labels_name)  # Set labels on x-axis with fontsize
#     plt.yticks(num_local, labels_name)  # Set labels on y-axis with fontsize
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#
#     if is_norm==True:
#         plt.savefig(r'.\cm_norm_' + '.png', format='png')
#     else:
#         plt.savefig(r'.\cm_' + '.png', format='png')
#     plt.show() # plt.show()在plt.savefig()之后
#     plt.close()
#
#
#
# y_true = y_test # 真实标签
# y_pred = y_pred_test
# label_name = ['ds2', 'ds3', 'ds4', 'ds6']
# cm = confusion_matrix(y_true, y_pred) # 调用库函数confusion_matrix
# plot_confusion_matrix(cm, label_name, is_norm=False) # 调用上面编写的自定义函数
# plot_confusion_matrix(cm, label_name, is_norm=True) # 经过归一化的混淆矩阵

def plot_confusion_matrix(cm, labels_name, is_norm=True, cmap=plt.cm.Blues):
    if is_norm:
        cm = numpy.around(cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis], 2)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

    plt.xticks(numpy.arange(len(labels_name)), labels_name)
    plt.yticks(numpy.arange(len(labels_name)), labels_name)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()
    plt.close()

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
plot_confusion_matrix(cm, ['class{}'.format(i + 1) for i in range(n_classes)], is_norm=False)
plot_confusion_matrix(cm, ['class{}'.format(i + 1) for i in range(n_classes)], is_norm=True)
