import time
import numpy
import tensorflow
from tensorflow.keras import backend as K

# Setting random seed for reproducibility
seed = 50
numpy.random.seed(seed)
tensorflow.random.set_seed(seed)
K.clear_session()

# Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Activation


# Dataset Processing Function
def dataset(n_features=6, n_classes=8):
    dataset = pd.read_csv('feature8(160).csv')

    def process(data):
        columns = data.columns
        label = data[columns[6]].tolist()
        data = data.values.tolist()
        X, Y = [], []
        for i in range(len(data)):
            X.append(data[i][:6])
            Y.append(label[i])
        X, Y = numpy.array(X), numpy.array(Y)
        return X, Y

    X, y = process(dataset)
    y = numpy.ravel(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=1)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, y_train, X_test, y_test


# Model definition function
def baseline_model(n_features=6, seed=100, n_classes=8):
    numpy.random.seed(seed)
    tensorflow.random.set_seed(seed)

    model = Sequential([
        Conv1D(32, 3, padding="same", input_shape=(n_features, 1),
               kernel_initializer=tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)),
        Activation('relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu',
              kernel_initializer=tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)),
        Dropout(0.2),
        Dense(64, activation='relu',
              kernel_initializer=tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)),
        Dense(n_classes, kernel_initializer=tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)),
        Activation('softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


# Training and evaluation function for single CNN
def train_CNN(X_train, y_train, epochs, batch_size, X_test, y_test, n_features=6, seed=100):
    numpy.random.seed(seed)
    tensorflow.random.set_seed(seed)

    model = baseline_model(n_features, seed)
    X_train_r = X_train.reshape(X_train.shape[0], n_features, 1)
    X_test_r = X_test.reshape(X_test.shape[0], n_features, 1)

    encoder = OneHotEncoder(sparse=False)
    y_train_b = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_b = encoder.transform(y_test.reshape(-1, 1))

    model.fit(X_train_r, y_train_b, epochs=epochs, batch_size=batch_size)

    print('\nSingle CNN evaluation on training data:')
    print(model.evaluate(X_train_r, y_train_b))

    start_time = time.time()
    print('\nSingle CNN evaluation on testing data:')
    print(model.evaluate(X_test_r, y_test_b))
    print("Testing time: {:.4f} seconds".format(time.time() - start_time))


# Reshape functions for CNN input
def reshape_for_CNN(X):
    return X.reshape(X.shape[0], X.shape[1], 1)


def onehot_for_CNN(y):
    encoder = OneHotEncoder(sparse=False)
    return encoder.fit_transform(y.reshape(-1, 1))


# Confusion matrix plotting function
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


# Main execution and training
n_features = 6
n_classes = 8
X_train, y_train, X_test, y_test = dataset(n_features, n_classes)
batch_size = 10

# Prepare data for CNN
X_train_r = reshape_for_CNN(X_train)
X_test_r = reshape_for_CNN(X_test)
y_train_b = onehot_for_CNN(y_train)
y_test_b = onehot_for_CNN(y_test)

# Train AdaBoost + CNN
from Improved_AdaBoost_CNN import AdaBoost_CNN

n_estimators = 6
epochs = 60

bdt_real_test_CNN = AdaBoost_CNN(
    base_estimator=baseline_model(n_features=n_features, n_classes=n_classes),
    n_estimators=n_estimators,
    learning_rate=1,
    epochs=epochs)

bdt_real_test_CNN.fit(X_train_r, y_train, batch_size)

start_time = time.time()
y_pred_train = bdt_real_test_CNN.predict(X_train_r)
print('\nTraining accuracy: {:.4f}'.format(accuracy_score(y_train, y_pred_train)))
print("Training time: {:.4f} seconds".format(time.time() - start_time))

start_time = time.time()
y_pred_test = bdt_real_test_CNN.predict(X_test_r)
print('\nTesting accuracy: {:.4f}'.format(accuracy_score(y_test, y_pred_test)))
print("Testing time: {:.4f} seconds".format(time.time() - start_time))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
plot_confusion_matrix(cm, ['class{}'.format(i + 1) for i in range(n_classes)], is_norm=False)
plot_confusion_matrix(cm, ['class{}'.format(i + 1) for i in range(n_classes)], is_norm=True)

# Train single CNN for comparison
train_CNN(X_train, y_train, epochs=100, batch_size=batch_size, X_test=X_test, y_test=y_test, n_features=n_features,
          seed=seed)
