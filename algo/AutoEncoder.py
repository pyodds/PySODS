from .base import Base
import tensorflow as tf
from tensorflow.keras import layers,losses
import numpy as np
from sklearn import preprocessing
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
class AutoEncoder(Base):
    def __init__(self,hidden_neurons=None,epoch=100,dropout_rate=0.2,contamination=0.1,regularizer_weight=0.1,activation='relu',kernel_regularizer=0.01,loss_function='mse',optimizer='adam'):
        self.hidden_neurons=hidden_neurons
        self.epoch=epoch
        self.dropout_rate=dropout_rate
        self.contamination=contamination
        self.regularizer_weight=regularizer_weight
        self.activation=activation
        self.kernel_regularizer=kernel_regularizer
        self.loss_function=loss_function
        self.optimizer=optimizer

        if self.hidden_neurons and  self.hidden_neurons != self.hidden_neurons[::-1]:
            print(self.hidden_neurons)
            raise ValueError("Hidden units should be symmetric")

    def _build_model(self):
        model =  tf.keras.Sequential()
        for neuron_num in self.hidden_neurons:
            model.add(layers.Dense(neuron_num,activation=self.activation,kernel_regularizer=tf.keras.regularizers.l1(self.kernel_regularizer)))
            model.add(layers.Dropout(self.dropout_rate))
        model.compile(loss=self.loss_function,optimizer=self.optimizer)
        return model

    def fit(self, X):
        scaler = preprocessing.RobustScaler().fit(X)
        X_train = scaler.transform(X)
        # self.L = np.random.randn(X.shape[0],X.shape[1])
        # self.S = np.random.randn(X.shape[0],X.shape[1])

        # np.random.shuffle(X_train)
        if self.hidden_neurons is None:
            self.hidden_neurons=[X_train.shape[1]//2+1,X_train.shape[1]//4+1,X_train.shape[1]//4+1,X_train.shape[1]//2+1]
        self.batch_size=X_train.shape[0]//10
        self.model=self._build_model()
        # for it in range(1):
        #     self.L = X_train - self.S

        # self.model.fit(self.L,self.L,epochs=self.epoch,batch_size=self.batch_size)

        # self.L_reconstruct = losses.MSE(self.L,self.model.predict(self.L))
        # self.L = tf.reduce_mean(tf.square((self.L,self.model.predict(self.L))))
        # self.S= l21shrink(epsilon=100,x=(X_train-self.L))

        self.model.fit(X_train,X_train,epochs=self.epoch,batch_size=self.batch_size)

        return self

    def predict(self, X):
        reconstruct_error= (np.square(self.model.predict(X)-X)).mean(axis=1)
        ranking = np.sort(reconstruct_error)
        threshold = ranking[int((1-self.contamination)*len(ranking))]
        self.threshold = threshold
        mask = (reconstruct_error>=threshold)
        ranking[mask]=-1
        ranking[np.logical_not(mask)]=1
        return ranking

    def decision_function(self,X):
        reconstruct_error= (np.square(self.model.predict(X)-X)).mean(axis=1)
        return reconstruct_error

def l21shrink(epsilon,x):

    output = x.numpy()
    norm = np.linalg.norm(x.numpy(), ord=2, axis=0)
    for i in range(x.numpy().shape[1]):
        if norm[i] > epsilon:
            for j in range(x.shape[0]):
                output[j, i] = x.numpy()[j, i] - epsilon * x.numpy()[j, i] / norm[i]
        else:
            output[:, i] = 0.
    return output