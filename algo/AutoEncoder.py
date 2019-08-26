from .base import Base
import tensorflow as tf

class AutoEncoder(Base):
    def __init__(self,hidden_neurons=None,epoch=20,dropout_rate=0.2,contamination=0.1,regularizer_weight=0.1):
        self.hidden_neurons=hidden_neurons
        self.epoch=epoch
        self.dropout_rate=dropout_rate
        self.contamination=contamination
        self.regularizer_weight=regularizer_weight

        if self.hidden_neurons is None:
            self.hidden_neurons = [64, 32, 32, 64]

        if not self.hidden_neurons == self.hidden_neurons[::-1]:
            print(self.hidden_neurons)
            raise ValueError("Hidden units should be symmetric")

    def _build_model(self):
        model = Sequential()
        # Input layer
        model.add(Dense(
            self.hidden_neurons_[0], activation=self.hidden_activation,
            input_shape=(self.n_features_,),
            activity_regularizer=l2(self.l2_regularizer)))
        model.add(Dropout(self.dropout_rate))

        # Additional layers
        for i, hidden_neurons in enumerate(self.hidden_neurons_, 1):
            model.add(Dense(
                hidden_neurons,
                activation=self.hidden_activation,
                activity_regularizer=l2(self.l2_regularizer)))
            model.add(Dropout(self.dropout_rate))

        # Output layers
        model.add(Dense(self.n_features_, activation=self.output_activation,
                        activity_regularizer=l2(self.l2_regularizer)))

        # Compile model
        model.compile(loss=self.loss, optimizer=self.optimizer)
        if self.verbose >= 1:
            print(model.summary())
        return model
    def fit(self, X, y=None):

    def predict(self, X):
