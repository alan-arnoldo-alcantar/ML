import numpy as np
from pydantic import PositiveInt, PositiveFloat

class AdalineGD:
    def __init__(
            self,
            eta: PositiveFloat = 0.01,
            epochs: PositiveInt = 10,
            random_seed: PositiveInt = 1,            
        ):
        self.eta = eta
        self.epochs = epochs
        self.random_seed = random_seed

    def fit(self, X: np.ndarray, y: np.ndarray):
        rgen = np.random.default_rng(seed=self.random_seed)
        self.w_ = rgen.normal(loc=0,scale=0.01,size=X.shape[1])
        self.b_ = np.float16(0.)
        self.losses_ = []

        for epoch in range(1,self.epochs+1):
            net_input = self.net_input(X=X)
            output = self.activation(net_input)
            errors = y - output
            self.w_ += self.eta * 2.0 * np.dot(X.T, errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)
        return self
    
    def net_input(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.w_) + self.b_
    
    def activation(self, X: np.ndarray):
        return X
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)