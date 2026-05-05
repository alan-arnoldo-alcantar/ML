import numpy as np
from pydantic import PositiveInt, PositiveFloat

class AdalineSGD:
    def __init__(
            self,
            eta: PositiveFloat = 0.1,
            epochs: PositiveInt = 10,
            random_seed: PositiveInt = 1
    ):
        self.eta = eta
        self.epochs = epochs
        self.rgen = np.random.default_rng(seed=random_seed)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float16(0.0)
        self.losses_ = []

        for epoch in range(self.epochs):
            perm = self.rgen.permutation(len(X))
            X, y = X[perm], y[perm]
            losses = []
            for xi, target in zip(X,y):
                losses.append(self._update_weights(xi, target))
            self.losses_.append(np.mean(losses))
        return self


    def _net_input(self, X: np.ndarray) -> np.ndarray | np.float16:
        return np.dot(X, self.w_) + self.b_
    
    def _activation(self, X: np.ndarray | np.float16) -> np.ndarray | np.float16:
        return X
    
    def predict(self, X: np.ndarray) -> np.ndarray | np.float16:
        return np.where(self._activation(self._net_input(X)) >= 0.5, 1., 0.)
    
    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        for xi, target in zip(X, y):
            self._update_weights(xi, target)
        return self
    
    def _update_weights(self, xi: np.ndarray, target: int):
        output = self._activation(self._net_input(xi))
        error = target - output
        self.w_ += self.eta * 2.0 * error * xi
        self.b_ += self.eta * 2.0 * error
        return error**2