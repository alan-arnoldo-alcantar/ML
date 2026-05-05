import numpy as np
from pydantic import PositiveInt, PositiveFloat
import matplotlib.pyplot as plt

class Perceptron:
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
        self.errors_ = []

        for epoch in range(1,self.epochs+1):
            error = 0
            for xi, label in zip(X,y):
                update = self.eta*(label - self.predict(xi))
                self.w_ += update*xi
                self.b_ += update
                error += int(update != 0.0)
            self.errors_.append(error)
        return self
    
    def net_input(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.w_) + self.b_
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.net_input(X) > 0.0, 1, 0)

    
if __name__ == "__main__":
    # Loading data
    data, target = load_iris(return_X_y=True, as_frame=True)
    X = data.loc[0:99,['sepal length (cm)','petal length (cm)']].values
    y = target[:100].values
    
    # Plotting clases
    plt.scatter(x=X[:50,0], y=X[:50,1], c='blue', marker='o', label='Setosa')
    plt.scatter(x=X[50:100,0], y=X[50:100,1], c='red', marker='s', label='Versicolor')
    plt.xlabel("Sepal length (cm)")
    plt.ylabel("Petal length (cm)")
    plt.legend(loc='upper left')
    plt.show()

    # Training and erros vs epochs
    my_perceptron = Perceptron(eta=0.1, epochs=10)
    my_perceptron.fit(X, y)
    plt.plot(range(1, len(my_perceptron.errors_)+1), my_perceptron.errors_, marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Clasiffication errors")
    plt.show()
