import numpy as np
from numpy import ndarray


class CustomCrh:

    def __init__(self):
        self._dataset = None
        self._initial_weights = None
        self.weights = None
        self.truth = None
        self.max_iterations = 20
        self.end_criteria = 0.001 # 这个是 真值相减 还是 weights 相减，应该是真值吧


    def fit(self, dataset: ndarray, initial_weights : ndarray = None):
        if (initial_weights is not None):
            assert initial_weights.ndim == 1
            assert dataset.shape[0] == initial_weights.shape[0]
            self._initial_weights = initial_weights
        assert dataset.ndim == 2
        self._dataset = dataset
        return self

    def truth_discoverty(self):
        assert self._dataset is not None # cannot use == None
        if self._initial_weights is None:
            self._initial_weights = np.empty(dataset.shape[0])
            self._initial_weights.fill(1/dataset.shape[0])

        for





if __name__ == '__main__':
    dataset = np.array([[1, 2, 3], [4, 5, 6]])
    initial_weights = np.array([0.5, 0.4])
    crh = CustomCrh()
    crh.fit(dataset)
    crh.truth_discoverty()
    print(crh._initial_weights)