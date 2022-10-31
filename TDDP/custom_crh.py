import numpy as np
from numpy import ndarray

import logging
import sys
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
# format = logging.Formatter("%(asctime)s-%(name)s-%(levelname)s: %(message)s")
#             "format": "%(asctime)s : %(levelname)s : %(module)s : %(funcName)s : %(lineno)d : (Process Details : (%(process)d, %(processName)s), Thread Details : (%(thread)d, %(threadName)s))\nLog : %(message)s"
# format = logging.Formatter("%(asctime)s : %(levelname)s : %(module)s : %(funcName)s : %(lineno)d : (Process Details : (%(process)d, %(processName)s), Thread Details : (%(thread)d, %(threadName)s)): %(message)s")
format = logging.Formatter("%(asctime)s-%(levelname)s-(%(module)s.%(funcName)s.%(lineno)d)-Thread(%(threadName)s.%(thread)d): %(message)s")
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(format)
LOG.addHandler(ch)



class CustomCrh:

    def __init__(self):
        self._dataset = None
        self._initial_weights = None
        self._weights = None
        self._truth = None
        self._max_iter_steps = 0
        self._end_threshold = 0 # 这个是 真值相减的差， 不是 weights 相减
        self._iter_truth = None
        self._iter_weights = None
        self._NUM_OF_SOURCES = 0
        self._NUM_OF_TASKS = 0
        self.eps = 1e-5 # 防止有的 source/user 的 weight 过小，最后趋近于 0，每轮迭代每个 source 都加上这个值


    def fit(self, dataset: ndarray, initial_weights : ndarray = None, max_iter_steps : int = 20, end_threshold = 0.001):
        if (initial_weights is not None):
            assert initial_weights.ndim == 1
            assert dataset.shape[0] == initial_weights.shape[0]
            self._initial_weights = initial_weights
        assert dataset.ndim == 2
        self._dataset = dataset
        self._NUM_OF_SOURCES = dataset.shape[0]
        self._NUM_OF_TASKS = dataset.shape[1]
        self._end_threshold = end_threshold
        self._max_iter_steps = max_iter_steps
        return self

    def truth_discoverty(self):
        assert self._dataset is not None # cannot use == None
        if self._initial_weights is None:
            self._initial_weights = np.empty(self._NUM_OF_SOURCES)
            self._initial_weights.fill(1/self._NUM_OF_SOURCES)

        # 计算每个任务的标准差
        std = [np.std(col) for col in self._dataset.T]

        self._iter_weights = np.array([self._initial_weights])
        self._iter_truth = np.zeros(self._NUM_OF_TASKS)

        curSteps = 0
        while True:
            # self._iter_weights = np.r_[self._iter_weights, np.dot(self._initial_weights, self._dataset)/np.sum(self._initial_weights)]
            # cur = np.dot(self._initial_weights, self._dataset)/np.sum(self._initial_weights)

            # 计算每个任务的真值
            cur_truth = np.dot(self._iter_weights[-1], self._dataset)/np.sum(self._iter_weights[-1])
            self._iter_truth = np.vstack([self._iter_truth, cur_truth])
            LOG.info("truth=%s" % cur_truth)

            distance = [np.sum((row - cur_truth)**2) for row in self._dataset]
            LOG.info("distance=%s" % distance)
            cur_weights = np.log(np.sum(distance) / distance) + self.eps
            LOG.info("weight=%s" % cur_weights)
            self._iter_weights = np.vstack([self._iter_weights, cur_weights])


            curSteps += 1
            if curSteps >= self._max_iter_steps:
                break






if __name__ == '__main__':
    dataset = np.array([[1, 2, 3], [4, 5, 10]])
    initial_weights = np.array([0.5, 0.5])
    crh = CustomCrh()
    crh.fit(dataset, initial_weights)
    crh.truth_discoverty()
    LOG.info(crh._iter_truth)
    LOG.info(crh._iter_weights)
