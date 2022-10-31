import numpy as np


class SyntheticDataset:

    @staticmethod
    def generate_dataset(num_of_sources, num_of_tasks):
        return np.random.randint(10, 100, size=(num_of_sources, num_of_tasks))


    @staticmethod
    def generate_init_weights(num_of_sources):
        return np.random.random(num_of_sources)

