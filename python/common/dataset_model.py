class Dataset:
    """
    A class for modelling an abstact dataset, that can be deployed in all experiments.
    """

    def __init__(self, train_X, train_y, test_X, test_y, raw_dataset, feature_names, target_names):
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.raw_dataset = raw_dataset
        self.feature_names = feature_names
        self.target_names = target_names
