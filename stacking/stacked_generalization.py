from sklearn.model_selection import KFold


class StackedGeneralization:
    def __init__(self, n_folds, train_data, train_target):
        self.n_folds = n_folds
        self.train_data = train_data
        self.train_target = train_target
        self.skf = KFold(n_splits=n_folds)
