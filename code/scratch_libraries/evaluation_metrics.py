import numpy as np

class RegressionMetrics:
    def __init__(self, model):
        self.model = model

    def compute_performance (self, y_test, y_pred):
        mae = (self.mean_absolute_error(y_test, y_pred))
        mape = self.mean_absolute_percentage_error(y_test, y_pred)
        mpe = self.mean_percentage_error(y_test, y_pred)
        mse = self.mean_squared_error(y_test, y_pred)
        rmse = self.root_mean_squared_error(y_test, y_pred)
        r2 = self.r_2(y_test, y_pred)
        return {'mae':mae,'mape': mape,'mpe': mpe,
                'mse': mse,'rmse': rmse,'r2': r2}

    def mean_absolute_error (self, y_test, y_pred):
        output_errors= np.abs(y_pred - y_test)
        return np.average(output_errors)

    def mean_squared_error (self, y_test, y_pred):
        output_errors= (y_pred - y_test)**2
        return np.average(output_errors)

    def root_mean_squared_error (self, y_test, y_pred):
        return np.sqrt(self.mean_squared_error(y_test, y_pred))

    def mean_absolute_percentage_error (self, y_test, y_pred):
        output_errors = np.abs((y_pred - y_test)/y_test)
        return np.average(output_errors)*100

    def mean_percentage_error (self, y_test, y_pred):
        output_errors = (y_pred - y_test)/y_test
        return np.average(output_errors)*100

    def r_2(self, y_test, y_pred):
        sst = np.sum ((y_test - y_test.mean())**2) # total deviation
        ssr = np.sum((y_pred-y_test)**2) # residual devaition
        r2 = 1-(ssr/sst)
        return r2


class ClassificationMetrics:
    def __init__(self, model):
        self.model = model

    def compute_performance(self, y_pred, y_test):
        self.cm = self.confusion_matrix(y_pred, y_test)
        self.tn = self.cm[0, 0]
        self.fp = self.cm[0, 1]
        self.fn = self.cm[1, 0]
        self.tp = self.cm[1, 1]

        return {
            "confusion_matrix": self.cm,
            "accuracy": self.accuracy(),
            "error_rate": self.error_rate(),
            "precision": self.precision(),
            "recall": self.recall(),
            "fn_rate": self.fn_rate(),
            "specificity": self.specificity(),
            "fp_rate": self.fp_rate(),
            "f1_score": self.f1_score(),
        }

    def confusion_matrix(self, y_pred, y_test):
        m = len(y_test)
        tp, tn, fp, fn = 0, 0, 0, 0

        for i in range(m):
            if y_pred[i] == y_test[i]:
                if y_pred[i] == 1:
                    tp += 1
                else:
                    tn += 1
            else:
                if y_pred[i] == 1:
                    fp += 1
                else:
                    fn += 1

        return np.array([[tn, fp], [fn, tp]])

    def accuracy(self):
        return (self.tp + self.tn) / self.cm.sum()

    def error_rate(self):
        return 1 - self.accuracy()

    def precision(self):
        return self.tp / (self.tp + self.fp)

    def recall(self):
        return self.tp / (self.tp + self.fn)

    def fn_rate(self):
        return 1 - self.recall()

    def specificity(self):
        return self.tn / (self.tn + self.fp)

    def fp_rate(self):
        return 1 - self.specificity()

    def f1_score(self):
        precision = self.precision()
        recall = self.recall()

        return 2 * (precision * recall) / (precision + recall)
