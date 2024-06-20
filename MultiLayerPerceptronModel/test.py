from settings import *
import matplotlib.pyplot as plt
import os

class TestBed:
    def __init__(self, model, x, y):
        self.MAPE = None
        self.model = model
        self.x = x
        self.y_real = np.array(y)
        self.y_pred = None
        self.ary_datetime = x.index.to_numpy()

    def predict(self):
        x = torch.Tensor(self.x.values)
        with torch.no_grad():
            self.y_pred = np.array(self.model.forward(x))

    def calcStatistics(self):
        # Calculate MAPE(Mean Absolute Percentage Error)
        # 0인 값을 무시하기 위한 부분(real 값이 0이면 연산 불가능)
        real, pred = self.y_real[self.y_real != 0], self.y_pred[self.y_real != 0]

        self.MAPE = np.mean(abs((real - pred) / real) * 100)

    def showFigureComparisonRealPred(self):
        plt.figure(figsize=(25, 5))
        plt.plot(self.ary_datetime, self.y_real, label='real')
        plt.plot(self.ary_datetime, self.y_pred, label='pred')
        plt.xlim(self.ary_datetime[0], self.ary_datetime[-1])
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def saveFigureComparisonRealPred(self, path):
        plt.ioff()
        plt.figure(figsize=(25, 5))
        plt.plot(self.ary_datetime, self.y_real, label='real')
        plt.plot(self.ary_datetime, self.y_pred, label='pred')
        plt.xlim(self.ary_datetime[0], self.ary_datetime[-1])
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()


def runTest(model, x, y, pt_path):
    model.load_state_dict(torch.load(pt_path))
    model.eval()

    test_bed = TestBed(model, x, y)
    test_bed.predict()
    test_bed.calcStatistics()
    print('MAPE: ', test_bed.MAPE)
    test_bed.showFigureComparisonRealPred()

def saveFigureDuringTraining(model, x, y, train_result_path, epoch):
    path_figure = train_result_path + 'figure/' + str(epoch) + '.png'
    dir_save = os.path.dirname(path_figure)
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)

    model.load_state_dict(torch.load(train_result_path + 'pt/' + str(epoch) + '.pt'))
    model.eval()
    test_bed = TestBed(model, x, y)
    test_bed.predict()
    test_bed.saveFigureComparisonRealPred(path_figure)
