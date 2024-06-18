import numpy as np
import torch
import matplotlib.pyplot as plt
import datetime

class TestBed:
    def __init__(self, model, x, y_real):
        self.MAPE = None
        self.model = model
        self.x = x
        self.y_real = y_real
        self.y_pred = None
        self.ary_datetime = x.index.to_numpy()

    def predict(self):
        with torch.no_grad():
            x = torch.Tensor(self.x.values)
            self.y_pred = np.array(self.model.forward(x))

    def calcStatistics(self):
        y_real = np.array(self.y_real)
        # Calculate MAPE(Mean Absolute Percentage Error)

        # 0이하의 값을 무시하기 위한 부분
        y_real = y_real[y_real > 0]
        # real 값의 0인 값이 연산에 어려움이 있는 것이기 때문이기에 real 값이 0인 부분을 제거
        y_pred = self.y_pred[y_real > 0]

        self.MAPE = np.mean(abs((y_real - y_pred) / y_real) * 100)

    def plot(self):
        plt.figure(figsize=(25, 5))
        plt.plot(self.ary_datetime, self.y_real, label='real')
        plt.plot(self.ary_datetime, self.y_pred, label='pred')
        plt.xlim(self.ary_datetime[0], self.ary_datetime[-1])
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def runTest(model, x_test, y_test, pt_path):
    model.load_state_dict(torch.load(pt_path))
    test_bed = TestBed(model, x_test, y_test)

    test_bed.predict()
    test_bed.calcStatistics()
    print('MAPE: ', test_bed.MAPE)
    test_bed.plot()

