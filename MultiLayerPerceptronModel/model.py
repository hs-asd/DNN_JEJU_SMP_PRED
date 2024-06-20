from settings import *
import torch.nn as nn
import torch.optim
import log
from test import saveFigureDuringTraining
import os
class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation_function):
        super(MultiLayerPerceptron, self).__init__()
        self.dim_input = input_dim
        self.dim_hidden = hidden_dim

        # activation function 설정 가능하도록 구성, 일단은 ReLU로 구성
        if activation_function == 'ReLU':
            self.activation_function = nn.ReLU()
        else:
            self.activation_function = nn.ReLU()

        # Linear1: input layer -> hidden layer
        self.Linear_to_hidden = nn.Linear(input_dim, hidden_dim)
        # Linear2: hidden layer -> output layer(1로 고정, 시간별 단일 SMP 예측)
        self.Linear_to_out = nn.Linear(hidden_dim, 1)

    def forward(self, X):
        out = self.Linear_to_hidden(X)
        out = self.activation_function(out)
        out = self.Linear_to_out(out)

        return out


def train(model, x_train, y_train, x_validation, y_validation,  x_test, y_test, epochs, lr, loss_fn, train_result_path):
    # logger 생성 및 설정
    logger_stream, logger_file = log.setLogger(__name__, file_name=train_result_path + 'log.log')

    # Train Setup log 작성
    log.loggingTrainSetup(logger_stream, logger_file)

    # pandas.DataFrame -> torch.Tensor
    list_dfs = [x_train, y_train, x_validation, y_validation, x_test, y_test]
    list_tensors = [torch.Tensor(df.values) for df in list_dfs]
    x_train_tensor, y_train_tensor, x_validation_tensor, y_validation_tensor, x_test_tensor, y_test_tensor = list_tensors

    # loss function 설정
    if loss_fn == 'L1':
        criterion = nn.L1Loss()
    elif loss_fn == 'MSE':
        criterion = nn.MSELoss()
    else:
        criterion = nn.L1Loss()

    # optimizer 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logger_file.info('Start Training.')
    logger_stream.info('Start Training.')

    for epoch in range(1, epochs + 1):
        model.train()
        out_train = model.forward(x_train_tensor)
        loss_train = criterion(out_train, y_train_tensor)

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_validation = model.forward(x_validation_tensor)
            out_test = model.forward(x_test_tensor)

        loss_validation = criterion(out_validation, y_validation_tensor)
        loss_test = criterion(out_test, y_test_tensor)

        if epoch % 500 == 0:
            # 출력을 위한 MAPE 계산
            ary_real_validation, ary_pred_validation = np.array(y_validation_tensor), np.array(out_validation)
            ary_real_test, ary_pred_test = np.array(y_test_tensor), np.array(out_test)

            ary_real, ary_pred = ary_real_validation[ary_real_validation != 0], ary_pred_validation[ary_real_validation != 0]
            MAPE_validation = np.mean(abs((ary_real - ary_pred) / ary_real) * 100)

            ary_real, ary_pred = ary_real_test[ary_real_test != 0], ary_pred_test[ary_real_test != 0]
            MAPE_test = np.mean(abs((ary_real - ary_pred) / ary_real) * 100)

            # Train Metrics log 작성
            log.loggingTrainMetrics(logger_stream, logger_file, epoch, loss_train, loss_validation, loss_test, MAPE_validation, MAPE_test)

            # Epoch 별 .pt파일 저장
            dir_save = os.path.dirname(train_result_path + 'pt/')
            if not os.path.exists(dir_save):
                os.makedirs(dir_save)
            torch.save(model.state_dict(), train_result_path + 'pt/' + str(epoch) + '.pt')

            saveFigureDuringTraining(model, x_test, y_test, train_result_path, epoch)

