import pandas as pd

from settings import *
import torch.nn as nn
import torch.optim
import log
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

        # Linear1(Linear_to_hidden): input layer -> hidden layer
        self.Linear_to_hidden = nn.Linear(input_dim, hidden_dim)
        # Linear2(Linear_to_out): hidden layer -> output layer(1로 고정, 시간별 단일 SMP 예측)
        self.Linear_to_out = nn.Linear(hidden_dim, 1)

    def forward(self, X):
        if isinstance(X, pd.DataFrame):
            X = torch.Tensor(X.values)
        else:
            X = torch.Tensor(X)

        out = self.Linear_to_hidden(X)
        out = self.activation_function(out)
        out = self.Linear_to_out(out)

        return out


def runTraining(model, x_train, y_train, x_validation, y_validation,  x_test, y_test, epochs, lr, loss_fn, result_dir_name):
    # Set path to save training result
    path_result_dir = 'train/' + result_dir_name + '/'
    # Set logger
    logger_stream, logger_file = log.setLogger(__name__, file_name=path_result_dir + 'log.log')
    # Log Train Setup
    log.loggingTrainingSetup(logger_stream, logger_file)
    # Make directory to save model to a state dict file(.pt)
    os.makedirs(path_result_dir + 'pt')

    # Set loss function
    if loss_fn == 'L1':
        criterion = nn.L1Loss()
    elif loss_fn == 'MSE':
        criterion = nn.MSELoss()
    else:
        criterion = nn.L1Loss()

    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logger_file.info('Start Training.')
    logger_stream.info('Start Training.')

    for epoch in range(1, epochs + 1):
        # Switch to training mode for model training
        model.train()

        out_train = model.forward(x_train)
        loss_train = criterion(out_train, torch.Tensor(y_train.values))

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        # 일정 Epoch 마다 오차 출력, state dict file 저장 등을 위한 조건문
        if epoch % 500 == 0:
            # Switch to evaluation mode for model evaluation
            model.eval()

            # Compute the loss function value for validation, test data
            with torch.no_grad():
                loss_validation = criterion(model.forward(x_validation), torch.Tensor(y_validation.values))
                loss_test = criterion(model.forward(x_test), torch.Tensor(y_test.values))

            # Calculate statistics for logging
            MAPE_validation = calcMAPE(model, x_validation, torch.Tensor(y_validation.values))
            MAPE_test = calcMAPE(model, x_test, torch.Tensor(y_test.values))

            # Log train metrics
            log.loggingTrainingMetrics(logger_stream, logger_file, epoch, loss_train, loss_validation, loss_test, MAPE_validation, MAPE_test)

            # Save the model to a state dict file(.pt) at certain epoch intervals
            torch.save(model.state_dict(), path_result_dir + 'pt/' + str(epoch) + '.pt')

def calcMAPE(model, x, y):
    model.eval()
    with torch.no_grad():
        out = model.forward(x)

    # 실제 값이 0인 값의 경우 MAPE 연산이 불가능하기 때문에 제외 후 연산 수행
    out_filtered = out[y != 0]
    y_filtered = y[y != 0]

    return torch.mean(torch.abs((out_filtered - y_filtered) / y_filtered) * 100)
