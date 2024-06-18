import datetime

import torch.nn as nn
import torch.optim
import numpy as np
from settings import *
import logging


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


def train(model, x_train, y_train, x_validation, y_validation,  x_test, y_test, epochs, lr, loss_fn, path_pt_files):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(path_pt_files + 'log_train.log', mode='a')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    logger.info('Train Settings')
    logger.info('   Cache File:             %s', CACHE_FILE)
    logger.info('   Learning Rate:          %s', LEARNING_RATE)
    logger.info('   Input Dimension:        %s', DIM_INPUT)
    logger.info('   Hidden Dimension:       %s', DIM_HIDDEN)
    logger.info('   Ouput Dimension:        %s', DIM_OUTPUT)
    logger.info('   Activation Function:    %s', ACTIVATION_FUNCTION)
    logger.info('   Scale Map:              %s', SCALING_MAP)
    logger.info('   Epochs:                 %s', EPOCHS)
    logger.info('   Loss Function:          %s', LOSS_FUNCTION)
    logger.info('   Train Range:            %s ~ %s', datetime.datetime(*RANGE_TRAIN[0]), datetime.datetime(*RANGE_TRAIN[-1]))
    logger.info('   Validation Range:       %s ~ %s', datetime.datetime(*RANGE_VALIDATION[0]), datetime.datetime(*RANGE_VALIDATION[-1]))
    logger.info('   Test Range:             %s ~ %s', datetime.datetime(*RANGE_TEST[0]), datetime.datetime(*RANGE_TEST[-1]))

    # pandas.DataFrame -> torch.Tensor
    list_df_datas = [x_train, y_train, x_validation, y_validation,  x_test, y_test]
    list_tensor_datas = [torch.Tensor(df.values) for df in list_df_datas]
    x_train, y_train, x_validation, y_validation, x_test, y_test = list_tensor_datas

    # loss function 설정
    if loss_fn == 'L1':
        criterion = nn.L1Loss()
    elif loss_fn == 'MSE':
        criterion = nn.MSELoss()
    else:
        criterion = nn.L1Loss()

    # optimizer 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # epoch 마다 loss 값 저장을 하기 위한 변수 선언
    ary_losses_train = np.zeros(epochs)
    ary_losses_validation = np.zeros(epochs)
    ary_losses_test = np.zeros(epochs)

    logger.info('Start Train.')

    for epoch in range(1, epochs + 1):
        logger.debug(epoch)
        out_train = model.forward(x_train)
        with torch.no_grad():
            out_validation = model.forward(x_validation)
            out_test = model.forward(x_test)

        loss_train = criterion(out_train, y_train)
        loss_train.backward()
        optimizer.step()

        loss_validation = criterion(out_validation, y_validation)
        loss_test = criterion(out_test, y_test)

        # 후에 로깅 등을 위해 에폭별로 로스값 기록
        ary_losses_train[epoch - 1] = loss_train.item()
        ary_losses_validation[epoch - 1] = loss_validation.item()
        ary_losses_test[epoch - 1] = loss_test.item()

        # 출력을 위한 MAPE 계산
        ary_real_validation, ary_pred_validation = np.array(y_validation), np.array(out_validation)
        ary_real_test, ary_pred_test = np.array(y_test), np.array(out_test)
        ary_real_test, ary_real_validation = ary_real_test[ary_real_test > 0], ary_real_validation[ary_real_validation > 0]

        MAPE_validation = np.mean(abs((ary_real_validation - ary_pred_validation) / ary_real_validation) * 100)
        MAPE_test = np.mean(abs((ary_real_test - ary_pred_test) / ary_real_test) * 100)

        if epoch % 200 == 0:
            logger.info('Epoch: %s / %s', epoch, epochs)
            logger.info('   Train Loss:         %.2f', loss_train.item())
            logger.info('   Validation Loss:    %.2f', loss_validation.item())
            logger.info('   Validation MAPE:    %.2f', MAPE_validation)
            logger.info('   Test MAPE:          %.2f', MAPE_test)

            torch.save(model.state_dict(), path_pt_files + str(epoch) + '.pt')

