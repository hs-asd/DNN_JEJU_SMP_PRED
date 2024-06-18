import torch.nn as nn
import torch.optim
import numpy as np
from settings import settings

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

    print('start train')

    for epoch in range(1, epochs + 1):
        print(epoch)
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

        if epoch % 500 == 0:
            print('epoch: ', epoch, '/', epochs, '\n',
                  f'Train loss: {loss_train.item(): 0.3f}', f'Validation loss: {loss_validation.item(): 0.3f}', '\n',
                  f'validation MAPE: {MAPE_validation: 0.3f}', f'test MAPE: {MAPE_test: 0.3f}')
            torch.save(model.state_dict(), path_pt_files + str(epoch) + '.pt')

    with open(path_pt_files + 'hyperparameters.txt', 'w') as f:
        for key, value in settings.items():
            f.write(f'{key}: {value}\n')
