import torch.nn as nn
import torch.optim
import numpy as np

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

def train(model, x_train, y_train, x_validation, y_validation, epochs, lr, loss_fn):
    # loss function 설정
    if loss_fn == 'L1':
        criterion = nn.L1Loss()
    else:
        criterion = nn.L1Loss()

    # optimizer 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # epoch 마다 loss 값 저장을 하기 위한 변수 선언
    ary_losses_train = np.zeros(epochs)
    ary_losses_validation = np.zeros(epochs)

    print('start train')

    for epoch in range(1, epochs+1):
        out_train = model.forward(x_train)
        with torch.no_grad():
            out_validation = model.forward(x_validation)

        loss_train = criterion(out_train, y_train)
        loss_train.backward()
        optimizer.step()

        loss_validation = criterion(out_validation, y_validation)

        ary_losses_train[epoch-1] = loss_train.item()
        ary_losses_validation[epoch-1] = loss_validation.item()

        if epoch % 500 == 0:
            print('epoch: ', epoch, '/', epochs, f'Train loss: {loss_train.item(): 0.3f}', f'Validation loss: {loss_validation.item(): 0.3f}')

class TestBed:
    def __init__(self):
        pass
