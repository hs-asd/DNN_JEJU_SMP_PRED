import torch.nn as nn


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, lr, activation_function):
        super(MultiLayerPerceptron, self).__init__()
        self.dim_input = input_dim
        self.dim_hidden = hidden_dim
        self.learning_rate = lr

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

def train(model, epochs):
    pass

