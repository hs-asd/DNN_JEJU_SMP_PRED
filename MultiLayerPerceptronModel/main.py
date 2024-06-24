from settings import *                      # (tmp)hyperparameters, static variables
from model import *                         # MLP based model
from data import DataLoader                 # data loading module
from data import DataPreprocessor           # data preprocessing module
from test import *


if __name__ == '__main__':
    # DataLoader
    data_loader = DataLoader(cache_path=CACHE_FILE)

    # DataPreprocessor
    data_preprocessor = DataPreprocessor(data_loader.features, data_loader.target)

    # DataFrame
    x_train, x_validation, x_test, y_train, y_validation, y_test = data_preprocessor.getDataFrameDataSet()

    # MLP Model
    MLP = MultiLayerPerceptron(input_dim=DIM_INPUT, hidden_dim=DIM_HIDDEN, activation_function=ACTIVATION_FUNCTION)

    # Training
    # runTraining(MLP, x_train, y_train, x_validation, y_validation, x_test, y_test, EPOCHS, LEARNING_RATE, LOSS_FUNCTION, 'test')

    # Test
    # runTest(MLP, x_test, y_test, 'train/test/pt/2500.pt')
    # saveFigures(MLP, x_test, y_test, 'train/shuffle_L1/')
