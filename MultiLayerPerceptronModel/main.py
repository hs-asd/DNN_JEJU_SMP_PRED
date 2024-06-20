from settings import *                      # (tmp)hyperparameters, static variables
from model import *                         # MLP based model
from data import DataLoader                 # data loading module
from data import DataPreprocessor           # data preprocessing module
from test import *


if __name__ == '__main__':
    # 엑셀 파일에서 데이터를 불러오기 위해 DataLoader 인스턴스 선언 및 pandas.DataFrame 데이터 생성
    data_loader = DataLoader(cache_path=CACHE_FILE)

    # 데이터 전처리를 위해 DataPreprocessor 인스턴스 선언
    data_preprocessor = DataPreprocessor(data_loader.df_x_data, data_loader.df_y_data)

    # 데이터 scaling
    data_preprocessor.scaleData(SCALING_MAP)

    # Train data, Validation data, Test data 생성
    data_preprocessor.splitData(RANGE_TRAIN, RANGE_VALIDATION, RANGE_TEST)

    # 학습 데이터의 랜덤 셔플링
    if RANDOM_SHUFFLE:
        data_preprocessor.df_x_train, data_preprocessor.df_y_train = sklearn.utils.shuffle(data_preprocessor.df_x_train, data_preprocessor.df_y_train, random_state=0)
    else:
        pass

    # 학습 데이터 선언
    x_train, x_validation, x_test, y_train, y_validation, y_test = data_preprocessor.getData()

    # MLP model 선언
    MLP = MultiLayerPerceptron(input_dim=DIM_INPUT, hidden_dim=DIM_HIDDEN, activation_function=ACTIVATION_FUNCTION)

    # Training
    train(MLP, x_train, y_train, x_validation, y_validation, x_test, y_test, EPOCHS, LEARNING_RATE, LOSS_FUNCTION, 'train/non_shuffle_L1/')

    # Test
    #runTest(MLP, x_test, y_test, 'pt/shuffle_L1/7500.pt')
