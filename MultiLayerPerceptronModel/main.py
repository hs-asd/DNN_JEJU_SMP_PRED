from model import *                         # MLP based model
from tmp_script_hyperparameters import *    # (tmp)hyperparameters, static variables
from data import DataLoader                 # data loading module
from data import DataPreprocessor           # data preprocessing module


if __name__ == '__main__':
    # 엑셀 파일에서 데이터를 불러오기 위해 DataLoader 인스턴스 선언 및 pandas.DataFrame 데이터 생성
    data_loader = DataLoader(load_from_cache=True, cache_path='pkl/data.pkl')

    # X, y 데이터 반환
    X, y = data_loader.df_x_data, data_loader.df_y_data

    # 데이터 전처리를 위해 DataPreprocessor 인스턴스 선언
    data_preprocessor = DataPreprocessor(X, y)

    # 데이터 scaling
    data_preprocessor.scaleData(SCALING_MAP)

    # Train data, Validation data, Test data 생성
    data_preprocessor.splitData(RANGE_TRAIN, RANGE_VALIDATION, RANGE_TEST)

    # pandas.DataFrame -> torch.Tensor 변환
    data_preprocessor.dataframeToTensor()

    # 학습 데이터 선언
    x_train, x_validation, x_test, y_train, y_validation, y_test = data_preprocessor.getData()

    # MLP model 선언
    MLP = MultiLayerPerceptron(input_dim=DIM_INPUT, hidden_dim=DIM_HIDDEN, activation_function=ACTIVATION_FUNCTION)

    # train
    # train(MLP, x_train, y_train, x_validation, y_validation, EPOCHS, LEARNING_RATE, LOSS_FUNCTION, path_pt)


