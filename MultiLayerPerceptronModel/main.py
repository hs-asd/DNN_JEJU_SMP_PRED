import model                                # MLP based model
from tmp_script_hyperparameters import *    # (tmp)hyperparameters, static variables
from data import DataLoader                 # data loading module
from data import DataPreprocessor          # data preprocessing module


if __name__ == '__main__':
    # 엑셀 파일에서 데이터를 불러오기 위해 인스턴스 DataLoader 선언
    data_loader = DataLoader('data/config.ini')

    # data/ 디렉토리의 config.ini 설정에 따라 pd.DataFrame 형태의 데이터 생성을 위한 DataSet.importData() 메서드 실행
    X, y = data_loader.importData()

    # 데이터 전처리를 위해 인스턴스 DataPreprocessor 선언
    data_preprocessor = DataPreprocessor(X, y)

    # 데이터 scaling
    data_preprocessor.scaleData(SCALING_MAP)

    # Train data, Validation data, Test data 생성
    x_train, x_val, x_test = data_preprocessor.splitData(RANGE_TRAIN, RANGE_VALIDATION, RANGE_TEST)

    # Todo torch.Tensor 데이터 변환 구조 생각(아마 DataPreprocessor에 메서드 정의해서 할듯)

    # MLP model 선언
    MLP = model.MultiLayerPerceptron(input_dim=DIM_INPUT, hidden_dim=DIM_HIDDEN, lr=LEARNING_RATE, activation_function=ACTIVATION_FUNCTION)

    # Todo train 구조 만들기(early stopping 구현 고려해야함, 구조만 구현 가능하게 고려하고 방식은 논문 서칭 후 구현)

    