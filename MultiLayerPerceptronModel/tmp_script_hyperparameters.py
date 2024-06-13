"""
<24.06.13>
    해당 스크립트에 선언되어있는 변수들은 나중에 하이퍼파라미터 최적화를 위해 하이퍼파라미터와 고정값들이 분리되어야 함
    현재는 변수들을 선언하는 용도로 사용
"""

# Statics
LABEL_OUTPUT = 'SMP JEJU'
DIM_OUTPUT = 1
RANGE_TRAIN = ((2015, 1, 1), (2021, 12, 31))
RANGE_VALIDATION = ((2022, 1, 1), (2022, 12, 31))
RANGE_TEST = ((2023, 1, 1), (2023, 12, 31))

# Hyperparameters
LEARNING_RATE = 1e-3
DIM_INPUT = 3
DIM_HIDDEN = 80
ACTIVATION_FUNCTION = 'ReLU'
# Todo SCALING_OPTION 변수명 변경(normalization, regularization과 같이 전처리를 데이터에 따라 다르게 할 것인데 해당 구분을 위한 dict 변수)
SCALING_MAP = {'SMP LAND': 'MinMax', 'BIDFORECAST JEJU': 'MinMax', 'BIDFORECAST LAND': 'MinMax'}
EPOCHS = 1e+5
LOSS_FUNCTION = None


