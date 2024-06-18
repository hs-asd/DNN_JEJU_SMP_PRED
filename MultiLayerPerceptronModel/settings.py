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
DIM_INPUT = 4
DIM_HIDDEN = 80
ACTIVATION_FUNCTION = 'ReLU'  # Choose one of the following options: 'ReLU',
SCALING_MAP = {'SMP LAND': 'MinMax', 'BIDFORECAST JEJU': 'MinMax', 'BIDFORECAST LAND': 'MinMax', 'DA SMP LAND': 'MinMax', 'DA SMP JEJU': 'MinMax'}
EPOCHS = int(5e+4)
LOSS_FUNCTION = 'L1'  # Choose one of the following options: 'L1', 'MSE'
CASH_FILE = 'pkl/data_without_smp_land_using_DA_value_to_240617.pkl'

settings = {
    'cache file': CASH_FILE,
    'learning rate': LEARNING_RATE,
    'input dimension': DIM_INPUT,
    'hidden dimension': DIM_HIDDEN,
    'activation function': ACTIVATION_FUNCTION,
    'scaile map': SCALING_MAP,
    'epochs': EPOCHS,
    'loss function': LOSS_FUNCTION
}
