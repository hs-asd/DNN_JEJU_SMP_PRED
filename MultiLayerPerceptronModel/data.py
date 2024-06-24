import pandas as pd
import sklearn.utils

from settings import *
import log

# Todo 아래 줄의 설정에 대한 설명 주석 추가
pd.options.mode.copy_on_write = True

logger_stream, logger_file = log.setLogger(__name__, file_name='pkl/data_configuration.log')


class DataLoader:
    def __init__(self, config_path='data/config.ini', cache_path='pkl/data.pkl'):
        # self.data_config_list_data_path: 데이터 종류별 디렉토리 경로를 원소로 갖는 list
        self.data_config_list_data_path = None
        self.data_config_start_date = None
        self.data_config_end_date = None
        self.data_config_label_output = None

        # self.config_label: 데이터 종류 구분을 위한 문자열 데이터
        self.config_label = None
        # self.config_list_file_path: 해당 데이터를 읽어올 파일 목록의 경로(문자열)를 값으로 갖는 list
        self.config_list_file_path = None
        # self.config_col_date: 날짜 데이터가 있는 열의 인덱스(zero-based)
        self.config_col_date = None
        # self.config_col_start_data: 목록, 범주, 날짜 부분 등을 제외한 실제 데이터가 시작하는 열의 인덱스(zero-based)
        self.config_col_start_data = None
        # self.config_row_start_data: 목록, 범주, 날짜 부분 등을 제외한 실제 데이터가 시작하는 행의 인덱스(zero-based)
        self.config_row_start_data = None
        # self.config_col_end_data: 최소, 최대, 평균 등을 제외한 실제 데이터가 끝나는 열의 인덱스(zero-based)
        self.config_col_end_data = None
        # self.config_date_dtype_conversion: 날짜 데이터가 문자열로 입력되어있을 경우 datetime 형식으로 변경해야 함을 나타내는 값
        self.config_date_dtype_conversion = False
        # self.config_datetime_foramt: datetime으로 변경할 문자열(혹은 정수) 데이터의 foramt을 나타냄(e.g. '%Y%m%d')
        self.config_datetime_format = None
        # self.config_data_format: 엑셀 파일 상의 데이터 형식을 나타내는 문자열 변수 값은 다음 두 값 중 하나 -> 'TABLE', 'STACK'
        self.config_data_format = None
        # self.config_interval: 데이터의 시간 간격을 나타내는 문자열 변수 값은 다음 값 중 하나 -> 'HOUR', 'DAY'
        self.config_interval = None
        # self.config_day_ahead_conversion: 전일값 사용 플래그(True일 시 전일 값을 입력 값으로 사용)
        self.config_day_ahead_conversion = None

        # features, target으로 구분하기 전 합쳐져있는 DataFrame
        self.data = None
        self.features = None
        self.target = None
        self.full_date_range = None

        self.info_duplicate = pd.DataFrame(columns=['datetime', 'value', 'label'])
        self.info_missing = None

        self.readDataConfig(config_path)
        self.path_cache = cache_path

        try:
            self.loadDataFromCacheFile()
        except FileNotFoundError:
            logger_stream.debug('Cache file not found.')
            logger_stream.debug('Load data from excel files.')
            self.loadDataFromExcelFiles()
            self.saveDataToCacheFile()

    """
    FUNCTION NAME: DataLoader.readConfig

    PARAMETER:
        1. path
            1-0. type: string
            1-1. data 폴더 내에는 각 '데이터별 폴더'에 엑셀 파일이 모아져 있음
            1-2. 1-1의 '데이터별 폴더'의 경로를 나타내는 문자열 값
            1-3. 폴더명 뒤에 '/'까지 입력 해야함

                e.g. 'data/bid forecast jeju/'

    RETURN:
        None

    DESCRIPTION:
        1. 입력 받은 path 경로에 있는 config.ini 파일을 읽는 동작을 수행
        2. config.ini 에서 읽어온 값들은 "self.config*" 와 같은 변수명으로 저장 (*: 임의의 문자열을 의미)

            e.g. self.config_list_file_path

    """
    def readConfig(self, path):
        config = configparser.ConfigParser()
        config.read(path + '/config.ini')

        str_file_list = config['FILES']['list file']
        list_file = str_file_list.split('\n')
        self.config_list_file_path = [path + file for file in list_file]

        self.config_col_date = config['PROPERTIES'].getint('col of date')
        self.config_col_start_data = config['PROPERTIES'].getint('col of start data')
        self.config_row_start_data = config['PROPERTIES'].getint('row of start data')
        self.config_col_end_data = config['PROPERTIES'].getint('col of end data')
        self.config_label = config['PROPERTIES']['data label']
        self.config_date_dtype_conversion = config['PROPERTIES'].getboolean('datetime convert flag')
        self.config_datetime_format = config['PROPERTIES']['datetime format']
        self.config_data_format = config['PROPERTIES']['data format']
        self.config_interval = config['PROPERTIES']['interval']
        self.config_day_ahead_conversion = config['PROPERTIES'].getboolean('day-ahead conversion')

    """
    FUNCTION NAME: DataLoader.readExcelFile

    PARAMETER:
        1. file_path
            1-0. type: string
            1-1. 읽어 올 엑셀 파일의 경로를 나타냄

                e.g. 'data/smp_jeju/smp_jeju_2015.xlsx'

    RETURN:
        1. df
            1-0. type: DataFrame

    DESCRIPTION:
        1. 엑셀 파일의 경로를 나타내는 file_path에 해당하는 엑셀 파일을 읽어와 DataFrame으로 반환하는 함수
        2. 엑셀 파일을 읽을 때 다음과 같은 설정이 고려됨
            2-1. 엑셀 시트 상에서 데이터의 범위(self.config_col_start_data, self.config_row_start_data, self.config_col_end_data)
            2-2. 엑셀 시트 상에서 날짜 열의 위치(self.config_col_date)
            2-3. 날짜열 데이터 형식 변환 필요(self.config_date_dtype_conversion)
            3-4. 날짜열 데이터 형식(self.config_datetime_format)

    """
    def readExcelFile(self, file_path):
        # 엑셀 데이터에서 날짜열과 데이터열만 불러오기 위해(최소, 최대와 같은 통계값 무시 등을 위해) 불러올 열 번호들의 list를 선언
        cols_to_read = [self.config_col_date] + [col for col in range(self.config_col_start_data, self.config_col_end_data + 1)]

        df = pd.read_excel(file_path, usecols=cols_to_read, skiprows=self.config_row_start_data, header=None)
        # 날짜 데이터의 column 이름을 datetime으로 변경
        df.rename(columns={0: 'datetime'}, inplace=True)

        # 날짜데이터가 int, str일 경우 datetime으로 변환
        if self.config_date_dtype_conversion:
            df['datetime'] = pd.to_datetime(df['datetime'], format=self.config_datetime_format)
        else:
            pass

        # 엑셀 데이터가 'TABLE'인 경우 'STACK' 형으로 변경
        if self.config_data_format == 'TABLE':
            df = pd.melt(df, id_vars=df.keys()[0], value_vars=df.keys()[1:])
            df['datetime'] = pd.to_datetime(df['datetime'] + pd.to_timedelta(df['variable'] - 1, unit='h'))
            df.drop(columns='variable', inplace=True)
            df.sort_values('datetime', inplace=True)
        # Todo 'STACK' 데이터인 경우 후에 날짜 데이터 형식보고 처리하기 아마 위의 datetime 변환하는 데에서 처리가 되야할거 같은데
        else:
            pass

        return df

    """
    FUNCTION NAME: DataLoader.importExcelFilesInPath

    PARAMETER:
        None

    RETURN:
        1. df
            1-0. type: DataFrame

    DESCRIPTION:
        1. Data.config_lsit_file_path에 있는 데이터들을 Data.readExcelFile 함수를 통해 DataFrame으로 불러와 합친 후 반환하는 함수

    """
    def importExcelFilesInPath(self):
        df = self.readExcelFile(self.config_list_file_path[0])

        for file_path in self.config_list_file_path[1:]:
            df = pd.concat([df, self.readExcelFile(file_path)], ignore_index=True)

        return df

    """
    FUNCTION NAME: DataLoader.readDataConfig

    PARAMETER:
        1. path_config
            1-0. type: string
            1.1 읽어올 config.ini 파일의 경로를 나타냄
            
                e.g. 'data/config.ini'

    RETURN:
        None

    DESCRIPTION:
        0. 해당 함수는 DataLoader의 instance 선언 과정에서 __init__ 에서 자동으로 실행
        
        1. data 폴더에 있는 config.ini 파일의 정보를 통해 DataLoader 설정에 필요한 값을 받아옴 파일의 정보는 다음과 같음
            1-0. 변수명 <- config.ini 파일 내부 key: 설명
            
            1-1. self.data_config_list_data_path <- data list: dataset에 사용될 데이터 파일의 경로 폴더 명을 나타내는 문자열 변수
                1-1-1. 해당 변수는 \n으로 구분되어 list 형으로 읽어 옴
            
            1-2. year_start <- start year: dataset에 사용할 날짜의 범위를 조절하기 위해 사용하는 변수로 시작 연도를 나타냄
            1-3. month_start <- start month: dataset에 사용할 날짜의 범위를 조절하기 위해 사용하는 변수로 시작 월을 나타냄
            1-4. day_start <- start day: dataset에 사용할 날짜의 범위를 조절하기 위해 사용하는 변수로 시작 일을 나타냄
            
            1-5. year_end <- end year: dataset에 사용할 날짜의 범위를 조절하기 위해 사용하는 변수로 마지막 연도를 나타냄
            1-6. month_end <- end month: dataset에 사용할 날짜의 범위를 조절하기 위해 사용하는 변수로 마지막 월을 나타냄
            1-7. day_end <- end day: dataset에 사용할 날짜의 범위를 조절하기 위해 사용하는 변수로 마지막 일을 나타냄
            
    """
    def readDataConfig(self, path_config):
        config = configparser.ConfigParser()
        config.read(path_config)

        str_data_list = config['DATA LIST']['data list']
        list_data = str_data_list.split('\n')
        self.data_config_list_data_path = ['data/' + data + '/' for data in list_data]

        # 데이터 시작 날짜
        year_start = config['PROPERTIES'].getint('start year')
        month_start = config['PROPERTIES'].getint('start month')
        day_start = config['PROPERTIES'].getint('start day')
        self.data_config_start_date = datetime.datetime(year_start, month_start, day_start)

        # 데이터 마지막 날짜
        year_end = config['PROPERTIES'].getint('end year')
        month_end = config['PROPERTIES'].getint('end month')
        day_end = config['PROPERTIES'].getint('end day')
        self.data_config_end_date = datetime.datetime(year_end, month_end, day_end, 23)

        self.full_date_range = pd.date_range(start=self.data_config_start_date, end=self.data_config_end_date, freq='h')

        # 출력의 label 저장
        self.data_config_label_output = config['PROPERTIES']['output label']

    """
    FUNCTION NAME: DataLoader.loadDataFromExcelFiles

    PARAMETER:
        None

    RETURN:
        1. data
            1-0. type: DateFrame

    DESCRIPTION:
        1. DataLoader.data_config_list_data_path 경로 별(데이터 별)로 DataFrame을 불러온 후 하나의 DataFrame으로 합친 후 instance로 저장 및 반환
            1-1. 데이터 별로 DataFrame으로 불러오면서 중복된 데이터는 제거 후 DataLoader.info_duplicate에 저장
            1-2. 모든 데이터의 DataFrame을 합친 후 결측치가 존재하는 날짜의 데이터를 삭제 후 DataLoader.info_missing에 저장
        
    """
    def loadDataFromExcelFiles(self):
        logger_stream.debug('loadDataFromExcelFiles is executed.')

        list_df = []
        for path in self.data_config_list_data_path:
            self.readConfig(path)
            df_current_path = self.importExcelFilesInPath()
            label = self.config_label

            # 중복 값이 있을 시, 제거 후 정보(datetime, value, label) 반환 (중복의 기준은 같은 날짜의 데이터가 있는 지 여부)
            if df_current_path['datetime'].duplicated().any():
                df_duplicates = df_current_path[df_current_path.duplicated('datetime', keep=False)]
                df_duplicates['label'] = label
                df_current_path.drop_duplicates('datetime', keep='first', inplace=True)
                self.info_duplicate = pd.concat([self.info_duplicate, df_duplicates])
            else:
                pass

            df_current_path.set_index('datetime', inplace=True)
            # 결측치 처리를 위해 datetime 값을 index로 재설정(결측치가 존재하는 날짜의 값은 nan으로 채워짐)
            df_indexed = df_current_path.reindex(self.full_date_range)
            # DataFrame 상에서 데이터 구분 용이를 위해 데이터 label을 column 이름으로 설정
            df_indexed.rename(columns={'value': self.config_label}, inplace=True)

            # 전일 값을 사용하는 데이터이고, 출력 값일 경우 전일의 값을 당일 예측에 사용한다고 보고 두 값을 df_indexed, df_day_ahead로 구분해서 모두 list_df에 추가
            if self.config_day_ahead_conversion and self.data_config_label_output == self.config_label:
                # 시간별 데이터이기 때문에 하루 전 값을 사용하기 위해 24개 데이터를 SHIFT
                df_day_ahead = df_indexed.shift(24)
                df_day_ahead.rename(columns={self.config_label: 'DA ' + self.config_label}, inplace=True)
                list_df.append(df_day_ahead)
                list_df.append(df_indexed)
            # 전일 값을 사용하는 데이터의 경우(self.config_day_ahead_conversion = True) 24개 행을 SHIFT한 df_day_ahead를 list_df에 추가
            elif self.config_day_ahead_conversion and self.data_config_label_output != self.config_label:
                # 시간별 데이터이기 때문에 하루 전 값을 사용하기 위해 24개 데이터를 SHIFT
                df_day_ahead = df_indexed.shift(24)
                df_day_ahead.rename(columns={self.config_label: 'DA ' + self.config_label}, inplace=True)
                list_df.append(df_day_ahead)
            else:
                list_df.append(df_indexed)

        data = pd.concat(list_df, axis=1)
        # 결측지가 존재하는 행을 self.info_missing에 반환
        # DataFrame.any(axis=1): index 별로 True of False 값을 갖는 Series를 반환(해당 index에 해당하는 데이터에 True 값이 하나라도 있을 경우 True 반환)
        self.info_missing = data[data.isna().any(axis=1)]
        # 결측치가 존재하는 행을 drop
        data.dropna(inplace=True)

        # columns: single label or list-like
        features = data.drop(columns=self.data_config_label_output)
        # df로 반환하기 위해 대괄호 두개로 인덱싱
        target = data[[self.data_config_label_output]]

        self.data = data
        self.features = features
        self.target = target

        log.loggingDataConfiguration(logger_stream, logger_file, self.path_cache, features.keys().values,
                                     self.data_config_label_output, self.full_date_range[0], self.full_date_range[-1],
                                     self.data.index[0], self.data.index[-1])

    """
    FUNCTION NAME: DataLoader.loadDataFromCacheFile

    PARAMETER:
        None

    RETURN:
        None

    DESCRIPTION:
        1. DataLoader.path_cache 경로의 cache 파일에서 다음 데이터를 불러옴
            1-1. DataLoader.data
            1-2. DataLoader.features
            1-3. DataLoader.target
            1-4. DataLoader.info_missing
            1-5. DataLoader.info_duplicate
        
    """
    def loadDataFromCacheFile(self):
        logger_stream.debug('Load data from cache file: %s', self.path_cache)
        with open(self.path_cache, 'rb') as f:
            loaded_data = pickle.load(f)

        self.data, self.features, self.target, self.info_missing, self.info_duplicate = loaded_data

    """
    FUNCTION NAME: DataLoader.saveDataToCacheFile

    PARAMETER:
        None

    RETURN:
        None

    DESCRIPTION:
        1. DataLoader.path_cache 경로의 cache 파일에 다음 데이터를 저장함
            1-1. DataLoader.data
            1-2. DataLoader.features
            1-3. DataLoader.target
            1-4. DataLoader.info_missing
            1-5. DataLoader.info_duplicate

    """
    def saveDataToCacheFile(self):
        logger_stream.debug('Save data to cache file: %s', self.path_cache)
        data = [self.data, self.features, self.target, self.info_missing, self.info_duplicate]

        with open(self.path_cache, 'wb') as f:
            pickle.dump(data, f)


class DataPreprocessor:
    def __init__(self, features, target):
        self.features = features
        self.target = target
        self.labels = list(features.keys())

        self.features_train = None
        self.features_validation = None
        self.features_test = None
        self.target_train = None
        self.target_validation = None
        self.target_test = None

        self.scaleData(SCALING_MAP)  # Data Scaling
        self.splitData(RANGE_TRAIN, RANGE_VALIDATION, RANGE_TEST)  # Data Split into set of train, validation, test
        if RANDOM_SHUFFLE:
            self.randomShuffleTrainData()  # Train Data random shuffle
        else:
            pass

    """
    FUNCTION NAME: DataPreprocessor.scaleData

    PARAMETER:
        1. scaling_map: dict 

    RETURN:
        None

    DESCRIPTION:
        1. 
        
    """
    def scaleData(self, scaling_map=None):
        for label in self.labels:
            if scaling_map[label] == 'MinMax':
                scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
                self.features[label] = scaler.fit_transform(self.features[label].values.reshape(-1, 1))
            else:
                pass

    """
    FUNCTION NAME: DataPreprocessor.splitData

    PARAMETER:
        1. train_range: tuple (start, end)
            1-1. 다음과 같은 형식의 2 dim tuple 자료형
            
                e.g. ((2000, 1, 1), (2010, 12, 31))
            
        2. validation_range: tuple (start, end)
            2-1. 1-1과 같음
        3. test_range: tuple (start, end)
            3-1. 1-1과 같음
        
    RETURN:
        None

    DESCRIPTION:
        1. train, validation, test 데이터를 입력 범위에 맞게 생성

    """
    def splitData(self, train_range, validation_range, test_range):
        datetime_start_train = datetime.datetime(*train_range[0])
        datetime_end_train = datetime.datetime(*train_range[-1])
        self.features_train = self.features[datetime_start_train:datetime_end_train]
        self.target_train = self.target[datetime_start_train:datetime_end_train]

        datetime_start_validation = datetime.datetime(*validation_range[0])
        datetime_end_validation = datetime.datetime(*validation_range[-1])
        self.features_validation = self.features[datetime_start_validation:datetime_end_validation]
        self.target_validation = self.target[datetime_start_validation:datetime_end_validation]

        datetime_start_test = datetime.datetime(*test_range[0])
        datetime_end_test = datetime.datetime(*test_range[-1])
        self.features_test = self.features[datetime_start_test:datetime_end_test]
        self.target_test = self.target[datetime_start_test:datetime_end_test]

    """
    FUNCTION NAME: DataPreprocessor.randomShuffleTrainData

    PARAMETER:
        None

    RETURN:
        None

    DESCRIPTION:
        1. train 데이터를 sklearn.utils.shuffle 메서드를 이용해 랜덤 셔플 

    """
    def randomShuffleTrainData(self):
        self.features_train, self.target_train = sklearn.utils.shuffle(self.features_train, self.target_train, random_state=0)

    """
    FUNCTION NAME: DataPreprocessor.getDataFrameDataSet

    PARAMETER:
        None

    RETURN:
        1. DataPreprocessor.features_train
        2. DataPreprocessor.features_validation
        3. DataPreprocessor.features_test
        4. DataPreprocessor.target_train
        5. DataPreprocessor.target_validation
        6. DataPreprocessor.target_test

    DESCRIPTION:
        1. train 데이터를 sklearn.utils.shuffle 메서드를 이용해 랜덤셔플 

    """
    def getDataFrameDataSet(self):
        return self.features_train, self.features_validation, self.features_test, \
            self.target_train, self.target_validation, self.target_test
