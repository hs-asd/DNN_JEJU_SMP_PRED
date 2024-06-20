import logging
from settings import *
import os

def setLogger(name, file_name, file_mode='a'):
    dir_save = os.path.dirname(file_name)
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)

    # logger 생성 및 설정
    logger_stream = logging.getLogger(name + '.stream')
    logger_file = logging.getLogger(name + '.file')
    logger_stream.setLevel(logging.DEBUG)
    logger_file.setLevel(logging.DEBUG)
    handler_stream = logging.StreamHandler()
    handler_file = logging.FileHandler(file_name, mode=file_mode)
    handler_stream.setLevel(logging.DEBUG)
    handler_file.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler_stream.setFormatter(formatter)
    handler_file.setFormatter(formatter)
    logger_stream.addHandler(handler_stream)
    logger_file.addHandler(handler_file)

    return logger_stream, logger_file

def loggingDataConfiguration(stream_logger, file_logger, cache_path, input_labels, output_labels, ipt_start, ipt_end, ept_start, ept_end):
    stream_logger.info('Cache File: %s', cache_path)
    stream_logger.info('    Input Labels:           %s', input_labels)
    stream_logger.info('    Ouput Label:            %s', output_labels)
    stream_logger.info('    Imported Data Range:    %s ~ %s', ipt_start, ipt_end)
    stream_logger.info('    EXported Data Range:    %s ~ %s\n', ept_start, ept_end)

    file_logger.info('Cache File: %s', cache_path)
    file_logger.info('\tInput Labels:\t\t%s', input_labels)
    file_logger.info('\tOuput Label:\t\t%s', output_labels)
    file_logger.info('\tImported Data Range:\t%s ~ %s', ipt_start, ipt_end)
    file_logger.info('\tEXported Data Range:\t%s ~ %s\n', ept_start, ept_end)


def loggingTrainSetup(stream_logger, file_logger):
    stream_logger.info('Training Setup.')
    stream_logger.info('   Cache File:             %s', CACHE_FILE)
    stream_logger.info('   Learning Rate:          %s', LEARNING_RATE)
    stream_logger.info('   Input Dimension:        %s', DIM_INPUT)
    stream_logger.info('   Hidden Dimension:       %s', DIM_HIDDEN)
    stream_logger.info('   Ouput Dimension:        %s', DIM_OUTPUT)
    stream_logger.info('   Activation Function:    %s', ACTIVATION_FUNCTION)
    stream_logger.info('   Scale Map:              %s', SCALING_MAP)
    stream_logger.info('   Epochs:                 %s', EPOCHS)
    stream_logger.info('   Loss Function:          %s', LOSS_FUNCTION)
    stream_logger.info('   Train Range:            %s ~ %s', datetime.datetime(*RANGE_TRAIN[0]), datetime.datetime(*RANGE_TRAIN[-1]))
    stream_logger.info('   Validation Range:       %s ~ %s', datetime.datetime(*RANGE_VALIDATION[0]), datetime.datetime(*RANGE_VALIDATION[-1]))
    stream_logger.info('   Test Range:             %s ~ %s', datetime.datetime(*RANGE_TEST[0]), datetime.datetime(*RANGE_TEST[-1]))
    stream_logger.info('   Random Shuffle:         %s', RANDOM_SHUFFLE)

    file_logger.info('Training Setup.')
    file_logger.info('\tCache File:\t\t%s', CACHE_FILE)
    file_logger.info('\tLearning Rate:\t\t%s', LEARNING_RATE)
    file_logger.info('\tInput Dimension:\t\t%s', DIM_INPUT)
    file_logger.info('\tHidden Dimension:\t\t%s', DIM_HIDDEN)
    file_logger.info('\tOuput Dimension:\t\t%s', DIM_OUTPUT)
    file_logger.info('\tActivation Function:\t%s', ACTIVATION_FUNCTION)
    file_logger.info('\tScale Map:\t\t%s', SCALING_MAP)
    file_logger.info('\tEpochs:\t\t\t%s', EPOCHS)
    file_logger.info('\tLoss Function:\t\t%s', LOSS_FUNCTION)
    file_logger.info('\tTrain Range:\t\t%s ~ %s', datetime.datetime(*RANGE_TRAIN[0]),datetime.datetime(*RANGE_TRAIN[-1]))
    file_logger.info('\tValidation Range:\t\t%s ~ %s', datetime.datetime(*RANGE_VALIDATION[0]), datetime.datetime(*RANGE_VALIDATION[-1]))
    file_logger.info('\tTest Range:\t\t%s ~ %s', datetime.datetime(*RANGE_TEST[0]), datetime.datetime(*RANGE_TEST[-1]))
    file_logger.info('\tRandom Shuffle:\t\t%s', RANDOM_SHUFFLE)


def loggingTrainMetrics(stream_logger, file_logger, epoch, train_loss, validation_loss, test_loss, MAPE_validation, MAPE_test):
    stream_logger.info('Epoch: %s / %s', epoch, EPOCHS)
    stream_logger.info('   Train Loss:         %.4f', train_loss.item())
    stream_logger.info('   Validation Loss:    %.4f', validation_loss.item())
    stream_logger.info('   Test Loss:          %.4f', test_loss.item())
    stream_logger.info('   Validation MAPE:    %.4f', MAPE_validation)
    stream_logger.info('   Test MAPE:          %.4f', MAPE_test)

    file_logger.info('Epoch: %s / %s', epoch, EPOCHS)
    file_logger.info('\tTrain Loss:\t\t%.4f', train_loss.item())
    file_logger.info('\tValidation Loss:\t\t%.4f', validation_loss.item())
    file_logger.info('\tTest Loss:\t\t%.4f', test_loss.item())
    file_logger.info('\tValidation MAPE:\t\t%.4f', MAPE_validation)
    file_logger.info('\tTest MAPE:\t\t%.4f', MAPE_test)

