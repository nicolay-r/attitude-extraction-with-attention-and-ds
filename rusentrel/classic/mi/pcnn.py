#!/usr/bin/python
import sys


sys.path.append('../../../')
from io_utils import RuSentRelBasedExperimentsIOUtils
from arekit.contrib.experiments.callback import CustomCallback
from arekit.contrib.networks.multi.configurations.max_pooling import MaxPoolingOverSentencesConfig
from arekit.contrib.networks.multi.architectures.max_pooling import MaxPoolingOverSentences
from arekit.common.evaluation.evaluators.two_class import TwoClassEvaluator
from arekit.contrib.networks.context.configurations.cnn import CNNConfig
from arekit.contrib.networks.context.architectures.pcnn import PiecewiseCNN

from arekit.contrib.experiments.multi.model import MultiInstanceTensorflowModel
from arekit.contrib.experiments.nn_io.rusentrel import RuSentRelBasedNeuralNetworkIO

from arekit.contrib.experiments.engine import run_testing
from rusentrel.mi_names import MaxPoolingModelNames
from rusentrel.classic.ctx.pcnn import ctx_pcnn_custom_config
from rusentrel.classic.common import \
    classic_common_callback_modification_func, \
    classic_mi_common_config_settings


def mi_pcnn_custom_config(config):
    ctx_pcnn_custom_config(config.ContextConfig)
    config.fix_context_parameters()


def run_testing_pcnn(name_prefix=u'',
                     cv_count=1,
                     model_names_classtype=MaxPoolingModelNames,
                     network_classtype=MaxPoolingOverSentences,
                     config_classtype=MaxPoolingOverSentencesConfig,
                     custom_config_func=mi_pcnn_custom_config,
                     custom_callback_func=classic_common_callback_modification_func):

    run_testing(full_model_name=name_prefix + model_names_classtype().PCNN,
                create_network=lambda: network_classtype(context_network=PiecewiseCNN()),
                create_config=lambda: config_classtype(context_config=CNNConfig()),
                create_nn_io=RuSentRelBasedNeuralNetworkIO,
                cv_count=cv_count,
                create_model=MultiInstanceTensorflowModel,
                evaluator_class=TwoClassEvaluator,
                create_callback=CustomCallback,
                experiments_io=RuSentRelBasedExperimentsIOUtils(),
                common_callback_modification_func=custom_callback_func,
                custom_config_modification_func=custom_config_func,
                common_config_modification_func=classic_mi_common_config_settings)


if __name__ == "__main__":

    run_testing_pcnn()
