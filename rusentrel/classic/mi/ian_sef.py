#!/usr/bin/python
import sys


sys.path.append('../../../')

from io_utils import RuSentRelBasedExperimentsIOUtils

from arekit.contrib.networks.multi.configurations.max_pooling import MaxPoolingOverSentencesConfig
from arekit.contrib.networks.context.architectures.ian_sef import IANSynonymEndsAndFrames
from arekit.contrib.networks.context.configurations.ian_sef import IANSynonymEndsAndFramesConfig
from arekit.contrib.networks.multi.architectures.max_pooling import MaxPoolingOverSentences
from arekit.contrib.experiments.callback import CustomCallback
from arekit.contrib.experiments.multi.model import MultiInstanceTensorflowModel
from arekit.contrib.experiments.nn_io.rusentrel import RuSentRelBasedNeuralNetworkIO
from arekit.contrib.experiments.engine import run_testing

from arekit.common.evaluation.evaluators.two_class import TwoClassEvaluator

from rusentrel.mi_names import MaxPoolingModelNames
from rusentrel.classic.ctx.ian_sef import ctx_ian_sef_custom_config
from rusentrel.classic.common import \
    classic_common_callback_modification_func, \
    classic_mi_common_config_settings


def mi_ian_custom_config(config):
    ctx_ian_sef_custom_config(config.ContextConfig)
    config.fix_context_parameters()


def run_testing_ian_sef(name_prefix=u'',
                       cv_count=1,
                       model_names_classtype=MaxPoolingModelNames,
                       network_classtype=MaxPoolingOverSentences,
                       config_classtype=MaxPoolingOverSentencesConfig,
                       custom_config_func=mi_ian_custom_config,
                       custom_callback_func=classic_common_callback_modification_func):

    run_testing(full_model_name=name_prefix + model_names_classtype().IANSynonymEndsAndFrames,
                create_network=lambda: network_classtype(context_network=IANSynonymEndsAndFrames()),
                create_config=lambda: config_classtype(context_config=IANSynonymEndsAndFramesConfig()),
                create_nn_io=RuSentRelBasedNeuralNetworkIO,
                create_model=MultiInstanceTensorflowModel,
                evaluator_class=TwoClassEvaluator,
                create_callback=CustomCallback,
                cv_count=cv_count,
                experiments_io=RuSentRelBasedExperimentsIOUtils(),
                common_callback_modification_func=custom_callback_func,
                custom_config_modification_func=custom_config_func,
                common_config_modification_func=classic_mi_common_config_settings)


if __name__ == "__main__":

    run_testing_ian_sef()

