#!/usr/bin/python
import sys


sys.path.append('../../../')
from arekit.contrib.networks.multi.configurations.max_pooling import MaxPoolingOverSentencesConfig
from arekit.contrib.experiments.multi.model import MultiInstanceTensorflowModel
from arekit.contrib.networks.context.configurations.rcnn import RCNNConfig
from arekit.contrib.networks.multi.architectures.max_pooling import MaxPoolingOverSentences
from arekit.common.evaluation.evaluators.two_class import TwoClassEvaluator
from arekit.contrib.networks.context.architectures.rcnn import RCNN

from arekit.contrib.experiments.nn_io.rusentrel import RuSentRelBasedNeuralNetworkIO

from arekit.contrib.experiments.callback import CustomCallback
from rusentrel.mi_names import MaxPoolingModelNames
from arekit.contrib.experiments.engine import run_testing
from rusentrel.classic.ctx.rcnn import ctx_rcnn_custom_config
from rusentrel.classic.common import \
    classic_common_callback_modification_func, \
    classic_mi_common_config_settings


def mi_rcnn_custom_config(config):
    ctx_rcnn_custom_config(config.ContextConfig)
    config.fix_context_parameters()


def run_testing_rcnn(name_prefix=u'',
                     cv_count=1,
                     model_names_classtype=MaxPoolingModelNames,
                     network_classtype=MaxPoolingOverSentences,
                     config_classtype=MaxPoolingOverSentencesConfig,
                     custom_config_func=mi_rcnn_custom_config,
                     custom_callback_func=classic_common_callback_modification_func):

    run_testing(full_model_name=name_prefix + model_names_classtype().RCNN,
                create_network=lambda: network_classtype(context_network=RCNN()),
                create_config=lambda: config_classtype(context_config=RCNNConfig()),
                create_nn_io=RuSentRelBasedNeuralNetworkIO,
                cv_count=cv_count,
                create_model=MultiInstanceTensorflowModel,
                evaluator_class=TwoClassEvaluator,
                create_callback=CustomCallback,
                common_callback_modification_func=custom_callback_func,
                custom_config_modification_func=custom_config_func,
                common_config_modification_func=classic_mi_common_config_settings)


if __name__ == "__main__":

    run_testing_rcnn()
