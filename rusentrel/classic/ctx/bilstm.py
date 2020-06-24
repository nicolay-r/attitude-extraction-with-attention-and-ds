#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys


sys.path.append('../../../')

from io_utils import RuSentRelBasedExperimentsIOUtils
from arekit.contrib.networks.context.configurations.bilstm import BiLSTMConfig
from arekit.contrib.networks.context.architectures.bilstm import BiLSTM
from arekit.networks.tf_helpers.sequence import CellTypes
from arekit.common.evaluation.evaluators.two_class import TwoClassEvaluator

from rusentrel.ctx_names import ModelNames
from arekit.contrib.experiments.engine import run_testing
from arekit.contrib.experiments.callback import CustomCallback

from arekit.contrib.experiments.nn_io.rusentrel import RuSentRelBasedNeuralNetworkIO
from arekit.contrib.experiments.single.model import SingleInstanceTensorflowModel

from rusentrel.classic.common import \
    classic_ctx_common_config_settings, \
    classic_common_callback_modification_func


def ctx_bilstm_custom_config(config):
    assert(isinstance(config, BiLSTMConfig))
    config.modify_hidden_size(128)
    config.modify_bags_per_minibatch(2)
    config.modify_cell_type(CellTypes.BasicLSTM)
    config.modify_dropout_rnn_keep_prob(0.8)
    config.modify_bags_per_minibatch(4)
    config.modify_terms_per_context(25)


def run_testing_bilstm(name_prefix=u'',
                       cv_count=1,
                       custom_config_func=ctx_bilstm_custom_config,
                       custom_callback_func=classic_common_callback_modification_func):

    run_testing(full_model_name=name_prefix + ModelNames().BiLSTM,
                create_network=BiLSTM,
                create_config=BiLSTMConfig,
                create_model=SingleInstanceTensorflowModel,
                create_nn_io=RuSentRelBasedNeuralNetworkIO,
                cv_count=cv_count,
                create_callback=CustomCallback,
                evaluator_class=TwoClassEvaluator,
                experiments_io=RuSentRelBasedExperimentsIOUtils(),
                common_callback_modification_func=custom_callback_func,
                custom_config_modification_func=custom_config_func,
                common_config_modification_func=classic_ctx_common_config_settings)


if __name__ == "__main__":

    run_testing_bilstm()
