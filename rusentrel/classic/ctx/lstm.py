#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys


sys.path.append('../../../')

from io_utils import RuSentRelBasedExperimentsIOUtils
from arekit.contrib.networks.context.configurations.rnn import RNNConfig
from arekit.networks.tf_helpers.sequence import CellTypes
from arekit.common.evaluation.evaluators.two_class import TwoClassEvaluator
from arekit.contrib.networks.context.architectures.rnn import RNN

from arekit.contrib.experiments.nn_io.rusentrel import RuSentRelBasedNeuralNetworkIO
from arekit.contrib.experiments.single.model import SingleInstanceTensorflowModel

from arekit.contrib.experiments.callback import CustomCallback
from rusentrel.ctx_names import ModelNames
from arekit.contrib.experiments.engine import run_testing

from rusentrel.classic.common import \
    classic_ctx_common_config_settings, \
    classic_common_callback_modification_func


def ctx_lstm_custom_config(config):
    assert(isinstance(config, RNNConfig))
    config.modify_cell_type(CellTypes.BasicLSTM)
    config.modify_hidden_size(128)
    config.modify_bags_per_minibatch(2)
    config.modify_dropout_rnn_keep_prob(0.8)
    config.modify_learning_rate(0.1)
    config.modify_terms_per_context(25)


def run_testing_lstm(cv_count=1,
                     name_prefix=u'',
                     custom_config_func=ctx_lstm_custom_config,
                     custom_callback_func=classic_common_callback_modification_func):

    run_testing(full_model_name=name_prefix + ModelNames().LSTM,
                create_network=RNN,
                create_config=RNNConfig,
                create_model=SingleInstanceTensorflowModel,
                create_nn_io=RuSentRelBasedNeuralNetworkIO,
                cv_count=cv_count,
                evaluator_class=TwoClassEvaluator,
                create_callback=CustomCallback,
                experiments_io=RuSentRelBasedExperimentsIOUtils(),
                common_callback_modification_func=custom_callback_func,
                custom_config_modification_func=custom_config_func,
                common_config_modification_func=classic_ctx_common_config_settings)


if __name__ == "__main__":

    run_testing_lstm()

