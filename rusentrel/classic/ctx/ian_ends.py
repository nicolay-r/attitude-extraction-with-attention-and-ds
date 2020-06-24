#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import tensorflow as tf


sys.path.append('../../../')

from io_utils import RuSentRelBasedExperimentsIOUtils
from arekit.contrib.networks.context.architectures.ian_ends import IANEndsBased
from arekit.contrib.networks.context.configurations.ian_ends import IANEndsBasedConfig
from arekit.common.evaluation.evaluators.two_class import TwoClassEvaluator
from arekit.networks.tf_helpers.sequence import CellTypes

from arekit.contrib.experiments.nn_io.rusentrel import RuSentRelBasedNeuralNetworkIO
from arekit.contrib.experiments.single.model import SingleInstanceTensorflowModel

from arekit.contrib.experiments.callback import CustomCallback
from rusentrel.ctx_names import ModelNames
from arekit.contrib.experiments.engine import run_testing

from rusentrel.classic.common import \
    classic_ctx_common_config_settings, \
    classic_common_callback_modification_func


def ctx_ian_ends_custom_config(config):
    assert(isinstance(config, IANEndsBasedConfig))
    config.modify_bags_per_minibatch(2)
    config.modify_hidden_size(128)
    config.modify_cell_type(CellTypes.BasicLSTM)
    config.modify_l2_reg(0.0)
    config.modify_learning_rate(0.1)
    config.modify_weight_initializer(tf.contrib.layers.xavier_initializer())
    config.modify_dropout_rnn_keep_prob(0.8)
    config.modify_gpu_memory_fraction(0.7)


def run_testing_ian_ends(cv_count=1,
                         name_prefix=u'',
                         custom_config_func=ctx_ian_ends_custom_config,
                         custom_callback_func=classic_common_callback_modification_func):

    run_testing(full_model_name=name_prefix + ModelNames().IANEnds,
                create_network=IANEndsBased,
                create_config=IANEndsBasedConfig,
                create_nn_io=RuSentRelBasedNeuralNetworkIO,
                cv_count=cv_count,
                create_model=SingleInstanceTensorflowModel,
                evaluator_class=TwoClassEvaluator,
                create_callback=CustomCallback,
                experiments_io=RuSentRelBasedExperimentsIOUtils(),
                common_callback_modification_func=custom_callback_func,
                custom_config_modification_func=custom_config_func,
                common_config_modification_func=classic_ctx_common_config_settings)


if __name__ == "__main__":

    run_testing_ian_ends()