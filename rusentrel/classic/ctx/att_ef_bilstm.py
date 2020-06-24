#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import tensorflow as tf


sys.path.append('../../../')

from io_utils import RuSentRelBasedExperimentsIOUtils
from arekit.contrib.networks.context.architectures.att_ef_bilstm import AttentionEndsAndFramesBiLSTM
from arekit.contrib.networks.context.configurations.att_ef_bilstm import AttentionEndsAndFramesBiLSTMConfig
from arekit.common.evaluation.evaluators.two_class import TwoClassEvaluator

from arekit.contrib.experiments.nn_io.rusentrel import RuSentRelBasedNeuralNetworkIO
from arekit.contrib.experiments.single.model import SingleInstanceTensorflowModel

from rusentrel.ctx_names import ModelNames
from arekit.contrib.experiments.engine import run_testing
from arekit.contrib.experiments.callback import CustomCallback

from rusentrel.classic.common import \
    classic_ctx_common_config_settings, \
    classic_common_callback_modification_func


def ctx_att_ef_bilstm_custom_config(config):
    assert(isinstance(config, AttentionEndsAndFramesBiLSTMConfig))
    config.modify_bags_per_minibatch(2)
    config.modify_weight_initializer(tf.contrib.layers.xavier_initializer())


def run_testing_att_ef_bilstm(name_prefix=u'',
                              cv_count=1,
                              custom_config_func=ctx_att_ef_bilstm_custom_config,
                              custom_callback_func=classic_common_callback_modification_func):

    run_testing(full_model_name=name_prefix + ModelNames().AttEndsAndFramesBiLSTM,
                create_network=AttentionEndsAndFramesBiLSTM,
                create_config=AttentionEndsAndFramesBiLSTMConfig,
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

    run_testing_att_ef_bilstm()
