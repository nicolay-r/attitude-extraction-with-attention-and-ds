#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys


sys.path.append('../../../')

from io_utils import RuSentRelBasedExperimentsIOUtils
from arekit.common.evaluation.evaluators.two_class import TwoClassEvaluator
from arekit.contrib.networks.context.architectures.self_att_bilstm import SelfAttentionBiLSTM
from arekit.contrib.networks.context.configurations.self_att_bilstm import SelfAttentionBiLSTMConfig
from arekit.networks.tf_helpers.sequence import CellTypes

from arekit.contrib.experiments.callback import CustomCallback
from arekit.contrib.experiments.engine import run_testing
from rusentrel.ctx_names import ModelNames
from rusentrel.classic.common import \
    classic_ctx_common_config_settings, \
    classic_common_callback_modification_func

from arekit.contrib.experiments.nn_io.rusentrel import RuSentRelBasedNeuralNetworkIO
from arekit.contrib.experiments.single.model import SingleInstanceTensorflowModel


def ctx_self_att_bilstm_custom_config(config):
    assert(isinstance(config, SelfAttentionBiLSTMConfig))
    config.modify_bags_per_minibatch(2)
    config.modify_penaltization_term_coef(0.5)
    config.modify_cell_type(CellTypes.BasicLSTM)
    config.modify_dropout_rnn_keep_prob(0.8)
    config.modify_terms_per_context(25)


def run_testing_self_att_bilstm(name_prefix=u'',
                                cv_count=1,
                                custom_config_func=ctx_self_att_bilstm_custom_config,
                                custom_callback_func=classic_common_callback_modification_func):

    run_testing(full_model_name=name_prefix + ModelNames().SelfAttentionBiLSTM,
                create_network=SelfAttentionBiLSTM,
                create_config=SelfAttentionBiLSTMConfig,
                cv_count=cv_count,
                create_nn_io=RuSentRelBasedNeuralNetworkIO,
                create_model=SingleInstanceTensorflowModel,
                evaluator_class=TwoClassEvaluator,
                experiments_io=RuSentRelBasedExperimentsIOUtils(),
                common_callback_modification_func=custom_callback_func,
                custom_config_modification_func=custom_config_func,
                common_config_modification_func=classic_ctx_common_config_settings,
                create_callback=CustomCallback)


if __name__ == "__main__":

    run_testing_self_att_bilstm()

