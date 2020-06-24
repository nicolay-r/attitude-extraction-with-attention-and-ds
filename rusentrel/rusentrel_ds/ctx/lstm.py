#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys


sys.path.append('../../../')

from io_utils import RuSentRelBasedExperimentsIOUtils
from arekit.common.evaluation.evaluators.two_class import TwoClassEvaluator
from arekit.contrib.experiments.single.model import SingleInstanceTensorflowModel
from arekit.contrib.experiments.nn_io.rusentrel_with_ruattitudes import RuSentRelWithRuAttitudesBasedExperimentIO
from arekit.contrib.networks.context.configurations.rnn import RNNConfig
from arekit.contrib.networks.context.architectures.rnn import RNN
from arekit.contrib.experiments.callback import CustomCallback
from rusentrel.ctx_names import ModelNames
from arekit.contrib.experiments.engine import run_testing
from rusentrel.rusentrel_ds.common import DS_NAME_PREFIX, \
    ds_ctx_common_config_settings, \
    ds_common_callback_modification_func
from rusentrel.classic.ctx.lstm import ctx_lstm_custom_config


def run_testing_lstm(
        cv_count=1,
        name_prefix=DS_NAME_PREFIX,
        common_callback_func=ds_common_callback_modification_func):

    run_testing(full_model_name=name_prefix + ModelNames().LSTM,
                create_network=RNN,
                create_config=RNNConfig,
                cv_count=cv_count,
                create_model=SingleInstanceTensorflowModel,
                create_nn_io=RuSentRelWithRuAttitudesBasedExperimentIO,
                evaluator_class=TwoClassEvaluator,
                create_callback=CustomCallback,
                experiments_io=RuSentRelBasedExperimentsIOUtils(),
                common_config_modification_func=ds_ctx_common_config_settings,
                common_callback_modification_func=common_callback_func,
                custom_config_modification_func=ctx_lstm_custom_config)

if __name__ == "__main__":

    run_testing_lstm()

