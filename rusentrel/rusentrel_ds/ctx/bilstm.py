#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys


sys.path.append('../../../')

from io_utils import RuSentRelBasedExperimentsIOUtils
from arekit.common.evaluation.evaluators.two_class import TwoClassEvaluator
from arekit.contrib.experiments.nn_io.rusentrel_with_ruattitudes import RuSentRelWithRuAttitudesBasedExperimentIO
from arekit.contrib.experiments.single.model import SingleInstanceTensorflowModel
from arekit.contrib.networks.context.configurations.bilstm import BiLSTMConfig
from arekit.contrib.networks.context.architectures.bilstm import BiLSTM

from rusentrel.ctx_names import ModelNames
from arekit.contrib.experiments.engine import run_testing
from arekit.contrib.experiments.callback import CustomCallback
from rusentrel.rusentrel_ds.common import DS_NAME_PREFIX, \
    ds_ctx_common_config_settings, \
    ds_common_callback_modification_func
from rusentrel.classic.ctx.bilstm import ctx_bilstm_custom_config


def run_testing_bilstm(
        name_prefix=DS_NAME_PREFIX,
        cv_count=1,
        common_callback_func=ds_common_callback_modification_func):

    run_testing(full_model_name=name_prefix + ModelNames().BiLSTM,
                create_network=BiLSTM,
                create_config=BiLSTMConfig,
                create_model=SingleInstanceTensorflowModel,
                create_nn_io=RuSentRelWithRuAttitudesBasedExperimentIO,
                cv_count=cv_count,
                create_callback=CustomCallback,
                evaluator_class=TwoClassEvaluator,
                experiments_io=RuSentRelBasedExperimentsIOUtils(),
                common_config_modification_func=ds_ctx_common_config_settings,
                common_callback_modification_func=common_callback_func,
                custom_config_modification_func=ctx_bilstm_custom_config)


if __name__ == "__main__":
    run_testing_bilstm()
