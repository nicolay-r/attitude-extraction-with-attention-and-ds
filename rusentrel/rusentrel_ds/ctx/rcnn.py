#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys


sys.path.append('../../../')

from io_utils import RuSentRelBasedExperimentsIOUtils
from arekit.common.evaluation.evaluators.two_class import TwoClassEvaluator
from arekit.contrib.networks.context.architectures.rcnn import RCNN
from arekit.contrib.networks.context.configurations.rcnn import RCNNConfig
from arekit.contrib.experiments.engine import run_testing
from arekit.contrib.experiments.callback import CustomCallback
from rusentrel.ctx_names import ModelNames
from rusentrel.rusentrel_ds.common import \
    DS_NAME_PREFIX, \
    ds_ctx_common_config_settings, \
    ds_common_callback_modification_func
from rusentrel.classic.ctx.rcnn import ctx_rcnn_custom_config

from arekit.contrib.experiments.single.model import SingleInstanceTensorflowModel
from arekit.contrib.experiments.nn_io.rusentrel_with_ruattitudes import RuSentRelWithRuAttitudesBasedExperimentIO


def run_testing_rcnn(
        name_prefix=DS_NAME_PREFIX,
        cv_count=1,
        common_callback_func=ds_common_callback_modification_func):

    run_testing(full_model_name=name_prefix + ModelNames().RCNN,
                create_network=RCNN,
                cv_count=cv_count,
                create_config=RCNNConfig,
                create_nn_io=RuSentRelWithRuAttitudesBasedExperimentIO,
                create_model=SingleInstanceTensorflowModel,
                evaluator_class=TwoClassEvaluator,
                create_callback=CustomCallback,
                experiments_io=RuSentRelBasedExperimentsIOUtils(),
                common_config_modification_func=ds_ctx_common_config_settings,
                common_callback_modification_func=common_callback_func,
                custom_config_modification_func=ctx_rcnn_custom_config)


if __name__ == "__main__":

    run_testing_rcnn()
