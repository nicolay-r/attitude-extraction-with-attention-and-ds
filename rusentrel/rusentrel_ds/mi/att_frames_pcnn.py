#!/usr/bin/python
import sys


sys.path.append('../../../')
from rusentrel.mi_names import MaxPoolingModelNames
from io_utils import RuSentRelBasedExperimentsIOUtils
from arekit.common.evaluation.evaluators.two_class import TwoClassEvaluator
from arekit.contrib.experiments.multi.model import MultiInstanceTensorflowModel
from arekit.contrib.experiments.nn_io.rusentrel_with_ruattitudes import RuSentRelWithRuAttitudesBasedExperimentIO
from arekit.contrib.networks.multi.configurations.max_pooling import MaxPoolingOverSentencesConfig
from arekit.contrib.networks.context.architectures.att_frames_pcnn import AttentionFramesPCNN
from arekit.contrib.networks.context.configurations.att_frames_pcnn import AttentionFramesPCNNConfig
from arekit.contrib.networks.multi.architectures.max_pooling import MaxPoolingOverSentences

from rusentrel.classic.mi.att_frames_pcnn import mi_att_frames_pcnn_custom_config
from arekit.contrib.experiments.callback import CustomCallback
from arekit.contrib.experiments.engine import run_testing
from rusentrel.rusentrel_ds.common import \
    ds_common_callback_modification_func, \
    ds_mi_common_config_settings, \
    DS_NAME_PREFIX


def run_testing_mi_att_frames_pcnn(
        name_prefix=DS_NAME_PREFIX,
        cv_count=1,
        model_names_classtype=MaxPoolingModelNames,
        network_classtype=MaxPoolingOverSentences,
        config_classtype=MaxPoolingOverSentencesConfig,
        common_callback_func=ds_common_callback_modification_func):

    run_testing(full_model_name=name_prefix + model_names_classtype().AttFramesPCNN,
                create_network=lambda: network_classtype(context_network=AttentionFramesPCNN()),
                create_config=lambda: config_classtype(context_config=AttentionFramesPCNNConfig()),
                create_nn_io=RuSentRelWithRuAttitudesBasedExperimentIO,
                cv_count=cv_count,
                create_model=MultiInstanceTensorflowModel,
                evaluator_class=TwoClassEvaluator,
                create_callback=CustomCallback,
                experiments_io=RuSentRelBasedExperimentsIOUtils(),
                common_callback_modification_func=common_callback_func,
                custom_config_modification_func=mi_att_frames_pcnn_custom_config,
                common_config_modification_func=ds_mi_common_config_settings)


if __name__ == "__main__":

    run_testing_mi_att_frames_pcnn()