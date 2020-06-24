#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys


sys.path.append('../../../')
from arekit.contrib.networks.multi.configurations.att_self import AttSelfOverSentencesConfig
from arekit.contrib.networks.multi.architectures.att_self import AttSelfOverSentences
from rusentrel.mi_names import AttSelfOverInstancesModelNames
from rusentrel.classic.mi.att_hidden_z_yang import run_testing_att_hidden_zyang_bilstm
from rusentrel.classic_cv.common import CV_COUNT, \
    classic_cv_common_callback_modification_func, \
    CV_NAME_PREFIX


if __name__ == "__main__":

    run_testing_att_hidden_zyang_bilstm(
        name_prefix=CV_NAME_PREFIX,
        cv_count=CV_COUNT,
        model_names_classtype=AttSelfOverInstancesModelNames,
        network_classtype=AttSelfOverSentences,
        config_classtype=AttSelfOverSentencesConfig,
        custom_callback_func=classic_cv_common_callback_modification_func)
