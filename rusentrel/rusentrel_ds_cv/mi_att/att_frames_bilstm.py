#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys


sys.path.append('../../../')
from arekit.contrib.networks.multi.architectures.att_self import AttSelfOverSentences
from arekit.contrib.networks.multi.configurations.att_self import AttSelfOverSentencesConfig
from rusentrel.mi_names import AttSelfOverInstancesModelNames
from rusentrel.rusentrel_ds.mi.att_frames_bilstm import run_testing_mi_att_frames_bilstm
from rusentrel.rusentrel_ds_cv.common import CV_COUNT, \
    ds_cv_common_callback_modification_func, \
    CV_DS_NAME_PREFIX

if __name__ == "__main__":

    run_testing_mi_att_frames_bilstm(
        cv_count=CV_COUNT,
        name_prefix=CV_DS_NAME_PREFIX,
        model_names_classtype=AttSelfOverInstancesModelNames,
        network_classtype=AttSelfOverSentences,
        config_classtype=AttSelfOverSentencesConfig,
        common_callback_func=ds_cv_common_callback_modification_func)
