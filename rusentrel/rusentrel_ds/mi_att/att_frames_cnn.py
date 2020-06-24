#!/usr/bin/python
import sys


sys.path.append('../../../')
from arekit.contrib.networks.multi.configurations.att_self import AttSelfOverSentencesConfig
from arekit.contrib.networks.multi.architectures.att_self import AttSelfOverSentences
from rusentrel.classic_cv.mi.att_frames_cnn import run_testing_att_frames_cnn
from rusentrel.mi_names import AttSelfOverInstancesModelNames

if __name__ == "__main__":

    run_testing_att_frames_cnn(
        model_names_classtype=AttSelfOverInstancesModelNames,
        network_classtype=AttSelfOverSentences,
        config_classtype=AttSelfOverSentencesConfig)
