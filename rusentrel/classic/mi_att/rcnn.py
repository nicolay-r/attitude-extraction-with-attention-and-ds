#!/usr/bin/python
import sys

sys.path.append('../../../')
from arekit.contrib.networks.multi.architectures.att_self import AttSelfOverSentences
from arekit.contrib.networks.multi.configurations.att_self import AttSelfOverSentencesConfig
from rusentrel.classic.mi.rcnn import run_testing_rcnn
from rusentrel.mi_names import AttSelfOverInstancesModelNames

if __name__ == "__main__":

    run_testing_rcnn(
        model_names_classtype=AttSelfOverInstancesModelNames,
        network_classtype=AttSelfOverSentences,
        config_classtype=AttSelfOverSentencesConfig
    )
