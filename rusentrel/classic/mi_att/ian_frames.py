#!/usr/bin/python
import sys


sys.path.append('../../../')
from arekit.contrib.networks.multi.architectures.att_self import AttSelfOverSentences
from arekit.contrib.networks.multi.configurations.att_self import AttSelfOverSentencesConfig
from rusentrel.mi_names import AttSelfOverInstancesModelNames
from rusentrel.classic.mi.ian_frames import run_testing_ian_frames


if __name__ == "__main__":
    run_testing_ian_frames(
        model_names_classtype=AttSelfOverInstancesModelNames,
        network_classtype=AttSelfOverSentences,
        config_classtype=AttSelfOverSentencesConfig
    )


