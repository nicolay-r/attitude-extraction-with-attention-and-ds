#!/usr/bin/python
import sys
sys.path.append('../../../')
from rusentrel.mi_names import AttSelfOverInstancesModelNames
from rusentrel.classic.mi.ian_se import run_testing_ian_se
from arekit.contrib.networks.multi.configurations.att_self import AttSelfOverSentencesConfig
from arekit.contrib.networks.multi.architectures.att_self import AttSelfOverSentences

if __name__ == "__main__":
    run_testing_ian_se(
        model_names_classtype=AttSelfOverInstancesModelNames,
        network_classtype=AttSelfOverSentences,
        config_classtype=AttSelfOverSentencesConfig
    )

