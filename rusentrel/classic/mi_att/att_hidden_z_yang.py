#!/usr/bin/python
import sys


sys.path.append('../../../')
from arekit.contrib.networks.multi.configurations.att_self import AttSelfOverSentencesConfig
from arekit.contrib.networks.multi.architectures.att_self import AttSelfOverSentences
from rusentrel.mi_names import AttSelfOverInstancesModelNames
from rusentrel.classic.mi.att_hidden_z_yang import run_testing_att_hidden_zyang_bilstm


if __name__ == "__main__":

    run_testing_att_hidden_zyang_bilstm(
        model_names_classtype=AttSelfOverInstancesModelNames,
        network_classtype=AttSelfOverSentences,
        config_classtype=AttSelfOverSentencesConfig
    )
