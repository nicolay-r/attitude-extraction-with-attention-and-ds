#!/usr/bin/python
import sys
sys.path.append('../../../')
from arekit.contrib.networks.multi.configurations.att_self import AttSelfOverSentencesConfig
from arekit.contrib.networks.multi.architectures.att_self import AttSelfOverSentences
from rusentrel.rusentrel_ds.mi.self_att_bilstm import run_testing_mi_self_att_bilstm
from rusentrel.mi_names import AttSelfOverInstancesModelNames

if __name__ == "__main__":

    run_testing_mi_self_att_bilstm(
        model_names_classtype=AttSelfOverInstancesModelNames,
        network_classtype=AttSelfOverSentences,
        config_classtype=AttSelfOverSentencesConfig
    )
