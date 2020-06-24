#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
sys.path.append('../../../')
from rusentrel.rusentrel_ds.ctx.cnn import run_testing_cnn
from rusentrel.rusentrel_ds_cv.common import CV_COUNT, \
    ds_cv_common_callback_modification_func, \
    CV_DS_NAME_PREFIX

if __name__ == "__main__":

    run_testing_cnn(
        cv_count=CV_COUNT,
        name_prefix=CV_DS_NAME_PREFIX,
        common_callback_func=ds_cv_common_callback_modification_func)
