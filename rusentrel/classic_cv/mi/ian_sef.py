#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
sys.path.append('../../../')

from rusentrel.classic.mi.ian_sef import run_testing_ian_sef
from rusentrel.classic_cv.common import CV_COUNT, \
    classic_cv_common_callback_modification_func, \
    CV_NAME_PREFIX


if __name__ == "__main__":

    run_testing_ian_sef(
        name_prefix=CV_NAME_PREFIX,
        cv_count=CV_COUNT,
        custom_callback_func=classic_cv_common_callback_modification_func)
