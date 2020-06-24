from rusentrel.rusentrel_ds.common import ds_common_callback_modification_func

CV_DS_NAME_PREFIX = u'ds_cv_'

CV_COUNT = 3


def ds_cv_common_callback_modification_func(callback):
    """
    This function describes configuration setup for all model callbacks.
    """
    ds_common_callback_modification_func(callback)
    callback.set_cancellation_acc_bound(0.999)
    callback.set_cancellation_f1_train_bound(0.85)
    callback.set_key_save_hidden_parameters(False)
    callback.set_key_stop_training_by_cost(True)
