from os import path

from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from fake_model import FakeTensorflowModel
from io_utils import RuSentRelBasedExperimentsIOUtils
from arekit.contrib.experiments.callback import CustomCallback


config = DefaultNetworkConfig()
io_data_utils = RuSentRelBasedExperimentsIOUtils()
callback = CustomCallback(log_dir=path.join(io_data_utils.get_data_root(), u"test_callback"))

callback.set_test_on_epochs([0])

model = FakeTensorflowModel(callback=callback,
                            config=config)

callback.on_initialized(model=model)

with callback:
    model.fit()
