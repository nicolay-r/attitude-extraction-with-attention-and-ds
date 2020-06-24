import numpy as np

from arekit.contrib.experiments.nn_io.rusentrel import RuSentRelBasedNeuralNetworkIO
from arekit.common.evaluation.results.two_class import TwoClassEvalResult
from arekit.networks.cancellation import OperationCancellation
from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.networks.data_type import DataType
from arekit.networks.tf_model import TensorflowModel
from arekit.networks.predict_log import NetworkInputDependentVariables
from rusentrel.tests.fake_network import FakeNeuralNetwork


class FakeTensorflowModel(TensorflowModel):

    def __init__(self, callback, config):
        assert(isinstance(config, DefaultNetworkConfig))
        config.modify_bag_size(1)
        io = RuSentRelBasedNeuralNetworkIO("test_model")
        network = FakeNeuralNetwork()
        self.config = config
        super(FakeTensorflowModel, self).__init__(
            nn_io=io, network=network, callback=callback)

    @property
    def Config(self):
        return self.config

    def fit(self):
        operation_cancel = OperationCancellation()
        for epoch_index in xrange(10):

            if operation_cancel.IsCancelled:
                break

            e_fit_cost, e_fit_acc = 0.0, 0.0

            if self.Callback is not None:
                self.Callback.on_epoch_finished(avg_fit_cost=e_fit_cost,
                                                avg_fit_acc=e_fit_acc,
                                                epoch_index=epoch_index,
                                                operation_cancel=operation_cancel)

        if self.Callback is not None:
            self.Callback.on_fit_finished()

    def get_hidden_parameters(self):
        return [], []

    def predict(self, dest_data_type=DataType.Test):
        # TODO.
        predict_log = NetworkInputDependentVariables()
        samples = self.Config.BagsPerMinibatch * self.Config.BagSize
        value = np.random.rand(samples)
        predict_log.add_input_dependent_values(names_list=[u"X"],
                                               tensor_values_list=[value],
                                               text_opinion_ids=range(samples),
                                               bags_per_minibatch=self.Config.BagsPerMinibatch,
                                               bag_size=self.Config.BagSize)

        for pair in predict_log.iter_by_parameter_values(u"X"):
            print pair

        eval_result = TwoClassEvalResult()
        eval_result.calculate()

        return eval_result, predict_log
