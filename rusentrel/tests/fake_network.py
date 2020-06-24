from arekit.networks.nn import NeuralNetwork


class FakeNeuralNetwork(NeuralNetwork):

    @property
    def ParametersDictionary(self):
        return {}

    @property
    def Cost(self):
        return None

    @property
    def Labels(self):
        return None

    @property
    def Accuracy(self):
        return None

    def iter_hidden_parameters(self):
        return [], []

    def iter_input_dependent_hidden_parameters(self):

        for name, value in super(FakeNeuralNetwork, self).iter_input_dependent_hidden_parameters():
            yield name, value

        yield [], []

    def compile(self, config, reset_graph):
        pass

    def create_feed_dict(self, input, data_type):
        pass
