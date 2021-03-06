from agents import *
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import neptune.new as neptune

def debug_print_transitions(transition):
    state, action, reward, next_state, done = transition.state, transition.action, transition.reward, transition.next_state, transition.done

    print("Transition: ")
    print("State")
    for k in range(state.shape[0]):
        out = ""    
        for i in range(state.shape[1]):
            for j in range(state.shape[2]):
                out += f"{int(state[k, i, j])} "
            out += "\n"
        out += "\n"
        print(out)

    print("Action ", action)
    print("Reward ", reward)
    print("Done", done)

    print("Next State")
    for k in range(next_state.shape[0]):
        out = ""    
        for i in range(next_state.shape[1]):
            for j in range(next_state.shape[2]):
                out += f"{int(next_state[k, i, j])} "
            out += "\n"
        out += "\n"
        print(out)


class NepLogger:
    class __NepLogger:
        def __init__(self, update_interval=2000):
            self.logger = None
            self.update_interval = update_interval

        def __str__(self):
            return repr(self)

        def initialize_writer(self):
            pass
            # self.logger = neptune.init(project='tenvinc/cs4246-project')

        def get_writer(self):
            pass
            # return self.logger

        def add_scalar(self, tag, item):
            pass
            # self.logger[tag].log(item)

        def add_params(self, params):
            pass
            # self.logger["parameters"] = params

    instance = None

    def __init__(self, update_interval):
        if not NepLogger.instance:
            NepLogger.instance = NepLogger.__NepLogger(update_interval)
        
    def __getattr__(self, name):
        return getattr(self.instance, name)

    @classmethod
    def initialize_writer(cls):
        return cls.instance.initialize_writer()

    @classmethod
    def get_writer(cls):
        return cls.instance.get_writer()

    @classmethod
    def should_update(cls):
        return cls.instance.num_epochs % cls.instance.update_interval == 0

    @classmethod
    def add_scalar(cls, tag, item):
        assert cls.instance is not None
        # if cls.should_update():
        cls.instance.add_scalar(tag, item)

    @classmethod
    def add_params(cls, params):
        assert cls.instance is not None
        cls.instance.add_params(params)

class TFWriter:
    class __TFWriter:
        def __init__(self):
            self.writer = None
            self.num_epochs = None
            self.update_interval = 2000
            pass

        def __str__(self):
            return repr(self)

        def initialize_writer(self):
            name = datetime.now().strftime("%Y_%m_%d_%H_%M")

            self.writer = SummaryWriter(f"runs/{name}")

        def get_writer(self):
            return self.writer

    instance = None

    def __init__(self):
        if not TFWriter.instance:
            TFWriter.instance = TFWriter.__TFWriter()
    def __getattr__(self, name):
        return getattr(self.instance, name)

    @classmethod
    def initialize_writer(cls):
        return cls.instance.initialize_writer()

    @classmethod
    def get_writer(cls):
        return cls.instance.get_writer()

    @classmethod
    def set_num_epochs(cls, num_epochs):
        cls.instance.num_epochs = num_epochs

    @classmethod
    def get_num_epochs(cls):
        return cls.instance.num_epochs

    @classmethod
    def should_update(cls):
        return cls.instance.num_epochs % cls.instance.update_interval == 0

    @classmethod
    def add_scalar(cls, tag, item, global_step):
        if cls.should_update():
            cls.instance.get_writer().add_scalar(tag, item, global_step)

def log_weights_biases(model):
    """ Uses Tensorboard to log all trainable parameters """

    parameters = model.state_dict()
    for key in parameters.keys():
        TFWriter.get_writer().add_histogram(key, parameters[key])
    