import neptune.new as neptune

default_update_interval = 0  # Not used
class NepLogger():
    ''' Logger that logs to Neptune.ai '''
    def __init__(self, update_interval=default_update_interval):
        self.logger = None
        self.update_interval = update_interval

    def __str__(self):
        return repr(self)

    def initialize_writer(self):
        self.logger = neptune.init(project='tenvinc/cs4246-project', source_files=['agent/*.py', 'requirements.txt'])

    def get_writer(self):
        return self.logger

    def add_scalar(self, tag, item):
        self.logger[tag].log(item)

    def add_params(self, params):
        self.logger["parameters"] = params

    def add_tag(self, tag):
        self.logger["sys/tags"].add([tag])

class GenericLogger():
    ''' Generic logger singleton '''
    instance = None

    def __init__(self, update_interval=default_update_interval):
        if not GenericLogger.instance:
            GenericLogger.instance = NepLogger(update_interval)
            print(GenericLogger.instance)
        
    def __getattr__(self, name):
        return getattr(self.instance, name)

    @classmethod
    def initialize_writer(cls):
        return cls.instance.initialize_writer()

    @classmethod
    def get_writer(cls):
        return cls.instance.get_writer()

    @classmethod
    def add_scalar(cls, tag, item):
        assert cls.instance is not None
        cls.instance.add_scalar(tag, item)

    @classmethod
    def add_params(cls, params):
        assert cls.instance is not None
        cls.instance.add_params(params)
    
    @classmethod
    def add_tag(cls, tag):
        assert cls.instance is not None
        cls.instance.add_tag(tag)
