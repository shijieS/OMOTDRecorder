import json
import os


class Configure:
    """
    Singleton pattern class. use get_config.
    """
    instance = None

    def __init__(self, name):
        self.data = None
        self.name = name
        configure_file = os.path.join(os.path.dirname(__file__), name)
        with open(configure_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        if self.data is None:
            raise Exception("Cannot load configure files {}".format(configure_file))

    @staticmethod
    def get_config(name=None):
        """
        Get the configure instance
        :param name: the configure file name
        :return: the Configure instance
        """
        if Configure.instance is None or (not name and Configure.instance.name != name):
            Configure.instance = Configure(name)

        return Configure.instance.data
