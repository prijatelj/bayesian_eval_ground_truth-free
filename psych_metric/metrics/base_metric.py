"""
Base Metric class to be inherited by other dataset classes
Overwrite these methods
"""

class BaseMetric(object):

    def train(self):
        raise NotImplementedError

    def calculate(self):
        raise NotImplementedError
