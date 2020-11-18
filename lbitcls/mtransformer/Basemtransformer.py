from abc import ABCMeta, abstractclassmethod

class Basemtransformer(metaclass=ABCMeta):
    r""" Base class for converting float model to quantization model
    Your models should also subclass this class.
    """

    def __init__(self):
        pass
    
    def __call__(self):
        raise NotImplementedError

    def __repr__(self):
        format_string = self.__class__.__name__
        return format_string