# -*- coding: utf-8 -*-
from utils.logger import G_LOGGER

def _init_():
    global _global_dict
    _global_dict = {}
 
def set_value(key,value):
    """ set global """
    _global_dict[key] = value
 
def get_value(key):
    """ get global """
    try:
        return _global_dict[key]
    except KeyError:
        G_LOGGER.info("Key '{}' does not exist.".format(key))