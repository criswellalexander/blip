import pytest
from blip.config import parse_config

def test_default_params():
    try:
        parse_config("params_default.ini", resume=True)
        parse_config("params_simple.ini", resume=True)
        parse_config("params_test.ini", resume=True)
    except:
        assert False, "One of the default config files is invalid!"
