import os
from blip.config import parse_config

os.path.pardir
blip_root_dir = os.path.dirname(os.path.dirname(__file__))

def test_default_params():
    try:
        parse_config(os.path.join(blip_root_dir, "params_default.ini"), resume=True)
        parse_config(os.path.join(blip_root_dir, "params_simple.ini"), resume=True)
        parse_config(os.path.join(blip_root_dir, "params_test.ini"), resume=True)
    except:
        assert False, "One of the default config files is invalid!"
