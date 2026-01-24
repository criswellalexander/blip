Configuration file format
=========================

BLIP's user interface is a single `.ini` file.

.. todo::
    Document the configuration options using Sphinx's confval directives. I haven't
    figured out a way to make the custom extension do this.

    Example of a confval directive:

    .. confval:: seglen
        :type: ``float``
        :default: 1e5

        Segment length for the analysis time-frequency grid.

Section `params`
----------------

.. blip-config-section:: SECTION_PARAMS


Section `inj`
-------------

.. blip-config-section:: SECTION_INJ


Section `run_params`
--------------------

.. blip-config-section:: SECTION_RUN_PARAMS
