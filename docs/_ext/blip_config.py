################################################################
# Custom extension to automatically document configuration options by processing the
# data in blip/config.py.
################################################################

from docutils import nodes
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective
import blip.config

# TODO make this generate the prettier `confval` sphinx directives.
# TODO also document spectral and spatial models.
# This tutorial may be of help:
# https://www.sphinx-doc.org/en/master/development/tutorials/autodoc_ext.html

class BlipConfigDirective(SphinxDirective):
    """
    Usage:
        .. blip-config-section:: SECTION_PARAMS
    
    Get the list of parameters and render it in RST.
    """

    required_arguments = 1

    def run(self):
        opts: list[blip.config.Option] = getattr(blip.config, self.arguments[0])

        optnodes = []
        for opt in opts:
            text = f"**{opt.name}**: {opt.desc}, default {opt.default}, required={opt.required}"
            optnodes.append(nodes.paragraph(text=text))
        return optnodes


def setup(app: Sphinx):
    app.add_directive("blip-config-section", BlipConfigDirective)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
