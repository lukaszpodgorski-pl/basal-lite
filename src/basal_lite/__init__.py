"""basal decisions from a stock vLLM server. See README.md."""

from ._basal.prompt import RequestError
from .client import BasalLite

__version__ = "2.4.1.1"  # basal 2.4.1 (the first three numbers), basal-lite release 1 for it (README.md, "Versions")
__all__ = ["BasalLite", "RequestError", "__version__"]
