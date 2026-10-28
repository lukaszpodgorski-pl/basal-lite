"""basal decisions from a stock vLLM server. See README.md."""
from ._basal.prompt import RequestError
from .client import BasalLite

__version__ = "2.4.1.post1"  # basal 2.4.1, the release its prompt format and calibration files follow
__all__ = ["BasalLite", "RequestError", "__version__"]
