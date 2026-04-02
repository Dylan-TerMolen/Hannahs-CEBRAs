"""Position decoding across environments A and B.

This module provides the pos_decoding_AvsB function by re-exporting
from pos_decoding_AvsB_DEP for backwards compatibility.
"""
from pos_decoding_AvsB_DEP import pos_decoding_AvsB_dep as pos_decoding_AvsB

__all__ = ['pos_decoding_AvsB']
