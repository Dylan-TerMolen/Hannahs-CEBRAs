"""Hannahs-CEBRAs: CEBRA-based neural decoding tools.

Provides functions for condition and position decoding across environments.
"""
import sys
from pathlib import Path

# Add parent directory to path so we can import root-level modules
_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

# Main decoding functions
from cond_decoding_AvsB import cond_decoding_AvsB
from pos_decoding_self import pos_decoding_self
from pos_decoding_AvsB import pos_decoding_AvsB

# Helper functions (also exported for convenience)
from hold_out import hold_out
from pos_score import pos_score
from CSUS_score import CSUS_score

__all__ = [
    'cond_decoding_AvsB',
    'pos_decoding_self',
    'pos_decoding_AvsB',
    'hold_out',
    'pos_score',
    'CSUS_score',
]
