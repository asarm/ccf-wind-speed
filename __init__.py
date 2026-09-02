"""Canonical implementation of the LACGNN model described in revision 2."""

from .config import LACGNNConfig
from .model import LACGNN

__all__ = ["LACGNN", "LACGNNConfig"]
