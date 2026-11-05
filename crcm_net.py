"""Compatibility import for the retired editorial name; no separate model."""

from grc_polypnet import GRCPolypNet, STUDY_ARMS

CRCMNet = GRCPolypNet

__all__ = ["CRCMNet", "STUDY_ARMS"]
