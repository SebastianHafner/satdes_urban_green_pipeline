"""Vendored from satdes_maxndvi/methods/composite.py (server-side max composite only)."""
from __future__ import annotations


def max_composite_server(datacube):
    return datacube.reduce_dimension(dimension="t", reducer="max")
