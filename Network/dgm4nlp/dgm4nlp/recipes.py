"""
:Authors: - Wilker Aziz
"""
import gzip
from io import TextIOWrapper


def smart_ropen(path):
    """Opens files directly or through gzip depending on extension."""
    if path.endswith('.gz'):
        return TextIOWrapper(gzip.open(path, 'rb'))
    else:
        return open(path, 'r')


def smart_wopen(path):
    """Opens files directly or through gzip depending on extension."""
    if path.endswith('.gz'):
        return TextIOWrapper(gzip.open(path, 'wb'))
    else:
        return open(path, 'w')
