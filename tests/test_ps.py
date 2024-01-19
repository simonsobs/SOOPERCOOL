"""
Unit tests for the soopercool.ps_utils module.
"""
from soopercool import ps_utils

nside = 32
bin_size = 8


def test_get_coupled_pseudo_cls():
    pass


def test_decouple_pseudo_cls():
    pass


def test_field_pairs_from_spins():
    cls_in = {
        "spin0xspin0": ["TT"],
        "spin0xspin2": ["TE", "TB"],
        "spin2xspin0": ["ET", "BT"],
        "spin2xspin2": ["EE", "EB", "BE", "BB"]
    }
    cls_out = ps_utils.field_pairs_from_spins(cls_in)
    for k in cls_out:
        assert cls_out[k] == k


def test_get_pcls_mat_transfer():
    pass
