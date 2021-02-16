"""
Languages Tests
---------------
"""

from kwx import languages


def test_language_returns():
    assert type(languages.lem_abbr_dict()) == dict
    assert type(languages.stem_abbr_dict()) == dict
    assert type(languages.sw_abbr_dict()) == dict
