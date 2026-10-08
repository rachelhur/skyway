import pytest

from skyway.telescope import get_telescope, list_telescopes


def test_all_keys_resolve():
    assert set(list_telescopes()) == {"blanco"}


def test_blanco_site_is_ctio():
    site = get_telescope("blanco").site
    assert site.lat == pytest.approx(-30.1691, abs=1e-3)
    loc = site.earth_location()
    assert loc.lat.deg == pytest.approx(-30.1691, abs=1e-3)


def test_unknown_key_raises():
    with pytest.raises(KeyError):
        get_telescope("nope")
