"""Tests for signal y-axis semantics: robust limits, shared scales, badges."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ggplot2_py as gg
from ggplot2_py.plot import ggplot_build
import ggtracks as ggt


MAPPER = ggt.GenomicMapper.from_intervals([(100, 200), (400, 500)])


def _cov(track, scale=1.0):
    return pd.DataFrame(
        {
            "xstart": [100, 150, 400],
            "xend": [150, 200, 500],
            "value": np.array([5.0, 9.0, 3.0]) * scale,
            "track": track,
        }
    )


def _track(name, *, scale=1.0, **kwargs):  # noqa: D401
    data = _cov(name, scale)
    return ggt.Track(
        name,
        [ggt.geom_coverage(gg.aes(xstart="xstart", xend="xend", y="value"), data=data)],
        **kwargs,
    )


# --------------------------------------------------------------------------
# signal_limits
# --------------------------------------------------------------------------


def test_baseline_is_zero_not_the_data_minimum():
    lo, _hi = ggt.signal_limits([10, 20, 30])
    assert lo == 0.0


def test_quantile_clips_a_lone_spike():
    """The point of the quantile: one outlier must not flatten the rest."""
    body = list(np.linspace(1, 10, 99))
    _lo, clipped = ggt.signal_limits(body + [10_000], q=0.99)
    _lo, raw = ggt.signal_limits(body + [10_000], q=1.0)
    assert clipped < raw / 10


def test_expand_leaves_headroom():
    _lo, hi = ggt.signal_limits([100], q=1.0, expand=1.25)
    assert hi == pytest.approx(125.0)


def test_q_one_is_the_plain_maximum():
    _lo, hi = ggt.signal_limits([1, 2, 3], q=1.0, expand=1.0)
    assert hi == pytest.approx(3.0)


def test_nans_are_ignored():
    _lo, hi = ggt.signal_limits([1.0, np.nan, 3.0], q=1.0, expand=1.0)
    assert hi == pytest.approx(3.0)


def test_flat_zero_track_still_gets_a_usable_axis():
    lo, hi = ggt.signal_limits([0.0, 0.0])
    assert hi > lo


def test_custom_baseline():
    lo, _hi = ggt.signal_limits([5, 6], baseline=-1.0)
    assert lo == -1.0


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"q": 0}, "q must be"),
        ({"q": 1.5}, "q must be"),
        ({"expand": 0}, "expand must be"),
    ],
)
def test_bad_arguments_fail_loud(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ggt.signal_limits([1, 2, 3], **kwargs)


def test_no_finite_values_fails_loud():
    with pytest.raises(ValueError, match="no finite values"):
        ggt.signal_limits([np.nan, np.inf])


# --------------------------------------------------------------------------
# Track y configuration
# --------------------------------------------------------------------------


def test_y_limits_are_validated():
    with pytest.raises(ValueError, match="high must exceed low"):
        ggt.Track("t", [], y_limits=(5, 5))


def test_range_label_requires_limits():
    """The badge reports the axis, so there must be an axis to report."""
    with pytest.raises(ValueError, match="range_label needs y_limits"):
        ggt.Track("t", [], range_label=True)


def test_defaults_leave_y_free():
    t = ggt.Track("t", [])
    assert t.y_limits is None and t.y_breaks is None and t.range_label is False


def _y_ranges(plot):
    """Distinct y ranges across the built panels."""
    built = ggplot_build(plot)
    return {
        tuple(np.round(np.asarray(pp["y_range"], dtype=float), 6))
        for pp in built.layout.panel_params
        if pp.get("y_range") is not None
    }


def test_shared_limits_make_two_tracks_comparable():
    """Free y would draw a weak track as tall as a strong one."""
    limits = ggt.signal_limits(np.r_[_cov("a").value, _cov("b", 10).value])
    tracks = [_track("a", y_limits=limits), _track("b", scale=10, y_limits=limits)]
    assert len(_y_ranges(ggt.plot_tracks(tracks, MAPPER, show=False))) == 1


def test_free_y_by_default_gives_different_panel_ranges():
    tracks = [_track("a"), _track("b", scale=10)]
    assert len(_y_ranges(ggt.plot_tracks(tracks, MAPPER, show=False))) == 2


def test_explicit_breaks_replace_the_defaults():
    built = ggplot_build(
        ggt.plot_tracks([_track("a", y_breaks=[0.0, 5.0])], MAPPER, show=False)
    )
    assert len(built.layout.panel_params[0]["y_labels"]) == 2


def test_blank_labels_quiet_a_gene_model_row():
    """A gene-model row's y is a row index; numeric ticks on it are noise."""
    built = ggplot_build(
        ggt.plot_tracks(
            [_track("a", y_breaks=[1.0], y_labels=[""])], MAPPER, show=False
        )
    )
    assert built.layout.panel_params[0]["y_labels"] == [""]


def test_empty_breaks_are_rejected():
    """Upstream scale_y_continuous(breaks=[]) prints panel-relative
    positions instead of removing the axis, so it must not be offered as a
    way to hide ticks."""
    with pytest.raises(ValueError, match="does not remove the axis"):
        ggt.Track("t", [], y_breaks=[])


def test_labels_require_matching_breaks():
    with pytest.raises(ValueError, match="needs matching y_breaks"):
        ggt.Track("t", [], y_labels=[""])
    with pytest.raises(ValueError, match="entries but y_breaks has"):
        ggt.Track("t", [], y_breaks=[1.0, 2.0], y_labels=[""])


def test_range_label_adds_one_badge_layer():
    plain = ggt.plot_tracks([_track("a")], MAPPER, show=False)
    badged = ggt.plot_tracks(
        [_track("a", y_limits=(0, 12), range_label=True)], MAPPER, show=False
    )
    assert len(badged.layers) == len(plain.layers) + 1


def test_badge_text_reads_as_a_range(tmp_path):
    from ggtracks.plot_tracks import _format_range

    assert _format_range(0, 1479) == "[0-1479]"
    # the badge is a compact statement of the axis, so it rounds
    assert _format_range(0.0, 232.895) == "[0-233]"
    assert _format_range(0.0, 1.234) == "[0-1.23]"
    p = ggt.plot_tracks(
        [_track("a", y_limits=(0, 12), range_label=True)], MAPPER, show=False
    )
    out = tmp_path / "badge.png"
    gg.ggsave(str(out), p, width=4, height=p.fig_height, dpi=72)
    assert out.stat().st_size > 0


def test_mixed_configured_and_free_tracks_render(tmp_path):
    tracks = [
        _track("a", y_limits=(0, 12), range_label=True),
        _track("b", scale=10),
        _track("c", y_breaks=[1.0], y_labels=[""]),
    ]
    p = ggt.plot_tracks(tracks, MAPPER, show=False)
    out = tmp_path / "mixed.png"
    gg.ggsave(str(out), p, width=4, height=p.fig_height, dpi=72)
    assert out.stat().st_size > 0


def test_track_order_must_name_real_tracks():
    with pytest.raises(ValueError, match="no matching Track"):
        ggt.plot_tracks([_track("a")], MAPPER, track_order=["a", "ghost"], show=False)


# --------------------------------------------------------------------------
# tracks with nothing to draw
# --------------------------------------------------------------------------


def test_empty_track_is_dropped_with_a_warning_not_an_error():
    """A cluster with no reads in the window is ordinary, not a mistake."""
    empty = _cov("b").iloc[0:0]
    tracks = [
        _track("a"),
        ggt.Track(
            "b",
            [ggt.geom_coverage(gg.aes(xstart="xstart", xend="xend", y="value"), data=empty)],
        ),
    ]
    with pytest.warns(UserWarning, match="no data for track"):
        p = ggt.plot_tracks(tracks, MAPPER, show=False)
    assert len(ggplot_build(p).layout.panel_params) == 1


def test_track_with_no_layers_is_dropped_too():
    with pytest.warns(UserWarning, match="no data for track"):
        ggt.plot_tracks([_track("a"), ggt.Track("b", [])], MAPPER, show=False)


def test_dropping_a_track_keeps_the_remaining_heights_aligned():
    """force_panelsizes shortens its list positionally, so a vanished row
    would otherwise shift every height onto the wrong panel."""
    from grid_py import Unit

    empty = _cov("a").iloc[0:0]
    with pytest.warns(UserWarning):
        mixed = ggt.plot_tracks(
            [
                ggt.Track("a", [ggt.geom_coverage(
                    gg.aes(xstart="xstart", xend="xend", y="value"), data=empty)],
                    height=Unit(0.5, "inches")),
                _track("b", height=Unit(3.0, "inches")),
            ],
            MAPPER, show=False,
        )
    alone = ggt.plot_tracks([_track("b", height=Unit(3.0, "inches"))], MAPPER, show=False)
    assert mixed.fig_height == pytest.approx(alone.fig_height, abs=1e-6)


def test_a_layer_missing_the_track_column_is_an_error():
    """It would be drawn on every panel rather than its own."""
    tracks = [
        _track("a"),
        ggt.Track("b", [ggt.geom_coverage(
            gg.aes(xstart="xstart", xend="xend", y="value"),
            data=_cov("b").drop(columns="track"))]),
    ]
    with pytest.raises(ValueError, match="drawn on every panel"):
        ggt.plot_tracks(tracks, MAPPER, show=False)


def test_a_mistyped_track_value_is_reported_as_a_mismatch():
    bad = _cov("a").assign(track="typo")
    tracks = [ggt.Track("a", [ggt.geom_coverage(
        gg.aes(xstart="xstart", xend="xend", y="value"), data=bad)])]
    with pytest.raises(ValueError, match="match no Track"):
        ggt.plot_tracks(tracks, MAPPER, show=False)


def test_no_data_at_all_is_an_error():
    tracks = [ggt.Track("a", [ggt.geom_coverage(
        gg.aes(xstart="xstart", xend="xend", y="value"), data=_cov("a").iloc[0:0])])]
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError, match="no track has any data"):
            ggt.plot_tracks(tracks, MAPPER, show=False)
