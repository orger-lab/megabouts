from matplotlib import colors


colorblind_list = [
    "#0173b2",
    "#de8f05",
    "#029e73",
    "#d55e00",
    "#cc78bc",
    "#ca9161",
    "#fbafe4",
    "#949494",
    "#ece133",
    "#56b4e9",
]

bouts_category_name = [
    "approach_swim",
    "slow1",
    "slow2",
    "short_capture_swim",
    "long_capture_swim",
    "burst_swim",
    "J_turn",
    "high_angle_turn",
    "routine_turn",
    "spot_avoidance_turn",
    "O_bend",
    "long_latency_C_start",
    "short_latency_C_start",
]
bouts_category_name_short = [
    "AS",
    "S1",
    "S2",
    "SCS",
    "LCS",
    "BS",
    "JT",
    "HAT",
    "RT",
    "SAT",
    "O",
    "LLC",
    "SLC",
]
bouts_category_color = [
    "#82cfff",
    "#4589ff",
    "#0000c8",
    "#5d5d66",
    "#000000",
    "#fcaf6d",
    "#ffb3b8",
    "#08bdba",
    "#24a148",
    "#9b82f3",
    "#ee5396",
    "#e3bc13",
    "#fa4d56",
]


cmp_bouts = colors.ListedColormap(bouts_category_color)
cmp_bouts.set_under(color="white")
cmp_bouts.set_over(color="white")
cmp_bouts.set_bad(color="grey", alpha=None)


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
