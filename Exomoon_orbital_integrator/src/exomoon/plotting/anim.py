import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

def build_animation(traj: dict, a_inner_au: float, a_outer_au: float, open_in_browser=True):
    if open_in_browser:
        pio.renderers.default = "browser"

    xyzarr_ms = traj["xyzarr_ms"].astype(np.float32, copy = False)
    xyzarr_mp = traj["xyzarr_mp"].astype(np.float32, copy = False)
    xyzarr_mm = traj["xyzarr_mm"].astype(np.float32, copy = False)

    timesteps = len(xyzarr_mp)
    frame_stride = max(1, timesteps // 1000)
    frame_indices = list(range(0, timesteps, frame_stride))

    # Ranges based on max possible positions of any of the bodies
    #all_x = np.concatenate([xyzarr_ms[:, 0], xyzarr_mp[:, 0], xyzarr_mm[:, 0]])
    #all_y = np.concatenate([xyzarr_ms[:, 1], xyzarr_mp[:, 1], xyzarr_mm[:, 1]])
    #x_range = [-1.2 * np.max(np.abs(all_x)), 1.2 * np.max(np.abs(all_x))]
    #y_range = [-1.2 * np.max(np.abs(all_y)), 1.2 * np.max(np.abs(all_y))]

    # Ranges based on habitable zone
    hz_r = float(a_outer_au)
    pad_frac = 0.05  # 5% padding
    x_range = [-(1 + pad_frac) * hz_r, (1 + pad_frac) * hz_r]
    y_range = [-(1 + pad_frac) * hz_r, (1 + pad_frac) * hz_r]


    # Moon relative to planet (for zoom)
    moon_rel = xyzarr_mm - xyzarr_mp
    r_rel = 1.2 * np.max(np.sqrt(moon_rel[:, 0] ** 2 + moon_rel[:, 1] ** 2))
    zoom_range = [-r_rel, r_rel]

    # Decimated paths for progressive trails (use only frame_indices to limit data size)
    ms_x = xyzarr_ms[frame_indices, 0]; ms_y = xyzarr_ms[frame_indices, 1]
    mp_x = xyzarr_mp[frame_indices, 0]; mp_y = xyzarr_mp[frame_indices, 1]
    mm_x = xyzarr_mm[frame_indices, 0]; mm_y = xyzarr_mm[frame_indices, 1]
    mr_x = moon_rel[frame_indices, 0];  mr_y = moon_rel[frame_indices, 1]

    # Number of points to keep visible in the trail window
    trail_window = 200

    # Layout with smaller right subplot
    zoom_height_frac = 0.35
    top_bottom = (1 - zoom_height_frac) / 2.0

    fig = make_subplots(
        rows=3, cols=2,
        column_widths=[0.78, 0.22],
        row_heights=[top_bottom, zoom_height_frac, top_bottom],
        horizontal_spacing=0.06,
        vertical_spacing=0.02,
        specs=[
            [{"type": "xy", "rowspan": 3}, None],  # left spans full height
            [None, {"type": "xy"}],                # right zoom only in middle row
            [None, None],
        ],
        subplot_titles=["System Orbit", "Moon Zoom (relative)"],
    )

    # Trails (left) — hidden before Play
    fig.add_trace(go.Scattergl(x=[], y=[], mode="lines",
                             line=dict(color="yellow", width=1), name="Star Trail", opacity=0.3, visible=False),
                  row=1, col=1)
    star_trail_idx = len(fig.data) - 1

    fig.add_trace(go.Scattergl(x=[], y=[], mode="lines",
                             line=dict(color="blue", width=1), name="Planet Trail", opacity=0.3, visible=False),
                  row=1, col=1)
    planet_trail_idx = len(fig.data) - 1

    fig.add_trace(go.Scattergl(x=[], y=[], mode="lines",
                             line=dict(color="red", width=1), name="Moon Trail", opacity=0.3, visible=False),
                  row=1, col=1)
    moon_trail_idx = len(fig.data) - 1

    # Moving markers (left)
    fig.add_trace(go.Scatter(x=[xyzarr_ms[0, 0]], y=[xyzarr_ms[0, 1]], mode="markers",
                             marker=dict(color="yellow", size=15), name="Star"),
                  row=1, col=1)
    star_marker_idx = len(fig.data) - 1

    fig.add_trace(go.Scatter(x=[xyzarr_mp[0, 0]], y=[xyzarr_mp[0, 1]], mode="markers",
                             marker=dict(color="blue", size=5), name="Planet"),
                  row=1, col=1)
    planet_marker_idx = len(fig.data) - 1

    fig.add_trace(go.Scatter(x=[xyzarr_mm[0, 0]], y=[xyzarr_mm[0, 1]], mode="markers",
                             marker=dict(color="red", size=3.5), name="Moon"),
                  row=1, col=1)
    moon_marker_idx = len(fig.data) - 1

    # Zoom panel (right, middle row)
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers",
                             marker=dict(color="blue", size=6), name="Planet (zoom)"),
                  row=2, col=2)
    fig.add_trace(go.Scattergl(x=[], y=[], mode="lines",
                             line=dict(color="red", width=1), name="Moon Trail (zoom)", opacity=0.3, visible=False),
                  row=2, col=2)
    moon_zoom_trail_idx = len(fig.data) - 1

    fig.add_trace(go.Scatter(x=[moon_rel[0, 0]], y=[moon_rel[0, 1]], mode="markers",
                             marker=dict(color="red", size=5), name="Moon (zoom)"),
                  row=2, col=2)
    moon_zoom_marker_idx = len(fig.data) - 1

    # Axes
    fig.update_xaxes(title_text="X (AU)", range=x_range, row=1, col=1)
    fig.update_yaxes(title_text="Y (AU)", range=y_range, scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_xaxes(title_text="ΔX (AU)", range=zoom_range, row=2, col=2)
    fig.update_yaxes(title_text="ΔY (AU)", range=zoom_range, scaleanchor="x2", scaleratio=1, row=2, col=2)

    # HZ rings (left only)
    fig.update_layout(
        shapes=[
            dict(type="circle", xref="x1", yref="y1",
                 x0=-a_outer_au, y0=-a_outer_au, x1=a_outer_au, y1=a_outer_au,
                 fillcolor="rgba(0,255,0,0.2)", line=dict(color="rgba(0,0,0,0)"), layer="below"),
            dict(type="circle", xref="x1", yref="y1",
                 x0=-a_inner_au, y0=-a_inner_au, x1=a_inner_au, y1=a_inner_au,
                 fillcolor="white", line=dict(color="rgba(0,0,0,0)"), layer="below"),
        ]
    )

    # Frames: reveal trails and update them with a sliding window; move markers
    frames = []
    for k, i in enumerate(frame_indices):
        start = max(0, k - trail_window + 1)

        frames.append(go.Frame(
            data=[
                # Trails (left)
                go.Scatter(x=ms_x[start:k+1], y=ms_y[start:k+1], visible=True),
                go.Scatter(x=mp_x[start:k+1], y=mp_y[start:k+1], visible=True),
                go.Scatter(x=mm_x[start:k+1], y=mm_y[start:k+1], visible=True),
                # Trail (zoom)
                go.Scatter(x=mr_x[start:k+1], y=mr_y[start:k+1], visible=True),
                # Moving markers
                go.Scatter(x=[xyzarr_ms[i, 0]], y=[xyzarr_ms[i, 1]]),
                go.Scatter(x=[xyzarr_mp[i, 0]], y=[xyzarr_mp[i, 1]]),
                go.Scatter(x=[xyzarr_mm[i, 0]], y=[xyzarr_mm[i, 1]]),
                go.Scatter(x=[moon_rel[i, 0]], y=[moon_rel[i, 1]]),
            ],
            traces=[
                star_trail_idx, planet_trail_idx, moon_trail_idx, moon_zoom_trail_idx,
                star_marker_idx, planet_marker_idx, moon_marker_idx, moon_zoom_marker_idx
            ],
            name=str(i)
        ))
    fig.frames = frames

    # Controls
    fig.update_layout(
        title="Three-Body Orbital Evolution",
        height=700,
        width=1200,
        updatemenus=[dict(
            type="buttons",
            direction="left",
            pad={"r": 10, "t": 70},
            showactive=True,
            x=0.02, xanchor="left",
            y=1.12, yanchor="top",
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, {"frame": {"duration": 80, "redraw": True},
                                  "fromcurrent": True,
                                  "transition": {"duration": 40}}]),
                dict(label="Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}]),
            ],
        )],
        sliders=[dict(
            active=0,
            yanchor="bottom", xanchor="left",
            currentvalue={"font": {"size": 16}, "prefix": "Timestep: ", "visible": True, "xanchor": "right"},
            transition={"duration": 40, "easing": "cubic-in-out"},
            pad={"b": 10, "t": 55},
            len=0.9, x=0.1, y=1.08,
            steps=[dict(
                args=[[str(k)], {"frame": {"duration": 40, "redraw": True},
                                 "mode": "immediate",
                                 "transition": {"duration": 40}}],
                label=str(k), method="animate"
            ) for k in frame_indices]
        )]
    )

    return fig