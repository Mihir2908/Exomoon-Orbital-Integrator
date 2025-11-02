"""import files - matplotlib for plotting; numpy for numerical calculations""" 
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Ensure Plotly opens in a web browser with full animation controls when run from VS Code/terminal
pio.renderers.default = "browser"

#
"""define constants"""
bigG = 6.6732e-11 #N m^2 kg^-2
yr = 3.15576e7 #seconds per year 
rsun = 6.9598e8 #meters
msun = 1.989e30 #kg
merth = 5.976e24 #kg
rerth = 6.378e6 #meters
mjup = 317.894*merth
au = 1.495979e11 #meters
hplnck = 6.626196e-34 #planck constant
kboltz = 1.380622e-23 #boltzmann constant
cspeed = 2.99792e8 #light speed
stefboltz = 5.6696e-8 #Watts m^-2 K^-4
parsec = 3.0856e16 #meters
F_earth = 1370. #W/m^2
#print(au)
#
"""create lists for storing data"""
xyzlist_mp = []
xyzlist_ms = []
xyzlist_mm = []

"""initialize run parameters"""
Ts = 3784 #star temperature in K
rs = 0.51 #star radius in solar radii
ms = 0.54 #star mass in solar masses
mp = 2.54 #planet mass in earth masses
dp = 5.5 #density of planet in cgs units
mm = 1.0 #moon mass in earth masses 
ap = 0.3006 #distance between star and planet (e = 0) in AU
am = 0.45 #distance between planet and moon (e = 0) in Hill-radii
ep = 0.0 #eccentricity of planet around star; planet will start at periapse
em = 0.0 #eccentricity of moon around star; moon will start at periapse
"""unit conversions"""
rp = (0.75*mp*merth/(dp*1.e3))**(1./3.)/1.e3 #planet radius in km if needed
rs = rs * rsun  # star radius in meters
ms = ms*4.*np.pi**2 
mp = mp*merth/msun*4.*np.pi**2
mm = mm*merth/msun*4.*np.pi**2

rhill = ap*(1-ep)*(mp/3/ms)**0.5 #Hill radius around planet in AU 
#The (1-e) factor in the Hill-radius is described in Hamilton and Burns (1992)
#"Orbital stability zones about asteroids"
am = am*rhill #distance between planet and moon (e = 0) in AU
xpm = ap*(1-ep)*ms/(ms+mp+mm) #location of planet-moon barycenter relative to 
#system barycenter in AU
ypm = zpm = 0. #location of planet-moon barycenter relative to 
#system barycenter in AU
vxpm = vzpm = 0. #velocity of planet-moon barycenter relative to 
#system barycenter in AU/yr
vypm = ms/((1-ep)*(mp+mm+ms)*ap)**0.5 #velocity of planet-moon barycenter relative to 
#system barycenter in AU/yr
xs = -1.*ap*(1-ep)*(mp+mm)/(ms+mp+mm) #location of star center 
#relative to the system barycenter in AU
ys = zs = 0. #location of star center relative to the system barycenter in AU
vxs = vzs = 0. #velocity of star center relative to system barycenter in AU/yr
vys = -1*(mp+mm)/((1-ep)*(mp+mm+ms)*ap)**0.5 #velocity of star center 
#relative to system barycenter in AU/yr
yp = zp = 0. #location of planet relative to planet-moon barycenter in AU
xp = -1*am*(1-em)*mm/(mp+mm) #location of planet relative to planet-moon barycenter in AU
xp = xpm+xp #location of planet relative to system barycenter in AU
vxp = vzp = 0. #velocity of planet relative to planet-moon barycenter in AU/yr
vyp = -1*mm/((1-em)*(mp+mm)*am)**0.5 #velocity of planet relative to planet-moon barycenter in AU/yr
vyp = vypm + vyp #velocity of planet relative to system barycenter in AU/yr
ym = zm = 0. #location of moon relative to planet-moon barycenter in AU
xm = am*(1-em)*mp/(mp+mm) #location of moon relative to planet-moon barycenter in AU
xm = xpm + xm #location of moon relative to system barycenter in AU
vxm = vzm = 0. #velocity of moon relative to planet-moon barycenter in AU/yr
vym = mp/((1-em)*(mp+mm)*am)**0.5 #velocity of moon relative to planet-moon barycenter in AU/yr
vym = vypm + vym #velocity of moon relative to system barycenter in AU
#
pos_arr_mp = np.array([xp,yp,zp])
pos_arr_ms = np.array([xs,ys,zs])
pos_arr_mm = np.array([xm,ym,zm])
vel_arr_mp = np.array([vxp,vyp,vzp])
vel_arr_ms = np.array([vxs,vys,vzs])
vel_arr_mm = np.array([vxm,vym,vzm])
#

# Calculate star luminosity (Watts)
L_star = 4 * np.pi * rs**2 * stefboltz * Ts**4

# Habitable zone flux boundaries (W/m^2)
F_inner = 1.1 * F_earth  
F_outer = 0.5 * F_earth  

# Calculate distance boundaries in meters
a_inner = np.sqrt(L_star / (4 * np.pi * F_inner))
a_outer = np.sqrt(L_star / (4 * np.pi * F_outer))

# Convert to AU
a_inner_au = a_inner / au
a_outer_au = a_outer / au

print("Inner Habitable Zone (AU):", a_inner_au)
print("Outer Habitable Zone (AU):", a_outer_au)

t = 0
orbprd_mm_ms = 2.*np.pi*ap**1.5/(mp+ms)**0.5#yrs
orbprd_mm_mp = 2.*np.pi*am**1.5/(mp+mm)**0.5#yrs
dt = orbprd_mm_mp/1.e3 #years #runs a 1000 steps per moon-planet orbit
tend = orbprd_mm_ms #years
#print(orbprd_mm_mp)
#print("\n")
"""start time loop"""
while (t < tend):  
    t = t + dt
    #print(t)  # Commented to avoid slowing down runs; re-enable if needed
    """do a leapfrog step"""
    """move planet"""
    pos_arr2_mp = pos_arr_mp + vel_arr_mp*dt/2
    accel = -1*ms*(pos_arr2_mp-pos_arr_ms)/np.linalg.norm(pos_arr2_mp-pos_arr_ms)**3 + \
            -1*mm*(pos_arr2_mp-pos_arr_mm)/np.linalg.norm(pos_arr2_mp-pos_arr_mm)**3
    vel_arr_mp = vel_arr_mp + accel*dt
    pos_arr_mp = pos_arr2_mp + vel_arr_mp*dt/2
    """move star"""
    pos_arr2_ms = pos_arr_ms + vel_arr_ms*dt/2
    accel = -1*mp*(pos_arr2_ms-pos_arr_mp)/np.linalg.norm(pos_arr2_ms-pos_arr_mp)**3 + \
            -1*mm*(pos_arr2_ms-pos_arr_mm)/np.linalg.norm(pos_arr2_ms-pos_arr_mm)**3
    vel_arr_ms = vel_arr_ms + accel*dt
    pos_arr_ms = pos_arr2_ms + vel_arr_ms*dt/2
    """move moon"""
    pos_arr2_mm = pos_arr_mm + vel_arr_mm*dt/2
    accel = -1*mp*(pos_arr2_mm-pos_arr_mp)/np.linalg.norm(pos_arr2_mm-pos_arr_mp)**3 + \
            -1*ms*(pos_arr2_mm-pos_arr_ms)/np.linalg.norm(pos_arr2_mm-pos_arr_ms)**3
    vel_arr_mm = vel_arr_mm + accel*dt
    pos_arr_mm = pos_arr2_mm + vel_arr_mm*dt/2
    """write position data to lists"""
    xyzlist_mp.append(pos_arr_mp) 
    xyzlist_ms.append(pos_arr_ms) 
    xyzlist_mm.append(pos_arr_mm) 
"""end time loop"""

"""create figure - can change numbers to make different size"""    
fig, ax = plt.subplots(figsize=(6,6))
ax.grid()
"""plot data for three objects"""
xyzarr_mp = np.array(xyzlist_mp)
xyzarr_mm = np.array(xyzlist_mm)
xyzarr_ms = np.array(xyzlist_ms)

timesteps = len(xyzarr_mp)  # total steps from your simulation

# Downsample frames to keep animation responsive (<= ~400 frames)
frame_stride = max(1, timesteps // 1000)
frame_indices = list(range(0, timesteps, frame_stride))

# Calculate axis ranges based on all objects
all_x = np.concatenate([xyzarr_ms[:,0], xyzarr_mp[:,0], xyzarr_mm[:,0]])
all_y = np.concatenate([xyzarr_ms[:,1], xyzarr_mp[:,1], xyzarr_mm[:,1]])
x_range = [-1.2*max(np.abs(all_x)), 1.2*max(np.abs(all_x))]
y_range = [-1.2*max(np.abs(all_y)), 1.2*max(np.abs(all_y))]

# Calculate moon's relative positions to the planet for each frame
moon_rel = xyzarr_mm - xyzarr_mp  # shape: (timesteps, 3)

# Zoom panel range (symmetric)
r_rel = 1.2 * np.max(np.sqrt(moon_rel[:,0]**2 + moon_rel[:,1]**2))
zoom_range = [-r_rel, r_rel]

zoom_height_frac = 0.35  # use this to control vertical size of zoom panel
top_bottom = (1 - zoom_height_frac) / 2

# Create main figure with inset
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
    subplot_titles=["System Orbit", "Moon Zoom"],
)

# Create frames for animation - each frame updates positions of all 3 bodies
frames = []

for i in frame_indices:
    frames.append(
        go.Frame(
            data=[
                go.Scatter(x=[xyzarr_ms[i,0]], y=[xyzarr_ms[i,1]]),   # left star
                go.Scatter(x=[xyzarr_mp[i,0]], y=[xyzarr_mp[i,1]]),   # left planet
                go.Scatter(x=[xyzarr_mm[i,0]], y=[xyzarr_mm[i,1]]),   # left moon
                go.Scatter(x=[moon_rel[i,0]], y=[moon_rel[i,1]]),     # right moon
            ],
            traces=[3, 4, 5, 8],
            name=str(i)
        )
    )
fig.frames = frames

# Initial data for the first frame - also add orbital trails
    # Orbital trails (static) : All left panel
fig.add_trace(    
    go.Scatter(x=xyzarr_ms[:,0], y=xyzarr_ms[:,1], 
              mode='lines', line=dict(color='yellow', width=1), 
              name='Star Trail', opacity=0.3), row=1, col=1),

fig.add_trace(
    go.Scatter(x=xyzarr_mp[:,0], y=xyzarr_mp[:,1], 
              mode='lines', line=dict(color='blue', width=1), 
              name='Planet Trail', opacity=0.3), row=1, col=1)

fig.add_trace(
    go.Scatter(x=xyzarr_mm[:,0], y=xyzarr_mm[:,1], 
              mode='lines', line=dict(color='red', width=1), 
              name='Moon Trail', opacity=0.3), row=1, col=1),

    # Moving objects

    # Left Panel
fig.add_trace(    
    go.Scatter(x=[xyzarr_ms[0,0]], y=[xyzarr_ms[0,1]], 
              mode='markers', marker=dict(color='yellow', size=15), 
              name='Star'), row=1, col=1)

fig.add_trace(
    go.Scatter(x=[xyzarr_mp[0,0]], y=[xyzarr_mp[0,1]], 
              mode='markers', marker=dict(color='blue', size=5), 
              name='Planet'), row=1, col=1)

fig.add_trace(
    go.Scatter(x=[xyzarr_mm[0,0]], y=[xyzarr_mm[0,1]], 
              mode='markers', marker=dict(color='red', size=3.5), 
              name='Moon'), row=1, col=1)

    # Right Panel
fig.add_trace(
    go.Scatter(x=[0], y=[0], 
               mode='markers', marker=dict(color='blue', size=3.5), 
               name='Planet Zoom'), row=2, col=2),

fig.add_trace( 
    go.Scatter(x=moon_rel[:,0], y=moon_rel[:,1], 
               mode='lines', line=dict(color='red', width=1), 
               name='Moon Trail (zoom)', opacity=0.3), row=2, col=2)

fig.add_trace(
    go.Scatter(x=[moon_rel[0, 0]], y=[moon_rel[0, 1]], 
               mode='markers', marker=dict(color='red', size=3.5), 
               name='Moon Zoom'), row=2, col=2)

# Add habitable zone circles
fig.update_layout(
    shapes=[
        dict(
            type="circle",
            xref="x1",
            yref="y1",
            x0=-a_outer_au,
            y0=-a_outer_au,
            x1=a_outer_au,
            y1=a_outer_au,
            fillcolor="rgba(0,255,0,0.2)",
            line=dict(color="rgba(0,255,0,0)"),
            layer="below"
        ),
        dict(
            type="circle",
            xref="x1",
            yref="y1",
            x0=-a_inner_au,
            y0=-a_inner_au,
            x1=a_inner_au,
            y1=a_inner_au,
            fillcolor="white",
            line=dict(color="rgba(0,0,0,0)"),
            layer="below"
        )
    ]
)

# Matplotlib 2D plot
#ax.plot(xyzarr_mp[:,0],xyzarr_mp[:,1],color="b")  
#ax.plot(xyzarr_mm[:,0],xyzarr_mm[:,1],color="r") 
#ax.plot(xyzarr_ms[:,0],xyzarr_ms[:,1],color="k") 

#

#ax.set_xlim([-1.1*ap*(1+ep),1.1*ap*(1+ep)])
#ax.set_ylim([-1.1*ap*(1+ep),1.1*ap*(1+ep)])
#ax.set_xlabel('X (AU)', fontsize='14')
#ax.set_ylabel('Y (AU)', fontsize='14')

# Axes styling
fig.update_xaxes(title_text="X (AU)", range=x_range, row=1, col=1)
fig.update_yaxes(title_text="Y (AU)", range=y_range, scaleanchor="x", scaleratio=1, row=1, col=1)

fig.update_xaxes(title_text="ΔX (AU)", range=zoom_range, row=2, col=2)
fig.update_yaxes(title_text="ΔY (AU)", range=zoom_range, scaleanchor="x2", scaleratio=1, row=2, col=2)

# Controls
fig.update_layout(
    title="Three-Body Orbital Evolution",
    height=700,
    width=1200,  # widened so the left panel remains visually large
    updatemenus=[dict(
        type="buttons",
        direction="left",
        pad={"r": 10, "t": 70},
        showactive=True,
        x=0.02, xanchor="left",
        y=1.12, yanchor="top",
        buttons=[
            dict(label="Play", method="animate",
                 args=[None, {"frame": {"duration": 100, "redraw": True},
                              "fromcurrent": True,
                              "transition": {"duration": 50}}]),
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
        transition={"duration": 50, "easing": "cubic-in-out"},
        pad={"b": 10, "t": 55},
        len=0.9, x=0.1, y=1.08,
        steps=[dict(
            args=[[str(k)], {"frame": {"duration": 50, "redraw": True},
                             "mode": "immediate",
                             "transition": {"duration": 50}}],
            label=str(k), method="animate"
        ) for k in frame_indices]
    )]
)

"""write to file"""
#fig.savefig("myxyorbitplot.jpg")
fig.show(renderer="browser")

fig.write_html('orbit_anim.html')  