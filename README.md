# Exomoon Orbital Integrator

Simulation Framework hosting codebase for a Leapfrog Algorithm-based Python integrator to numerically simulate the orbital evolution of any characterized 3 body (Star-Planet-Moon) system, analyzed and orbitally animated via a Dash Plotly UI. Supports automated parameterization from the NASA Exoplanet Archive and tools accessible to Claude (LLM), via an MCP Server. 

## Key Functions supported by Claude

The MCP server provides tool endpoints to Claude (or another LLM agent) for:

**fetch_exoplanet:** Lookup and import system properties from the NASA Exoplanet Archive by planet name.

**run_sim & run_sim_years:** Run a simulation for the specified or default duration; outputs a Plotly HTML animation.

**check_moon_stability, assess_stability, assess_moon_stability, moon_escape_info:** Analyze moon orbital stability, estimate escape times, and summarize system evolution.

**dash_url:** Generate a browser-accessible Dash UI session for real-time visualization and manual parameter tweaking.

## Input Parameters

| Parameter       | Description                        | Units               |
|-----------------|----------------------------------|---------------------|
| Ts              | Star effective temperature       | Kelvin (K)          |
| rs_solar        | Star radius                     | Solar radii (R☉)     |
| ms_solar        | Star mass                       | Solar masses (M☉)    |
| mp_earth        | Planet mass                    | Earth masses (M⊕)    |
| dp_cgs          | Planet mean density             | g/cm³               |
| ap_AU           | Planet semi-major axis           | Astronomical Units (AU) |
| ep              | Planet eccentricity             | Dimensionless       |
| mm_earth        | Moon mass                      | Earth masses (M⊕)    |
| am_hill         | Moon orbit as fraction of Hill radius | Dimensionless   |
| em              | Moon eccentricity               | Dimensionless       |
| moon_retrograde | Moon retrograde flag            | Boolean             |
| years           | Simulation duration (optional)   | Years               |


## Constraining Exomoon Habitability and Orbital Stability

Condition A: An exomoon orbit is stable if Planetary Hill Radius > Moon Semi Major Axis, where 

$$
r_\mathrm{Hill} = a_p \, (1-e_p) \left( \frac{M_p}{3 M_s} \right)^{1/3}
$$

Condition B: An exomoon orbit is habitable if Habitable Zone (Inner Boundary) < Moon-Star Distance < Habitable Zone (Outer Boundary) 

$$
a_\mathrm{inner} = \sqrt{\frac{L_*}{4 \pi F_\mathrm{inner}}}
$$

$$
a_\mathrm{outer} = \sqrt{\frac{L_*}{4 \pi F_\mathrm{outer}}}
$$

with 
$$
L_* = 4 \pi R_*^2 \sigma T_*^4
$$

## Numerical Integration (Leapfrog Algorithm)

For every object, the code integrates orbits using the velocity Verlet (leapfrog) scheme:

For each time step:

1. Advance position by half-step:

$$
\mathbf{x}_{i+1/2} = \mathbf{x}_i + \frac{1}{2} \mathbf{v}_i \, \Delta t
$$

2. Update velocity by full-step (acceleration is sum of pairwise gravitation):

$$
\mathbf{v}_{i+1} = \mathbf{v}_i + \mathbf{a}_{i+1/2} \, \Delta t
$$

3. Advance position by half-step:

$$
\mathbf{x}_{i+1} = \mathbf{x}_{i+1/2} + \frac{1}{2} \mathbf{v}_{i+1} \, \Delta t
$$

4. Accelerations are calculated via

$$
\mathbf{a}_{ij} = -\mu_j \frac{\mathbf{r}_{ij}}{|\mathbf{r}_{ij}|^3}
$$

for each interacting pair

## Building and Testing the MCP Server with Claude Desktop 

1. With VSCode installed, run:
   - Windows: 'code $env:AppData\Claude\claude_desktop_config.json'
   - MacOS: 'code ~/Library/Application\ Support/Claude/claude_desktop_config.json'
  
2. Setup the MCP Server Configuration
   ```
   {
    "mcpServers": {
    "exomoon": {
            "command": "C:\\Path\\to\\Python\\Installation",
            "args": ["-m", "exomoon.mcp_server"],
            "env": {
                "PYTHONPATH": "\\Absolute\\Path\\to\\Parent\\Folder"
            }
        }
    }
}
```

Start the MCP server:
python mcp_server.py

Use Claude or compatible LLM/agent to invoke MCP tools with input parameters.

Visualize the system or override settings via the provided Dash UI (/outputs/exomoon_sim.html).

## Output
HTML/Plotly animation of the system’s evolution

Numerical summary of stability, escape times, and orbital diagnostics

Option to analyze or download output via Claude or from storage


