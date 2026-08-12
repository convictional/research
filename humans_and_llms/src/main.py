import os
import numpy as np
import json

from .settings import settings

# --- Model Parameters (D is fixed) ---
D_fixed = 1000.0


# Python model functions (ensure they are defined in the session)
def calculate_H_total_py(T, D, Phi):
    if isinstance(Phi, (np.ndarray)):
        Phi_safe = Phi.copy()
        Phi_safe[Phi_safe <= 1e-9] = 1e-9
    elif Phi <= 1e-9:
        Phi_safe = 1e-9
    else:
        Phi_safe = Phi
    term1 = (1 - T) ** 2 * D
    term2 = (D * T * (1 - T)) / Phi_safe
    return term1 + term2


def calculate_A_total_py(T, D, p_A):
    if isinstance(p_A, (np.ndarray)):
        p_A_safe = p_A.copy()
        p_A_safe[p_A_safe <= 1e-9] = 1e-9
    elif p_A <= 1e-9:
        p_A_safe = 1e-9
    else:
        p_A_safe = p_A
    return (D / p_A_safe) * T * (2 - T)


def calculate_ratio_A_H_py(A_total_val, H_total_val, cap=100):
    A_total_val = np.asarray(A_total_val)
    H_total_val = np.asarray(H_total_val)
    ratio = np.full_like(H_total_val, fill_value=np.nan, dtype=float)

    # Ensure H_total_val is float for division, handle potential 0s by replacing with small number
    H_total_val_safe = np.where(np.abs(H_total_val) <= 1e-9, 1e-9, H_total_val)

    idx_h_pos_original = H_total_val > 1e-9  # Where original H_total was positive

    # Calculate ratio using safe H_total, then apply logic based on original H_total
    ratio_temp = A_total_val / H_total_val_safe

    # Case 1: H_total is positive
    ratio[idx_h_pos_original] = ratio_temp[idx_h_pos_original]

    # Case 2: H_total is zero or very small
    idx_h_zero_original = ~idx_h_pos_original
    idx_a_zero = np.abs(A_total_val) < 1e-9

    ratio[idx_h_zero_original & idx_a_zero] = 0.0  # 0/0 ~ 0
    ratio[idx_h_zero_original & ~idx_a_zero] = cap  # A > 0, H = 0 ~ cap (A < 0, H=0 would be -cap)
    # Handle A < 0, H = 0 if necessary
    neg_A_H_zero_idx = idx_h_zero_original & ~idx_a_zero & (A_total_val < 0)
    ratio[neg_A_H_zero_idx] = -cap

    # Apply cap to all valid ratios
    ratio[ratio > cap] = cap
    ratio[ratio < -cap] = -cap

    ratio[np.isnan(ratio)] = 0  # Catch any other NaNs
    return ratio


# --- Parameters for Sliders and Initial Plot ---
# Plot 1
Phi_default_plot1 = 5.0
p_A_default_plot1 = 1.0
# Plot 2
p_A_default_plot2 = 1.0
# Plot 3
Phi_default_plot3 = 5.0
# Plot 4
p_A_default_plot4 = 1.0  # This will be the slider for plot 4

# General parameter ranges
Phi_min = 1.0
Phi_max = 20.0
Phi_step = 1.0
p_A_min = 0.2
p_A_max = 5.0
p_A_step = 0.1

# Trust (T)
T_min_model = 0.01
T_max_model = 0.99
T_points_plot1 = 50
T_points_3d = 25  # For T-axis in 3D plots
Phi_points_3d = 20  # For Phi-axis
p_A_points_3d = 20  # For p_A-axis

# --- Evolutionary Path Functions ---
# Defines different functions for how trust, review capability, and productivity evolve over time

# Time parameters
time_steps = 100  # Number of time steps for evolutionary paths
time_values = np.linspace(0, 1, time_steps)  # Normalized time from 0 to 1


# Trust evolution functions
def trust_linear(t_values):
    """Linear trust growth: 0.01 to 0.99 over time"""
    return T_min_model + t_values * (T_max_model - T_min_model)


def trust_accelerating(t_values, power=2):
    """Accelerating trust growth: slower at first, then faster (polynomial)"""
    return T_min_model + (t_values**power) * (T_max_model - T_min_model)


def trust_oscillating(t_values, amplitude=0.15, frequency=6):
    """Oscillating trust with overall upward trend"""
    base = trust_linear(t_values)  # Linear base trend
    oscillation = amplitude * np.sin(frequency * np.pi * t_values)  # Oscillating component
    result = base + oscillation
    # Ensure values stay within bounds
    return np.clip(result, T_min_model, T_max_model)


def trust_with_incidents(
    t_values, incident_times=[0.3, 0.6, 0.8], severity=[0.2, 0.1, 0.15], recovery_rate=[15, 20, 10]
):
    """Trust with sudden pullbacks (incidents) followed by recovery"""
    base = trust_accelerating(t_values, power=1.5)  # Slightly accelerating base

    # Apply each incident as a negative shock followed by recovery
    for idx, incident_time in enumerate(incident_times):
        if idx < len(severity) and idx < len(recovery_rate):
            # Calculate the impact of each incident
            time_since_incident = np.maximum(0, t_values - incident_time)
            incident_effect = -severity[idx] * np.exp(-recovery_rate[idx] * time_since_incident)

            # Only apply the effect after the incident occurs
            incident_effect = np.where(t_values >= incident_time, incident_effect, 0)
            base += incident_effect

    # Ensure values stay within bounds
    return np.clip(base, T_min_model, T_max_model)


# Review capability evolution functions
def review_capability_static(t_values, value=5.0):
    """Static review capability over time"""
    return np.full_like(t_values, value)


def review_capability_decreasing(t_values, start=10.0, end=2.0, inflection=0.7):
    """Decreasing review capability (sigmoid shape) - as models become more complex"""
    # Using logistic function for smooth transition
    k = -10  # Steepness of the transition (negative for decreasing)
    x0 = inflection  # Inflection point
    return start + (end - start) / (1 + np.exp(k * (t_values - x0)))


def review_capability_increasing(t_values, start=3.0, end=15.0, power=1.5):
    """Increasing review capability as tools improve"""
    return start + (end - start) * (t_values**power)


# Productivity evolution functions
def productivity_static(t_values, value=1.0):
    """Static AI productivity over time"""
    return np.full_like(t_values, value)


def productivity_increasing(t_values, start=0.5, end=3.0, power=2):
    """Increasing AI productivity (polynomial)"""
    return start + (end - start) * (t_values**power)


def productivity_decreasing(t_values, start=2.0, end=0.5):
    """Decreasing AI productivity - as more complex tasks are assigned"""
    return start + (end - start) * t_values


# --- Data for Plot 1 ---
T_values_plot1 = np.linspace(T_min_model, T_max_model, T_points_plot1)
H_initial_plot1 = calculate_H_total_py(T_values_plot1, D_fixed, Phi_default_plot1)
A_initial_plot1 = calculate_A_total_py(T_values_plot1, D_fixed, p_A_default_plot1)

# --- Data for Plot 2 (H_total vs T, Phi; color by A/H ratio dep. on p_A_slider_plot2) ---
T_values_3d_plot2 = np.linspace(T_min_model, T_max_model, T_points_3d)
Phi_values_3d_plot2 = np.linspace(Phi_min, Phi_max, Phi_points_3d)
T_grid_plot2, Phi_grid_plot2 = np.meshgrid(T_values_3d_plot2, Phi_values_3d_plot2)
H_surface_initial_plot2 = calculate_H_total_py(T_grid_plot2, D_fixed, Phi_grid_plot2)
A_surface_for_ratio_plot2 = calculate_A_total_py(T_grid_plot2, D_fixed, p_A_default_plot2)
Ratio_surface_initial_plot2 = calculate_ratio_A_H_py(A_surface_for_ratio_plot2, H_surface_initial_plot2)

# --- Data for Plot 3 (A_total vs T, p_A; color by A/H ratio dep. on Phi_slider_plot3) ---
T_values_3d_plot3 = np.linspace(T_min_model, T_max_model, T_points_3d)
p_A_values_3d_plot3 = np.linspace(p_A_min, p_A_max, p_A_points_3d)
T_grid_plot3, p_A_grid_plot3 = np.meshgrid(T_values_3d_plot3, p_A_values_3d_plot3)
A_surface_initial_plot3 = calculate_A_total_py(T_grid_plot3, D_fixed, p_A_grid_plot3)
H_for_ratio_plot3 = calculate_H_total_py(T_grid_plot3, D_fixed, Phi_default_plot3)  # Phi is slider
Ratio_surface_initial_plot3 = calculate_ratio_A_H_py(A_surface_initial_plot3, H_for_ratio_plot3)

# --- Data for Plot 4 (Parametric: H vs T, A_total/H_total; color by Phi; p_A is slider) ---
# Underlying grid parameters for Plot 4: T and Phi
T_values_parametric_plot4 = np.linspace(T_min_model, T_max_model, T_points_3d)
Phi_values_parametric_plot4 = np.linspace(Phi_min, Phi_max, Phi_points_3d)
T_grid_parametric_plot4, Phi_grid_parametric_plot4 = np.meshgrid(
    T_values_parametric_plot4, Phi_values_parametric_plot4
)

# Calculate X, Z, and Color for Plot 4 surface
X_surface_plot4 = T_grid_parametric_plot4  # X-coordinate is Trust (from T/Phi grid)
Z_surface_plot4 = calculate_H_total_py(
    T_grid_parametric_plot4, D_fixed, Phi_grid_parametric_plot4
)  # Z-coordinate is H_total (from T/Phi grid)
Color_surface_plot4 = Phi_grid_parametric_plot4  # Color by Phi (from T/Phi grid)

# Calculate initial A_total based on default p_A for Plot 4
A_total_initial_plot4 = calculate_A_total_py(T_grid_parametric_plot4, D_fixed, p_A_default_plot4)

# **MODIFIED**: Y-coordinate is now Ratio A_total / H_total
Y_surface_plot4_ratio = calculate_ratio_A_H_py(A_total_initial_plot4, Z_surface_plot4)


# --- Evolutionary Paths Calculations ---
# Each evolutionary path consists of:
# 1. Time series of trust, review capability, and productivity values
# 2. The resulting human workforce and AI agents at each time step
# 3. The 3D coordinates for visualization on our existing surfaces


# Function to calculate evolutionary path for various scenarios
def calculate_evolutionary_path(trust_func, phi_func, p_a_func, name):
    """Calculate an evolutionary path based on specified functions for trust, review capability, and productivity."""
    # Calculate parameter values over time
    trust_values = trust_func(time_values)
    phi_values = phi_func(time_values)
    p_a_values = p_a_func(time_values)

    # Calculate resulting workforce values
    human_workforce = calculate_H_total_py(trust_values, D_fixed, phi_values)
    ai_agents = calculate_A_total_py(trust_values, D_fixed, p_a_values)
    ai_human_ratio = calculate_ratio_A_H_py(ai_agents, human_workforce)

    return {
        "name": name,
        "time": time_values,
        "trust": trust_values,
        "phi": phi_values,
        "p_a": p_a_values,
        "human_workforce": human_workforce,
        "ai_agents": ai_agents,
        "ai_human_ratio": ai_human_ratio,
    }


# Define the evolutionary paths for our scenarios
evolutionary_paths = [
    # Scenario 1: Accelerating trust with static review capability
    calculate_evolutionary_path(
        trust_func=lambda t: trust_accelerating(t, power=2),
        phi_func=lambda t: review_capability_static(t, value=5.0),
        p_a_func=lambda t: productivity_static(t, value=1.0),
        name="Accelerating Trust × Static Review Capability",
    ),
    # Scenario 2: Accelerating trust with decreasing review capability
    calculate_evolutionary_path(
        trust_func=lambda t: trust_accelerating(t, power=2),
        phi_func=lambda t: review_capability_decreasing(t, start=10.0, end=2.0),
        p_a_func=lambda t: productivity_static(t, value=1.0),
        name="Accelerating Trust × Decreasing Review Capability",
    ),
    # Scenario 3: Oscillating trust with static review capability
    calculate_evolutionary_path(
        trust_func=trust_oscillating,
        phi_func=lambda t: review_capability_static(t, value=5.0),
        p_a_func=lambda t: productivity_static(t, value=1.0),
        name="Oscillating Trust × Static Review Capability",
    ),
    # Scenario 4: Linear trust with decreasing productivity
    calculate_evolutionary_path(
        trust_func=trust_linear,
        phi_func=lambda t: review_capability_static(t, value=5.0),
        p_a_func=productivity_decreasing,
        name="Linear Trust × Decreasing Productivity",
    ),
    # Scenario 5: Trust with incidents
    calculate_evolutionary_path(
        trust_func=trust_with_incidents,
        phi_func=lambda t: review_capability_static(t, value=5.0),
        p_a_func=lambda t: productivity_static(t, value=1.0),
        name="Trust with Incidents × Static Review Capability",
    ),
]

# --- HTML and JavaScript Generation ---
html_content_combined = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>The Human-AI Workforce Model</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; background-color: #eef1f5; color: #333; display: flex; flex-direction: column; align-items: center; padding-bottom: 50px; }}
        .main-container {{ width: 90%; max-width: 1200px; margin-top: 20px; }}
        .intro-text, .plot-description {{ background-color: #fff; padding: 25px; margin-bottom: 30px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.05); line-height: 1.6; font-size: 16px; }}
        .intro-text h1, .intro-text h2, .plot-description h2 {{ color: #2c3e50; margin-top: 0; }}
        .intro-text h1 {{ text-align: center; font-size: 28px; margin-bottom: 20px; }}
        .intro-text h2 {{ font-size: 22px; margin-bottom: 15px; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
        .intro-text ul, .plot-description ul {{ list-style-type: disc; margin-left: 20px; padding-left: 0px; }}
        .intro-text code {{ background-color: #f4f4f4; padding: 2px 5px; border-radius: 4px; font-family: Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace;}}

        .plot-section {{ background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.05); margin-bottom: 40px; }}
        .controls {{
            margin-bottom: 25px;
            padding: 20px;
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 8px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px 30px;
            align-items: center;
        }}
        .controls label {{ font-weight: 600; color: #444; font-size: 15px; }}
        .controls input[type="range"] {{ width: 100%; max-width: 200px; cursor: pointer; accent-color: #3498db; }}
        .controls span {{ font-weight: 500; color: #3498db; min-width: 40px; display: inline-block; font-size: 15px; }}
        .plotDiv {{ width: 100%; height: 500px; }} /* Default height for 2D plots */
        .plotDiv3D {{ width: 100%; height: 600px; }} /* Taller height for 3D plots */

        /* Evolutionary paths section styles */
        .evolution-scenario {{ margin-bottom: 40px; background-color: #f9f9f9; padding: 20px; border-radius: 8px; border: 1px solid #ddd; }}
        .evolution-scenario h3 {{ color: #2c3e50; margin-top: 0; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
        .evolution-plots-container {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 20px; }}
        .evolution-plot {{ height: 300px; background-color: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border-radius: 4px; }}
        .evolution-plot3D {{ height: 400px; background-color: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border-radius: 4px; }}

        .math-formula {{
            font-family: 'CMU Serif', 'Latin Modern Math', 'STIX Two Math', 'Times New Roman', Times, serif;
            font-style: normal;
            margin: 20px 0;
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 3px;
            border-left: 3px solid #ddd;
            text-align: center;
            font-size: 1.2em;
            line-height: 1.5;
            letter-spacing: 0.5px;
        }}

        /* LaTeX-style fraction */
        .frac {{
            display: inline-block;
            position: relative;
            vertical-align: middle;
            margin: 0 0.2em;
            text-align: center;
            font-size: 0.9em;
        }}
        .frac-num {{
            display: block;
            padding: 0.1em;
            border-bottom: 1px solid;
        }}
        .frac-denom {{
            display: block;
            padding: 0.1em;
            margin-top: 1px;
        }}
    </style>
</head>
<body>
    <div class="main-container">
        <div class="intro-text">
            <h1>The Human-AI Workforce: How Trust and AI Capability Could Reshape Our Teams</h1>
            <p>Artificial Intelligence, particularly advanced Large Language Models (LLMs) that can act as "agents," is rapidly evolving. Many of us are wondering: how will this technology change the way we work? If AI can handle more tasks, what does that mean for human jobs?</p>
            <p>This post explores a simple mathematical model to help us think about these questions. Instead of predicting one specific future, we want to provide a way for you to explore different <em>possibilities</em>. How might the number of human workers and AI agents change as our trust in AI grows and its capabilities improve?</p>

            <h2>Our Simple Model: The Core Ideas</h2>
            <p>We're making a few key assumptions to build our model:</p>
            <ol>
                <li><strong>Total Work (Demand):</strong> We assume there's a fixed amount of work to be done. For this exploration, we've set this at <strong>1000 units of work</strong>. Think of this as the baseline number of people that would be needed if no AI was involved. This fixed number helps us see the relative impact of other changing factors.</li>
                <li><strong>Trust is Key (T):</strong> The biggest factor is how much we <em>trust</em> AI to do tasks correctly and reliably. We represent "Trust in AI" (<code>T</code>) on a scale from near 0 (no trust, 0.01 in our charts) to near 1 (full trust, 0.99 in our charts).</li>
                <li><strong>Three Tiers of AI Use:</strong> Based on this trust, work gets divided:
                    <ul>
                        <li><strong>Fully Automated Tasks (AI Does It Alone):</strong> When trust is high, some tasks (<code>T * Demand</code>) are given entirely to AI agents.</li>
                        <li><strong>Human-Only Tasks (Humans Do It Alone):</strong> When trust is very low for certain tasks, or for tasks where AI isn't suitable, humans do the work directly. We model this as <code>(1-T)<sup>2</sup> * Demand</code>. The squared term here was chosen simply for the sake of symmetry; here we are effectively saying that of the (1-T) tasks that are not automated via Agents, (1-T) of those are entirely completed by humans. In reality, these tiers would be far more complex (see the limitations section below for more on this).</li>
                        <li><strong>AI-Assisted Tasks (Humans Review AI Work):</strong> This is the middle ground. AI does the initial work, but a human reviews, guides, and approves it. The amount of work here is <code>Demand * T * (1-T)</code>.</li>
                    </ul>
                </li>
                <li><strong>Human Review Power (Φ):</strong> When humans shift from <em>doing</em> tasks to <em>reviewing</em> AI-generated work, they can often oversee multiple AI outputs. "Human Review Power" (<code>Φ</code>) represents how many AI-completed tasks one person can effectively review and guide in the time it would have taken them to do one task manually. A <code>Φ</code> of 5 means one human can oversee 5 AI agents/tasks.</li>
                <li><strong>AI Agent Speed/Efficiency (p<sub>A</sub>):</strong> This parameter, "AI Agent Speed/Efficiency" (<code>p<sub>A</sub></code>), tells us how productive an AI agent is. If <code>p<sub>A</sub> = 2</code>, one AI agent can do the work of two humans (or twice as fast).</li>
            </ol>

            <h2>The Math (Simplified)</h2>
            <p>For a total demand <code>D</code> (fixed at 1000 in our interactive charts):</p>
            <div class="math-formula">Humans Needed (H<sub>total</sub>) = D(1-T)(1-T(1+<span class="frac"><span class="frac-num">1</span><span class="frac-denom">Φ</span></span>))</div>
            <div class="math-formula">AI Agents Deployed (A<sub>total</sub>) = <span class="frac"><span class="frac-num">DT(2-T)</span><span class="frac-denom">p<sub>A</sub></span></span></div>

            <h2>Important Limitations</h2>
            <p>This is a simplified model! It doesn't capture everything:</p>
            <ul>
                <li>It doesn't account for new jobs created by AI (e.g., AI trainers, ethicists, AI system builders).</li>
                <li>It assumes demand (<code>D</code>) is constant. AI could increase demand or create new types of demand.</li>
                <li>"Trust" (<code>T</code>) is a single number, but in reality, it's complex and task-specific.</li>
                <li>The parameters (<code>Φ</code>, <code>p<sub>A</sub></code>) might change as AI evolves or as we get better at working with it.</li>
            </ul>
            <p><strong>On Employment Impact Predictions:</strong> This model does not make predictions about overall human employment reduction or job displacement. The AI employment debate spans from optimists predicting new role creation and demand expansion to pessimists forecasting mass unemployment (10-20%) and widespread entry-level job loss within 2-3 years. We take a pragmatic stance: we are not economists and cannot speak to demand elasticity or economic responses to increased AI supply. However, using these first principles, we believe human knowledge work composition will change considerably - the <em>nature</em> of human roles may shift dramatically even if total employment levels remain debated.</p>
            <p><strong>Assumptions about Task Allocation:</strong> The way tasks are divided between full AI automation, AI-assistance with human review, and human-only work is based on simplified functions of Trust (T) – specifically, the <code>(1-T)</code> factor driving human-involved work and the <code>T(2-T)</code> factor for AI-driven work. These specific mathematical forms (<code>H_total = D(1-T)(1 + 1/Φ)</code> and <code>A_total = DT(2-T)/P_A</code>) were chosen for their illustrative properties and tractability within this model. In reality, how tasks are allocated as trust evolves is far more complex and likely not static or describable by such simple universal ratios. For instance, the current model structure implies specific symmetries in task distribution based on trust. A more granular model might introduce additional parameters to capture these nuances, but this would significantly increase complexity and move beyond the scope of this exploratory post. Our aim here is to provide a foundational model to spark thinking, not to perfectly predict the future.</p>
            <p>Despite these limitations, the model can help us see potential trends and understand the interplay between these crucial factors. Let's explore!</p>
        </div>

        <!-- Plot 1 Section -->
        <div class="plot-section">
            <div class="plot-description">
                <h2>Plot 1: Workforce Dynamics as Trust in AI Grows</h2>
                <p>This first chart shows how the "Human Workforce Needed" (black line) and "AI Agents Deployed" (yellow line) might change as "Trust in AI" (shown on the horizontal axis) increases from nearly none (0.01) to almost full (0.99). Remember, the total demand for work is fixed at 1000 units.</p>
                <p>Use the sliders below to see how things change:</p>
                <ul>
                    <li><strong>Human Review Power (Φ):</strong> If you look at the formula for total human workforce again, you might notice the 1/Φ term, which is the only place human review capability comes into play. This means that when a single human can review more than 1 agent's work, the number of humans in the workforce becomes almost entirely determined by Trust. This means that you may not see much change, particularly as Φ starts get large (e.g. > 5)</li>
                    <li><strong>AI Agent Speed/Efficiency (p<sub>A</sub>):</strong> Again, reviewing our formulas for total humans and agents above, you might also notice that p<sub>A</sub> doesn't appear in the human workforce calculation, implying that the efficiency of agents doesn't change how many humans are needed to oversee and work with them. This conclusion actually draws implicitly on our assumption that the cost of agents approaches the cost of energy, making them negligble in cost compared to an equivalent human expert - without this assumption, would require a parameterization of cost, in which case agent efficiency may tip the scales in favour of humans when it is very low. In our model, agent speed/efficiency only determines how many agents are ultimately needed given a trust, T.</li>
                </ul>
            </div>
            <div class="controls">
                <div><label for="Phi_slider_plot1">Human Review Power (Φ):</label><input type="range" id="Phi_slider_plot1" min="{Phi_min}" max="{Phi_max}" value="{Phi_default_plot1}" step="{Phi_step}"><span id="Phi_value_plot1">{Phi_default_plot1}</span></div>
                <div><label for="p_A_slider_plot1">AI Agent Speed/Efficiency (p<sub>A</sub>):</label><input type="range" id="p_A_slider_plot1" min="{p_A_min}" max="{p_A_max}" value="{p_A_default_plot1}" step="{p_A_step}"><span id="p_A_value_plot1">{p_A_default_plot1}</span></div>
            </div>
            <div id="plot1Div" class="plotDiv"></div>
        </div>

        <!-- Plot 2 Section -->
        <div class="plot-section">
            <div class="plot-description">
                <h2>Plot 2: Mapping the Human Workforce Landscape</h2>
                <p>This second visualization is a 3D surface plot showing "Human Workforce Needed" (height) vs. "Trust in AI" (x-axis) and "Human Review Power" (y-axis). The surface color indicates the "AI-to-Human Ratio," influenced by the "AI Agent Speed/Efficiency" (<code>p<sub>A</sub></code>) slider.</p>
                 <ul>
                    <li>Purples/cool colors indicate a lower ratio (fewer AI agents per human).</li>
                    <li>Yellows/warm colors indicate a higher ratio (more AI agents per human).</li>
                </ul>
                <p>The surface below looks relatively uninteresting, which is again due to that 1/Φ term in the human workforce total. Once Φ is greater than 1, it starts to have very little impact on the human workforce total putting all accountable variance into the Trust variable.</p>
                </ul>
            </div>
            <div class="controls">
                <div><label for="p_A_slider_plot2">AI Agent Speed/Efficiency (p<sub>A</sub>) (affects Plot 2 color):</label><input type="range" id="p_A_slider_plot2" min="{p_A_min}" max="{p_A_max}" value="{p_A_default_plot2}" step="{p_A_step}"><span id="p_A_value_plot2">{p_A_default_plot2}</span></div>
            </div>
            <div id="plot2Div" class="plotDiv3D"></div>
        </div>

        <!-- Plot 3 Section -->
        <div class="plot-section">
            <div class="plot-description">
                <h2>Plot 3: The AI Agent Deployment Landscape</h2>
                <p>This 3D surface plot shows "AI Agents Deployed" (height) vs. "Trust in AI" (x-axis) and "AI Agent Speed/Efficiency" (<code>p<sub>A</sub></code>, y-axis). The surface color represents the "AI-to-Human Ratio," influenced by the "Human Review Power" (<code>Φ</code>) slider.</p>
                 <ul>
                    <li>Purples/cool colors indicate a lower AI-to-Human ratio.</li>
                    <li>Yellow/warm colors indicate a higher ratio.</li>
                </ul>
                <p>As we saw above in the human workforce surface plot, human review power has limited effect on human workforce (and hence agent / human ratio) once it scales beyond 5(ish); which you can see in the changing colors as you adjust the slider.</p>
                <p>The surface itself is relatively 'flat' once agents become more productive than humans as we assume fixed demand. A lot of posts talk about 'fleets' of agents being deployed by organizations - which may be true if agents are either less productive than humans, or demand keeps pace with increasing trust. In reality, demand elasticity may lead to no more agents than humans being deployed in a majority of scenarios. Note, we're assuming intelligent foundation models - effectively 'AGI' and are not talking about ensembles of small models working in consensus or via path exploration. Although these may be facets of any sufficiently intelligent agent, but that is beyond the scope of this post.</p>
            </div>
            <div class="controls">
                <div><label for="Phi_slider_plot3">Human Review Power (Φ) (affects Plot 3 color):</label><input type="range" id="Phi_slider_plot3" min="{Phi_min}" max="{Phi_max}" value="{Phi_default_plot3}" step="{Phi_step}"><span id="Phi_value_plot3">{Phi_default_plot3}</span></div>
            </div>
            <div id="plot3Div" class="plotDiv3D"></div>
        </div>

        <!-- Plot 4 Section -->
        <div class="plot-section">
            <div class="plot-description">
                <!-- **MODIFIED** Plot 4 Description -->
                <h2>Plot 4: Workforce Configurations - Humans vs. Trust and Agents per Human</h2>
                <p>This final 3D surface plot directly visualizes the relationship between the "Human Workforce Needed" (<code>H<sub>total</sub></code>, Z-axis height), "Trust in AI" (<code>T</code>, X-axis), and the "Agents per Human Ratio" (<code>A<sub>total</sub> / H<sub>total</sub></code>, Y-axis).</p>
                <p>The surface itself is generated by exploring different combinations of "Trust" (<code>T</code>) and "Human Review Power" (<code>Φ</code>). The color of the surface corresponds to this underlying "Human Review Power" (<code>Φ</code>) value that generated each point:
                    <ul>
                        <li>Cooler colors (e.g., purples/blues) typically represent lower <code>Φ</code> values (humans can review fewer AI tasks).</li>
                        <li>Warmer colors (e.g., greens/yellows) represent higher <code>Φ</code> values (humans are more leveraged in their review capacity).</li>
                    </ul>
                </p>
                <p>The "AI Agent Speed/Efficiency" (<code>p<sub>A</sub></code>) slider will primarily reshape the surface along the Y-axis (Agents per Human). Higher AI speed means fewer agents are needed for a given workload, which generally decreases the <code>A<sub>total</sub></code> component of the ratio, thus affecting the Y-value. Observe how this parameter stretches or compresses the surface in that dimension.</p>
                <p>This plot allows you to see how different levels of human review power (<code>Φ</code>, shown by color) create different trade-off surfaces. As you adjust the 'AI Agent Speed/Efficiency' (<code>p<sub>A</sub></code>) slider, observe how the 'Agents per Human' ratio (Y-axis) changes, and how that corresponds to the 'Human Workforce Needed' (Z-axis) for any given 'Trust' level (X-axis).</p>
                <p>You might be wondering why we have such a weird shape? The reason is that both agents (and hence agents per human) and humans are functions of (primarily) Trust and so we 'solve' the parameter combinations at each point, leading to the limited surface defined by what is allowed by the model.</p>
            </div>
            <div class="controls">
                <div><label for="p_A_slider_plot4">AI Agent Speed/Efficiency (p<sub>A</sub>) (affects Plot 4 Y-dim & shape):</label><input type="range" id="p_A_slider_plot4" min="{p_A_min}" max="{p_A_max}" value="{p_A_default_plot4}" step="{p_A_step}"><span id="p_A_value_plot4">{p_A_default_plot4}</span></div>
            </div>
            <div id="plot4Div" class="plotDiv3D"></div>
        </div>

        <!-- Evolutionary Paths Section -->
        <div class="plot-section">
            <div class="plot-description">
                <h2>Evolutionary Paths: How Trust and Capability Might Change Over Time</h2>
                <p>The model above gives a hypothetical relationship between trust, productivity and review metrics but says nothing about how trust might evolve. That specifically, is beyond the scope of this post, but we can look at hypothetical functions of time for trust, productivity and review that show how human staffing might change. You can think of this as a path being traced across the surfaces above.</p>
                <p>We may see unpredictable changes such as:</p>
                <ul>
                    <li>Trust increases at first, but sees frequent pull backs due to 'incidents'; these pull-backs could vary in amplitude, 'damping' (e.g. how quickly it pulls back and rebounds) and timing</li>
                    <li>Trust increases in an accelerated way, such as polynomial, exponential, etc.</li>
                    <li>Trust evolves in a sinusoidal pattern, increasing then decreasing back and forth as human trust in agent capability and judgement oscillates while increasing</li>
                    <li>Human review capability may actually decrease over time as model's intelligence surpasses human's leading to longer periods required to review an agent's work</li>
                    <li>On the other hand, new tools may be made available allowing one human to review more agent's work even as the complexity and stakes of tasks increase</li>
                </ul>
                <p>Below we show different potential path parameterizations of trust, review capability, and productivity. For each scenario, we display:</p>
                <ol>
                    <li>A 2D graph showing how parameters (trust, review capability, etc.) evolve over time</li>
                    <li>A 2D graph showing how workforce composition (humans and AI agents) changes as a result</li>
                    <li>A 3D visualization of the path on our model's surface, showing how the system traverses the parameter space</li>
                </ol>
            </div>

            <!-- Evolution Scenario 1 -->
            <div class="evolution-scenario">
                <h3>Accelerating Trust × Static Review Capability</h3>
                <p>In this scenario, trust in AI accelerates over time (growing slowly at first, then more rapidly), while human review capability remains constant. This represents a world where people become increasingly comfortable with AI systems as they prove themselves, but the fundamental human ability to review AI work doesn't change significantly.</p>
                <div class="evolution-plots-container">
                    <div id="path1_params_div" class="evolution-plot"></div>
                    <div id="path1_workforce_div" class="evolution-plot"></div>
                    <div id="path1_3d_div" class="evolution-plot3D"></div>
                </div>
            </div>

            <!-- Evolution Scenario 2 -->
            <div class="evolution-scenario">
                <h3>Accelerating Trust × Decreasing Review Capability</h3>
                <p>Here, trust accelerates as in the previous scenario, but human review capability decreases over time. This might occur if AI systems become increasingly complex and sophisticated, making their work harder for humans to effectively evaluate. At a certain point, reviewing AI output becomes more cognitively demanding than doing the work directly.</p>
                <div class="evolution-plots-container">
                    <div id="path2_params_div" class="evolution-plot"></div>
                    <div id="path2_workforce_div" class="evolution-plot"></div>
                    <div id="path2_3d_div" class="evolution-plot3D"></div>
                </div>
            </div>

            <!-- Evolution Scenario 3 -->
            <div class="evolution-scenario">
                <h3>Oscillating Trust × Static Review Capability</h3>
                <p>This scenario models trust that follows a cyclical pattern while overall increasing. The oscillations represent periodic "trust crises" where AI incidents or failures temporarily reduce trust, followed by recovery periods where trust rebuilds. This pattern acknowledges the reality that progress often isn't linear, especially with emerging technologies.</p>
                <div class="evolution-plots-container">
                    <div id="path3_params_div" class="evolution-plot"></div>
                    <div id="path3_workforce_div" class="evolution-plot"></div>
                    <div id="path3_3d_div" class="evolution-plot3D"></div>
                </div>
            </div>

            <!-- Evolution Scenario 4 -->
            <div class="evolution-scenario">
                <h3>Linear Trust × Decreasing Productivity</h3>
                <p>In this scenario, trust increases linearly, but AI productivity decreases over time. This counterintuitive situation might emerge if AI systems are initially applied to "easy" tasks where they excel, but are gradually tasked with more complex work where their relative advantage over humans diminishes.</p>
                <div class="evolution-plots-container">
                    <div id="path4_params_div" class="evolution-plot"></div>
                    <div id="path4_workforce_div" class="evolution-plot"></div>
                    <div id="path4_3d_div" class="evolution-plot3D"></div>
                </div>
            </div>

            <!-- Evolution Scenario 5 -->
            <div class="evolution-scenario">
                <h3>Trust with Incidents × Static Review Capability</h3>
                <p>This final scenario models trust that generally increases but experiences sudden dramatic drops following specific "incidents" - major AI failures or errors that damage trust. Each incident causes a sharp drop followed by a recovery period. This pattern reflects how catastrophic events can cause lasting but not permanent shifts in trust.</p>
                <div class="evolution-plots-container">
                    <div id="path5_params_div" class="evolution-plot"></div>
                    <div id="path5_workforce_div" class="evolution-plot"></div>
                    <div id="path5_3d_div" class="evolution-plot3D"></div>
                </div>
            </div>
        </div>
    </div>

    <script src="llms_and_humans.js"></script>
</body>
</html>
"""

# JavaScript content
javascript_content = f"""
        const D_fixed_js = {D_fixed};
        const ratio_cap_js = 100; // Cap for A/H ratio color scales

        function calculate_H_total_js(T, D, Phi) {{
            let Phi_safe = Phi;
            if (Phi_safe <= 1e-9) Phi_safe = 1e-9;

            if (Array.isArray(T)) {{
                return T.map(t_val => {{
                    const term1 = Math.pow(1 - t_val, 2) * D;
                    const term2 = (D * t_val * (1 - t_val)) / Phi_safe;
                    return term1 + term2;
                }});
            }} else {{
                const term1 = Math.pow(1 - T, 2) * D;
                const term2 = (D * T * (1 - T)) / Phi_safe;
                return term1 + term2;
            }}
        }}

        function calculate_A_total_js(T, D, p_A) {{
            let p_A_safe = p_A;
            if (p_A_safe <= 1e-9) p_A_safe = 1e-9;

            if (Array.isArray(T)) {{
                return T.map(t_val => (D / p_A_safe) * t_val * (2 - t_val));
            }} else {{
                return (D / p_A_safe) * T * (2 - T);
            }}
        }}

        function calculate_ratio_A_H_js(A_total_val, H_total_val, cap) {{
            let ratio;
            if (H_total_val > 1e-9) {{ // H_total is positive
                ratio = A_total_val / H_total_val;
            }} else {{ // H_total is zero or very small
                if (Math.abs(A_total_val) < 1e-9) {{ // A_total is also zero
                    ratio = 0.0; // 0/0 ~ 0
                }} else {{ // A_total is non-zero
                    ratio = (A_total_val > 0) ? cap : -cap; // Assign cap or -cap based on A_total's sign
                }}
            }}
            return Math.min(Math.max(ratio, -cap), cap); // Apply cap
        }}

        // --- Plot 1 ---
        const T_values_plot1_js = {json.dumps(T_values_plot1.tolist())};
        let H_initial_plot1_js = {json.dumps(H_initial_plot1.tolist())};
        let A_initial_plot1_js = {json.dumps(A_initial_plot1.tolist())};
        const plot1Div = document.getElementById('plot1Div');
        const Phi_slider_plot1 = document.getElementById('Phi_slider_plot1');
        const p_A_slider_plot1 = document.getElementById('p_A_slider_plot1');
        const Phi_value_plot1_display = document.getElementById('Phi_value_plot1');
        const p_A_value_plot1_display = document.getElementById('p_A_value_plot1');
        const trace_H_plot1 = {{ x: T_values_plot1_js, y: H_initial_plot1_js, mode: 'lines', name: 'Human Workforce (H<sub>total</sub>)', line: {{ color: '#2C3E50', width: 2.5 }} }};
        const trace_A_plot1 = {{ x: T_values_plot1_js, y: A_initial_plot1_js, mode: 'lines', name: 'AI Agents (A<sub>total</sub>)', line: {{ color: '#FDE725', width: 2.5 }} }};
        const layout_plot1 = {{ xaxis: {{ title: 'Trust (T)', font:{{size:12}}}}, yaxis: {{ title: 'Number of Workers / Agents', rangemode: 'tozero', font:{{size:12}}}}, margin: {{t:20,b:60,l:80,r:30}}, legend: {{x:0.5,y:1.1,orientation:'h',xanchor:'center', font:{{size:12}}}}, hovermode:'x unified'}};
        Plotly.newPlot(plot1Div, [trace_H_plot1, trace_A_plot1], layout_plot1);

        function updatePlot1() {{
            const Phi = parseFloat(Phi_slider_plot1.value);
            const p_A = parseFloat(p_A_slider_plot1.value);
            Phi_value_plot1_display.textContent = Phi.toFixed(1);
            p_A_value_plot1_display.textContent = p_A.toFixed(1);
            Plotly.react(plot1Div, [
                {{ ...trace_H_plot1, y: calculate_H_total_js(T_values_plot1_js, D_fixed_js, Phi) }},
                {{ ...trace_A_plot1, y: calculate_A_total_js(T_values_plot1_js, D_fixed_js, p_A) }}
            ], layout_plot1);
        }}
        Phi_slider_plot1.addEventListener('input', updatePlot1);
        p_A_slider_plot1.addEventListener('input', updatePlot1);

        // --- Plot 2 ---
        const T_values_3d_plot2_js = {json.dumps(T_values_3d_plot2.tolist())};
        const Phi_values_3d_plot2_js = {json.dumps(Phi_values_3d_plot2.tolist())};
        const H_surface_initial_plot2_js = {json.dumps(H_surface_initial_plot2.tolist())}; // This is Z
        let Ratio_surface_initial_plot2_js = {
    json.dumps(Ratio_surface_initial_plot2.tolist())
}; // This is surfacecolor
        const T_grid_plot2 = {json.dumps(T_grid_plot2.tolist())}; // Add T_grid for updatePlot2 function
        const plot2Div = document.getElementById('plot2Div');
        const p_A_slider_plot2 = document.getElementById('p_A_slider_plot2');
        const p_A_value_plot2_display = document.getElementById('p_A_value_plot2');
        // H_surface_initial_plot2_js provides Z values. T_values for X, Phi_values for Y.
        const trace_surface_plot2 = {{
            type: 'surface',
            x: T_values_3d_plot2_js, // T-axis (columns of the grid)
            y: Phi_values_3d_plot2_js, // Phi-axis (rows of the grid)
            z: H_surface_initial_plot2_js,
            surfacecolor: Ratio_surface_initial_plot2_js,
            colorscale: 'Viridis', cmin:0, cmax: ratio_cap_js/5, // Adjusted cap for better visual spread
            colorbar:{{title:'AI/Human Ratio', titleside:'right'}}
        }};
        const layout_plot2 = {{ title: {{text:'Human Workforce (Z) vs. Trust (X) & Review Power (Y)', font:{{size:16}} }}, scene: {{ xaxis:{{title:'Trust (T)'}}, yaxis:{{title:'Human Review Power (Φ)'}}, zaxis:{{title:'Human Workforce (H<sub>total</sub>)'}} }}, margin:{{t:50,b:0,l:0,r:0}} }};
        Plotly.newPlot(plot2Div, [trace_surface_plot2], layout_plot2);

        function updatePlot2() {{
            const p_A = parseFloat(p_A_slider_plot2.value);
            p_A_value_plot2_display.textContent = p_A.toFixed(1);
            // H_surface_initial_plot2_js is a 2D array (Phi_grid rows, T_grid cols)
            // T_values_3d_plot2_js is 1D array for T, Phi_values_3d_plot2_js is 1D for Phi
            // T_grid_plot2 (from python) has shape (Phi_points, T_points)
            // H_surface_initial_plot2 has shape (Phi_points, T_points)
            const new_Ratio_surface = H_surface_initial_plot2_js.map((h_row, i_phi) => {{ // i_phi iterates over Phi dimension
                const t_row_for_A = T_grid_plot2[i_phi]; // T values for this Phi row
                const a_values_row = calculate_A_total_js(t_row_for_A, D_fixed_js, p_A);
                return h_row.map((h_val, j_t) => {{ // j_t iterates over T dimension
                    return calculate_ratio_A_H_js(a_values_row[j_t], h_val, ratio_cap_js);
                }});
            }});
            Plotly.react(plot2Div, [{{ ...trace_surface_plot2, surfacecolor: new_Ratio_surface }}], layout_plot2);
        }}
        p_A_slider_plot2.addEventListener('input', updatePlot2);

        // --- Plot 3 ---
        const T_values_3d_plot3_js = {json.dumps(T_values_3d_plot3.tolist())};
        const p_A_values_3d_plot3_js = {json.dumps(p_A_values_3d_plot3.tolist())};
        const A_surface_initial_plot3_js = {json.dumps(A_surface_initial_plot3.tolist())}; // This is Z
        let Ratio_surface_initial_plot3_js = {
    json.dumps(Ratio_surface_initial_plot3.tolist())
}; // This is surfacecolor
        const T_grid_plot3 = {json.dumps(T_grid_plot3.tolist())}; // Add T_grid for updatePlot3 function
        const plot3Div = document.getElementById('plot3Div');
        const Phi_slider_plot3 = document.getElementById('Phi_slider_plot3');
        const Phi_value_plot3_display = document.getElementById('Phi_value_plot3');
        const trace_surface_plot3 = {{
            type: 'surface',
            x: T_values_3d_plot3_js, // T-axis
            y: p_A_values_3d_plot3_js, // p_A-axis
            z: A_surface_initial_plot3_js,
            surfacecolor: Ratio_surface_initial_plot3_js,
            colorscale: 'Viridis', cmin:0, cmax:ratio_cap_js/5,
            colorbar:{{title:'AI/Human Ratio', titleside:'right'}}
        }};
        const layout_plot3 = {{ title: {{text:'AI Agents (Z) vs. Trust (X) & AI Speed (Y)', font:{{size:16}} }}, scene: {{ xaxis:{{title:'Trust (T)'}}, yaxis:{{title:'AI Agent Speed (p<sub>A</sub>)'}}, zaxis:{{title:'AI Agents Deployed (A<sub>total</sub>)'}} }}, margin:{{t:50,b:0,l:0,r:0}} }};
        Plotly.newPlot(plot3Div, [trace_surface_plot3], layout_plot3);

        function updatePlot3() {{
            const Phi = parseFloat(Phi_slider_plot3.value);
            Phi_value_plot3_display.textContent = Phi.toFixed(1);
            // A_surface_initial_plot3_js has shape (p_A_points, T_points)
            // T_grid_plot3 has shape (p_A_points, T_points)
            const new_Ratio_surface = A_surface_initial_plot3_js.map((a_row, i_pA) => {{ // i_pA iterates over p_A dimension
                // For H, T values come from T_grid_plot3, Phi is from slider
                const t_values_for_this_row = T_grid_plot3[i_pA]; // These are the T values for current p_A row
                const h_values_row = calculate_H_total_js(t_values_for_this_row, D_fixed_js, Phi);
                return a_row.map((a_val, j_t) => {{ // j_t iterates over T dimension
                    return calculate_ratio_A_H_js(a_val, h_values_row[j_t], ratio_cap_js);
                }});
            }});
            Plotly.react(plot3Div, [{{ ...trace_surface_plot3, surfacecolor: new_Ratio_surface }}], layout_plot3);
        }}
        Phi_slider_plot3.addEventListener('input', updatePlot3);

        // --- Plot 4 ---
        // **MODIFIED**: Pass T_grid and Z_surface (H_total) to JS for updates
        const T_grid_parametric_plot4_js = {
    json.dumps(T_grid_parametric_plot4.tolist())
}; // Grid of T values (X-coords, also used for A_total calc)
        let X_surface_plot4_js = {json.dumps(X_surface_plot4.tolist())}; // T-values (X-coords for plot)
        let Y_surface_plot4_js = {
    json.dumps(Y_surface_plot4_ratio.tolist())
}; // **MODIFIED**: Initial Ratio A/H values (Y-coords)
        let Z_surface_plot4_js = {
    json.dumps(Z_surface_plot4.tolist())
}; // H_total values (Z-coords, also for ratio denom)
        let Color_surface_plot4_js = {json.dumps(Color_surface_plot4.tolist())}; // Phi values (Color)

        const plot4Div = document.getElementById('plot4Div');
        const p_A_slider_plot4 = document.getElementById('p_A_slider_plot4');
        const p_A_value_plot4_display = document.getElementById('p_A_value_plot4');

        const trace_surface_plot4 = {{
            type: 'surface',
            x: X_surface_plot4_js,
            y: Y_surface_plot4_js, // **MODIFIED**: Now holds Ratio A/H
            z: Z_surface_plot4_js,
            surfacecolor: Color_surface_plot4_js,
            colorscale: 'Viridis',
            cmin: {Phi_min},
            cmax: {Phi_max},
            colorbar: {{ title: 'Human Review Power (Φ)', titleside: 'right' }}
        }};
        // **MODIFIED**: Updated yaxis title
        const layout_plot4 = {{
            title: {{ text: 'Humans (Z) vs. Trust (X) & Agents per Human (Y)', font:{{size:16}} }}, // **MODIFIED** Title
            scene: {{
                xaxis: {{ title: 'Trust (T)', range: [{T_min_model}, {T_max_model}] }},
                yaxis: {{ title: 'Agents per Human (A<sub>total</sub>/H<sub>total</sub>)', autorange: true }}, // **MODIFIED** Y-axis Label
                zaxis: {{ title: 'Human Workforce (H<sub>total</sub>)', autorange: true }}
            }},
            margin: {{ t: 50, b: 0, l: 0, r: 0 }}
        }};
        Plotly.newPlot(plot4Div, [trace_surface_plot4], layout_plot4);

        // **MODIFIED**: updatePlot4 logic
        function updatePlot4() {{
            const p_A = parseFloat(p_A_slider_plot4.value);
            p_A_value_plot4_display.textContent = p_A.toFixed(1);

            // Recalculate Y_surface_plot4_js (Ratio A/H) based on new p_A
            // T_grid_parametric_plot4_js provides the T values for calculating A_total.
            // Z_surface_plot4_js provides the H_total values (denominator for the ratio).
            // These T_grid and Z_surface are shaped (num_phi_points, num_t_points)

            const new_Y_surface_plot4_ratio = [];
            for (let i = 0; i < T_grid_parametric_plot4_js.length; i++) {{ // Iterates over Phi dimension
                const t_values_row = T_grid_parametric_plot4_js[i]; // T values for this specific Phi
                const h_values_row = Z_surface_plot4_js[i];       // H_total values for this specific Phi

                // Calculate A_total for each T in this row, using the current p_A
                const a_values_row = calculate_A_total_js(t_values_row, D_fixed_js, p_A);

                const row_ratio_A_H = [];
                for (let j = 0; j < t_values_row.length; j++) {{ // Iterates over T dimension
                    const ratio_val = calculate_ratio_A_H_js(a_values_row[j], h_values_row[j], ratio_cap_js);
                    row_ratio_A_H.push(ratio_val);
                }}
                new_Y_surface_plot4_ratio.push(row_ratio_A_H);
            }}
            // Update the Y data of the plot
            Plotly.react(plot4Div, [{{ ...trace_surface_plot4, y: new_Y_surface_plot4_ratio }}], layout_plot4);
        }}
        p_A_slider_plot4.addEventListener('input', updatePlot4);

        // --- Evolutionary Paths Visualization ---

        // Convert Python arrays to JavaScript for visualization
        const time_values_js = {json.dumps(time_values.tolist())};
        const evolutionary_paths_js = [
            // Scenario 1: Accelerating trust with static review capability
            {
    json.dumps(
        {
            "name": evolutionary_paths[0]["name"],
            "time": evolutionary_paths[0]["time"].tolist(),
            "trust": evolutionary_paths[0]["trust"].tolist(),
            "phi": evolutionary_paths[0]["phi"].tolist(),
            "p_a": evolutionary_paths[0]["p_a"].tolist(),
            "human_workforce": evolutionary_paths[0]["human_workforce"].tolist(),
            "ai_agents": evolutionary_paths[0]["ai_agents"].tolist(),
            "ai_human_ratio": evolutionary_paths[0]["ai_human_ratio"].tolist(),
        }
    )
},

            // Scenario 2: Accelerating trust with decreasing review capability
            {
    json.dumps(
        {
            "name": evolutionary_paths[1]["name"],
            "time": evolutionary_paths[1]["time"].tolist(),
            "trust": evolutionary_paths[1]["trust"].tolist(),
            "phi": evolutionary_paths[1]["phi"].tolist(),
            "p_a": evolutionary_paths[1]["p_a"].tolist(),
            "human_workforce": evolutionary_paths[1]["human_workforce"].tolist(),
            "ai_agents": evolutionary_paths[1]["ai_agents"].tolist(),
            "ai_human_ratio": evolutionary_paths[1]["ai_human_ratio"].tolist(),
        }
    )
},

            // Scenario 3: Oscillating trust with static review capability
            {
    json.dumps(
        {
            "name": evolutionary_paths[2]["name"],
            "time": evolutionary_paths[2]["time"].tolist(),
            "trust": evolutionary_paths[2]["trust"].tolist(),
            "phi": evolutionary_paths[2]["phi"].tolist(),
            "p_a": evolutionary_paths[2]["p_a"].tolist(),
            "human_workforce": evolutionary_paths[2]["human_workforce"].tolist(),
            "ai_agents": evolutionary_paths[2]["ai_agents"].tolist(),
            "ai_human_ratio": evolutionary_paths[2]["ai_human_ratio"].tolist(),
        }
    )
},

            // Scenario 4: Linear trust with decreasing productivity
            {
    json.dumps(
        {
            "name": evolutionary_paths[3]["name"],
            "time": evolutionary_paths[3]["time"].tolist(),
            "trust": evolutionary_paths[3]["trust"].tolist(),
            "phi": evolutionary_paths[3]["phi"].tolist(),
            "p_a": evolutionary_paths[3]["p_a"].tolist(),
            "human_workforce": evolutionary_paths[3]["human_workforce"].tolist(),
            "ai_agents": evolutionary_paths[3]["ai_agents"].tolist(),
            "ai_human_ratio": evolutionary_paths[3]["ai_human_ratio"].tolist(),
        }
    )
},

            // Scenario 5: Trust with incidents
            {
    json.dumps(
        {
            "name": evolutionary_paths[4]["name"],
            "time": evolutionary_paths[4]["time"].tolist(),
            "trust": evolutionary_paths[4]["trust"].tolist(),
            "phi": evolutionary_paths[4]["phi"].tolist(),
            "p_a": evolutionary_paths[4]["p_a"].tolist(),
            "human_workforce": evolutionary_paths[4]["human_workforce"].tolist(),
            "ai_agents": evolutionary_paths[4]["ai_agents"].tolist(),
            "ai_human_ratio": evolutionary_paths[4]["ai_human_ratio"].tolist(),
        }
    )
}
        ];

        // Functions to create plots for each evolutionary path
        function createParametersPlot(path, divId) {{
            const plotDiv = document.getElementById(divId);
            const traces = [
                {{
                    x: path.time,
                    y: path.trust,
                    name: 'Trust (T)',
                    line: {{ color: '#3498db', width: 2 }}
                }},
                {{
                    x: path.time,
                    y: path.phi,
                    name: 'Review Power (Φ)',
                    line: {{ color: '#2ecc71', width: 2 }}
                }},
                {{
                    x: path.time,
                    y: path.p_a,
                    name: 'AI Productivity (p<sub>A</sub>)',
                    line: {{ color: '#9b59b6', width: 2 }}
                }}
            ];

            const layout = {{
                title: {{ text: 'Parameter Evolution', font: {{ size: 14 }} }},
                xaxis: {{ title: 'Time', titlefont: {{ size: 12 }} }},
                yaxis: {{ title: 'Parameter Value', titlefont: {{ size: 12 }} }},
                legend: {{ font: {{ size: 10 }}, orientation: 'h', y: -0.2 }},
                margin: {{ t: 30, b: 50, l: 50, r: 20 }},
                hovermode: 'closest'
            }};

            Plotly.newPlot(plotDiv, traces, layout);
        }}

        function createWorkforcePlot(path, divId) {{
            const plotDiv = document.getElementById(divId);
            const traces = [
                {{
                    x: path.time,
                    y: path.human_workforce,
                    name: 'Human Workforce',
                    line: {{ color: '#2C3E50', width: 2.5 }}
                }},
                {{
                    x: path.time,
                    y: path.ai_agents,
                    name: 'AI Agents',
                    line: {{ color: '#FDE725', width: 2.5 }}
                }}
            ];

            const layout = {{
                title: {{ text: 'Workforce Evolution', font: {{ size: 14 }} }},
                xaxis: {{ title: 'Time', titlefont: {{ size: 12 }} }},
                yaxis: {{ title: 'Number of Workers / Agents', titlefont: {{ size: 12 }} }},
                legend: {{ font: {{ size: 10 }}, orientation: 'h', y: -0.2 }},
                margin: {{ t: 30, b: 50, l: 50, r: 20 }},
                hovermode: 'closest'
            }};

            Plotly.newPlot(plotDiv, traces, layout);
        }}

        function create3DPathPlot(path, divId) {{
            const plotDiv = document.getElementById(divId);

            // Determine which parameter is changing (non-static)
            // We'll inspect variance to determine which parameter to use as y-axis
            const trustVar = Math.variance(path.trust);
            const phiVar = Math.variance(path.phi);
            const p_aVar = Math.variance(path.p_a);

            // Define a threshold for "static" parameters (very small variance)
            const varianceThreshold = 0.0001;

            // Select primary evolving parameter
            let primaryParam = {{name: 'trust', values: path.trust, title: 'Trust (T)'}};

            // Find the most variable parameter (after time)
            if (phiVar > trustVar && phiVar > p_aVar && phiVar > varianceThreshold) {{
                primaryParam = {{name: 'phi', values: path.phi, title: 'Review Power (Φ)'}};
            }} else if (p_aVar > trustVar && p_aVar > phiVar && p_aVar > varianceThreshold) {{
                primaryParam = {{name: 'p_a', values: path.p_a, title: 'AI Productivity (p<sub>A</sub>)'}};
            }}

            // Create the 3D path trace
            const trace = {{
                type: 'scatter3d',
                x: path.time,  // Time on X-axis
                y: primaryParam.values,  // Most variable parameter on Y-axis
                z: path.human_workforce,  // Human workforce on Z-axis
                mode: 'lines+markers',
                line: {{
                    color: path.ai_human_ratio, // Color by AI/Human ratio
                    width: 5,
                    colorscale: 'Viridis',
                    cmin: 0,
                    cmax: 5
                }},
                marker: {{
                    size: 3,
                    color: path.ai_human_ratio,
                    colorscale: 'Viridis',
                    cmin: 0,
                    cmax: 5
                }},
                hoverinfo: 'text',
                hovertext: path.time.map((t, i) =>
                    `Time: ${{t.toFixed(2)}}<br>` +
                    `${{primaryParam.title}}: ${{primaryParam.values[i].toFixed(2)}}<br>` +
                    `Trust: ${{path.trust[i].toFixed(2)}}<br>` +
                    `Review Power: ${{path.phi[i].toFixed(1)}}<br>` +
                    `AI Productivity: ${{path.p_a[i].toFixed(1)}}<br>` +
                    `Humans: ${{path.human_workforce[i].toFixed(0)}}<br>` +
                    `AI/Human Ratio: ${{path.ai_human_ratio[i].toFixed(2)}}`
                )
            }};

            // For surface, we'll create a mesh that shows how humans change with time and the primary parameter
            const timeSamples = 10;
            const paramSamples = 10;

            // Create a grid of time vs primary parameter
            const timeValues = Array.from({{length: timeSamples}}, (_, i) => i / (timeSamples - 1));

            // Set param range based on primary parameter
            let paramMin, paramMax;
            if (primaryParam.name === 'trust') {{
                paramMin = {T_min_model};
                paramMax = {T_max_model};
            }} else if (primaryParam.name === 'phi') {{
                paramMin = {Phi_min};
                paramMax = {Phi_max};
            }} else {{ // p_a
                paramMin = {p_A_min};
                paramMax = {p_A_max};
            }}

            const paramValues = Array.from({{length: paramSamples}}, (_, i) =>
                paramMin + (paramMax - paramMin) * i / (paramSamples - 1));

            // Create the surface
            const xSurface = []; // Time values
            const ySurface = []; // Parameter values
            const zSurface = []; // Human workforce
            const colorSurface = []; // AI/Human ratio

            // For each time-parameter combination, calculate the corresponding human workforce
            for (let i = 0; i < timeValues.length; i++) {{
                for (let j = 0; j < paramValues.length; j++) {{
                    // Interpolate values at this time for parameters that aren't the primary
                    const timeVal = timeValues[i];
                    const paramVal = paramValues[j];

                    // Interpolate other parameters based on time
                    let trustVal, phiVal, p_aVal;

                    // Find the parameters for this time value
                    if (primaryParam.name === 'trust') {{
                        trustVal = paramVal;
                        phiVal = interpolateAtTime(timeVal, path.time, path.phi);
                        p_aVal = interpolateAtTime(timeVal, path.time, path.p_a);
                    }} else if (primaryParam.name === 'phi') {{
                        trustVal = interpolateAtTime(timeVal, path.time, path.trust);
                        phiVal = paramVal;
                        p_aVal = interpolateAtTime(timeVal, path.time, path.p_a);
                    }} else {{ // p_a
                        trustVal = interpolateAtTime(timeVal, path.time, path.trust);
                        phiVal = interpolateAtTime(timeVal, path.time, path.phi);
                        p_aVal = paramVal;
                    }}

                    // Calculate workforce values
                    const humanVal = calculate_H_total_js(trustVal, D_fixed_js, phiVal);
                    const aiVal = calculate_A_total_js(trustVal, D_fixed_js, p_aVal);
                    const ratioVal = calculate_ratio_A_H_js(aiVal, humanVal, ratio_cap_js);

                    xSurface.push(timeVal);
                    ySurface.push(paramVal);
                    zSurface.push(humanVal);
                    colorSurface.push(ratioVal);
                }}
            }}

            // Create a simplified surface using delaunay triangulation
            const surfaceTrace = {{
                type: 'mesh3d',
                x: xSurface,
                y: ySurface,
                z: zSurface,
                opacity: 0.3,
                colorscale: 'Viridis',
                intensity: colorSurface,
                cmin: 0,
                cmax: 5,
                showscale: false
            }};

            const layout = {{
                title: {{ text: 'Path Through Parameter Space', font: {{ size: 14 }} }},
                scene: {{
                    xaxis: {{ title: 'Time', range: [0, 1] }},
                    yaxis: {{ title: primaryParam.title, range: [paramMin, paramMax] }},
                    zaxis: {{ title: 'Human Workforce', autorange: true }},
                    aspectratio: {{ x: 1, y: 1, z: 1 }}
                }},
                margin: {{ t: 30, b: 0, l: 0, r: 0 }}
            }};

            Plotly.newPlot(plotDiv, [surfaceTrace, trace], layout);
        }}

        // Helper function for interpolation
        function interpolateAtTime(time, times, values) {{
            // Find the index where our target time would be inserted
            let i = 0;
            while (i < times.length && times[i] < time) {{
                i++;
            }}

            // If time is before first sample or after last sample, use the endpoint
            if (i === 0) return values[0];
            if (i === times.length) return values[times.length - 1];

            // Linear interpolation between adjacent points
            const t0 = times[i-1];
            const t1 = times[i];
            const v0 = values[i-1];
            const v1 = values[i];

            // Calculate interpolation weight (0 to 1)
            const weight = (time - t0) / (t1 - t0);

            // Perform interpolation
            return v0 + weight * (v1 - v0);
        }}

        // Add variance calculation helper
        Math.variance = function(arr) {{
            if (arr.length === 0) return 0;
            const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
            return arr.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / arr.length;
        }};

        // Initialize plots for each scenario
        document.addEventListener('DOMContentLoaded', function() {{
            // Scenario 1: Accelerating trust with static review capability
            createParametersPlot(evolutionary_paths_js[0], 'path1_params_div');
            createWorkforcePlot(evolutionary_paths_js[0], 'path1_workforce_div');
            create3DPathPlot(evolutionary_paths_js[0], 'path1_3d_div');

            // Scenario 2: Accelerating trust with decreasing review capability
            createParametersPlot(evolutionary_paths_js[1], 'path2_params_div');
            createWorkforcePlot(evolutionary_paths_js[1], 'path2_workforce_div');
            create3DPathPlot(evolutionary_paths_js[1], 'path2_3d_div');

            // Scenario 3: Oscillating trust with static review capability
            createParametersPlot(evolutionary_paths_js[2], 'path3_params_div');
            createWorkforcePlot(evolutionary_paths_js[2], 'path3_workforce_div');
            create3DPathPlot(evolutionary_paths_js[2], 'path3_3d_div');

            // Scenario 4: Linear trust with decreasing productivity
            createParametersPlot(evolutionary_paths_js[3], 'path4_params_div');
            createWorkforcePlot(evolutionary_paths_js[3], 'path4_workforce_div');
            create3DPathPlot(evolutionary_paths_js[3], 'path4_3d_div');

            // Scenario 5: Trust with incidents
            createParametersPlot(evolutionary_paths_js[4], 'path5_params_div');
            createWorkforcePlot(evolutionary_paths_js[4], 'path5_workforce_div');
            create3DPathPlot(evolutionary_paths_js[4], 'path5_3d_div');
        }});
"""


# Save the HTML and JavaScript content to separate files
def main():
    if not os.path.exists(settings.output_path):
        os.makedirs(settings.output_path, exist_ok=True)

    # Generate HTML file
    html_output_path = f"{settings.output_path}/llms_and_humans.html"
    with open(html_output_path, "w", encoding="utf-8") as f:
        f.write(html_content_combined)

    # Generate JavaScript file
    js_output_path = f"{settings.output_path}/llms_and_humans.js"
    with open(js_output_path, "w", encoding="utf-8") as f:
        f.write(javascript_content)

    final_output_message = f"The interactive HTML page has been generated and saved as '{html_output_path}'. The JavaScript has been separated into '{js_output_path}'. Both files are needed - upload the JS file to your public hosting (e.g., Google Cloud) and update the script src path in the HTML before using in Webflow."
    print(final_output_message)


if __name__ == "__main__":
    main()
