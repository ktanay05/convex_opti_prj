using LinearAlgebra
using Random
using Statistics

include("agent_dynamics.jl")
include("ekf_localization.jl")
include("problem_data.jl")

"""
Reinforcement Learning Environment for Multi-Agent Localization Control.

Agents learn optimal turn rates to minimize localization error while
maintaining communication connectivity.
"""

mutable struct LocControlEnv
    # Environment state
    agents::Vector{AgentState}
    true_agents::Vector{AgentState}
    anchors::Matrix{Float64}
    
    # Simulation parameters
    dt::Float64
    rc::Float64
    σ_anchor::Float64
    σ_range::Float64
    σ_process::Float64
    
    # Episode tracking
    step_count::Int
    max_steps::Int
    
    # Action limits
    ω_max::Float64
    
    # Initial positions (for reset)
    initial_positions::Vector{Vector{Float64}}
end

"""
Create a new RL environment.
"""
function LocControlEnv(;
    n_agents::Int=5,
    n_anchors::Int=5*23,
    dt::Float64=0.1,
    rc::Float64=5.0,
    ω_max::Float64=0.5,
    max_steps::Int=100,
    seed::Int=42
)
    Random.seed!(seed)
    
    # Generate anchors and initial positions
    _, anchor_pos, _ = generate_network_data(
        n_agents=n_agents, n_anchors=n_anchors, d=2, seed=seed
    )
    
    # Random initial positions
    initial_positions = [10.0 * (rand(2) .- 0.5) for _ in 1:n_agents]
    
    # Create agents
    agents = [create_agent(i, initial_positions[i], 2π*rand(), 2.0) 
              for i in 1:n_agents]
    true_agents = deepcopy(agents)
    
    return LocControlEnv(
        agents, true_agents, anchor_pos,
        dt, rc, 0.05, 0.1, 0.1,
        0, max_steps, ω_max, initial_positions
    )
end

"""
Get observation for a single agent.
"""
function get_observation(env::LocControlEnv, agent_idx::Int)
    agent = env.agents[agent_idx]
    true_agent = env.true_agents[agent_idx]
    
    # Agent's own EKF estimate
    px_est, py_est = agent.position
    heading_est = agent.heading
    pos_uncertainty = sqrt(tr(agent.P[1:2, 1:2]))
    
    # Nearest anchor
    dists_to_anchors = [norm(agent.position - env.anchors[i, :]) 
                        for i in 1:size(env.anchors, 1)]
    nearest_anchor_idx = argmin(dists_to_anchors)
    anchor_vec = env.anchors[nearest_anchor_idx, :] - agent.position
    anchor_dist = norm(anchor_vec)
    
    # Neighbor information
    neighbors = find_neighbors(agent_idx, env.true_agents, env.rc)
    n_neighbors = length(neighbors)
    
    if n_neighbors > 0
        neighbor_vecs = [env.true_agents[j].position - agent.position 
                         for j in neighbors]
        avg_neighbor_vec = mean(neighbor_vecs)
    else
        avg_neighbor_vec = [0.0, 0.0]
    end
    
    # Construct observation vector
    obs = [
        px_est, py_est,
        heading_est,
        pos_uncertainty,
        anchor_vec[1], anchor_vec[2],
        anchor_dist,
        Float64(n_neighbors),
        avg_neighbor_vec[1], avg_neighbor_vec[2],
        agent.speed,
        agent.turn_rate
    ]
    
    return obs
end

"""
Get observations for all agents.
"""
function get_observations(env::LocControlEnv)
    n_agents = length(env.agents)
    return [get_observation(env, i) for i in 1:n_agents]
end

"""
Compute reward for a single agent.
"""
function compute_reward(env::LocControlEnv, agent_idx::Int, action::Float64)
    agent = env.agents[agent_idx]
    true_agent = env.true_agents[agent_idx]
    
    # Position error (localization quality)
    pos_error = norm(agent.position - true_agent.position)
    localization_reward = -pos_error^2
    
    # Connectivity (number of neighbors)
    neighbors = find_neighbors(agent_idx, env.true_agents, env.rc)
    connectivity_reward = Float64(length(neighbors))
    
    # Smoothness penalty (discourage jerky control)
    prev_turn_rate = agent.turn_rate
    smoothness_penalty = -(action - prev_turn_rate)^2
    
    # Combined reward
    w1, w2, w3 = 10.0, 1.0, 0.1
    reward = w1 * localization_reward + w2 * connectivity_reward + w3 * smoothness_penalty
    
    return reward
end

"""
Reset environment to initial state.
"""
function reset!(env::LocControlEnv)
    n_agents = length(env.agents)
    
    # Reset agents to initial positions with random headings
    for i in 1:n_agents
        heading = 2π * rand()
        env.agents[i] = create_agent(i, env.initial_positions[i], heading, 2.0)
        env.true_agents[i] = deepcopy(env.agents[i])
    end
    
    env.step_count = 0
    
    return get_observations(env)
end

"""
Execute one environment step with actions from all agents.
"""
function step!(env::LocControlEnv, actions::Vector{Float64})
    n_agents = length(env.agents)
    
    # Clamp actions to valid range
    actions = clamp.(actions, -env.ω_max, env.ω_max)
    
    # Apply actions to true agents (set turn rates)
    for i in 1:n_agents
        env.true_agents[i].turn_rate = actions[i]
        env.true_agents[i].x[5] = actions[i]
    end
    
    # Propagate true states
    for i in 1:n_agents
        env.true_agents[i].x = propagate_state_nonlinear(env.true_agents[i].x, env.dt)
        env.true_agents[i].position = env.true_agents[i].x[1:2]
        env.true_agents[i].heading = env.true_agents[i].x[3]
    end
    
    # Update EKF estimates
    for i in 1:n_agents
        ekf_step!(env.agents[i], env.dt, env.anchors, env.true_agents,
                 rc=env.rc, σ_anchor=env.σ_anchor, σ_range=env.σ_range,
                 σ_process=env.σ_process, enable_fusion=true)
    end
    
    # Compute rewards
    rewards = [compute_reward(env, i, actions[i]) for i in 1:n_agents]
    
    # Get new observations
    observations = get_observations(env)
    
    # Check if episode is done
    env.step_count += 1
    done = env.step_count >= env.max_steps
    dones = fill(done, n_agents)
    
    # Info dictionary
    info = Dict(
        "position_errors" => [norm(env.agents[i].position - env.true_agents[i].position) 
                              for i in 1:n_agents],
        "n_neighbors" => [length(find_neighbors(i, env.true_agents, env.rc)) 
                          for i in 1:n_agents]
    )
    
    return observations, rewards, dones, info
end

"""
Get state and action dimensions.
"""
function get_dims(env::LocControlEnv)
    obs = get_observation(env, 1)
    state_dim = length(obs)
    action_dim = 1  # Single continuous action (turn rate)
    return state_dim, action_dim
end

# Test environment
if abspath(PROGRAM_FILE) == @__FILE__
    println("\n=== RL Environment Test ===\n")
    
    env = LocControlEnv(n_agents=3, max_steps=10)
    state_dim, action_dim = get_dims(env)
    
    println("State dimension: $state_dim")
    println("Action dimension: $action_dim")
    println("Number of agents: $(length(env.agents))")
    
    # Test reset
    obs = reset!(env)
    println("\nInitial observations (agent 1): $(obs[1])")
    
    # Test step
    actions = [0.1, -0.1, 0.0]  # Random actions
    obs, rewards, dones, info = step!(env, actions)
    
    println("\nAfter step:")
    println("  Rewards: $rewards")
    println("  Position errors: $(info["position_errors"])")
    println("  Neighbors: $(info["n_neighbors"])")
    println("  Done: $(dones[1])")
end
