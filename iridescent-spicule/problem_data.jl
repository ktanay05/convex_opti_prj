using LinearAlgebra
using Random
using Statistics

"""
Generate synthetic sensor network data for localization problem.

Parameters:
- n_agents: Number of agents with unknown positions
- n_anchors: Number of anchors with known positions  
- d: Dimension (2D or 3D)
- noise_std: Standard deviation of Gaussian measurement noise
- outlier_ratio: Fraction of measurements to corrupt as outliers
- outlier_scale: Scale factor for outlier corruption
- seed: Random seed for reproducibility

Returns:
- agent_pos_true: Ground truth agent positions (n_agents × d)
- anchor_pos: Known anchor positions (n_anchors × d)
- measurements: Noisy distance measurements
- edges: List of edges (i, j, true_distance, measured_distance, is_outlier)
"""
function generate_network_data(;
    n_agents::Int = 20,
    n_anchors::Int = 5,
    d::Int = 2,
    noise_std::Float64 = 0.1,
    outlier_ratio::Float64 = 0.2,
    outlier_scale::Float64 = 3.0,
    seed::Int = 42
)
    Random.seed!(seed)
    
    # Generate true agent positions uniformly in [0, 10]^d
    agent_pos_true = 10.0 * rand(n_agents, d)
    
    # Generate anchor positions at boundaries
    anchor_pos = zeros(n_anchors, d)
    for i in 1:n_anchors
        angle = 2π * (i - 1) / n_anchors
        if d == 2
            anchor_pos[i, :] = [5.0 + 6.0 * cos(angle), 5.0 + 6.0 * sin(angle)]
        else  # d == 3
            anchor_pos[i, :] = [5.0 + 6.0 * cos(angle), 5.0 + 6.0 * sin(angle), 5.0]
        end
    end
    
    # Build edges: agent-agent and agent-anchor
    edges = []
    
    # Agent-agent edges (proximity-based, distance < 4.0)
    for i in 1:n_agents
        for j in (i+1):n_agents
            dist_true = norm(agent_pos_true[i, :] - agent_pos_true[j, :])
            if dist_true < 4.0  # Only connect nearby agents
                push!(edges, (:agent, i, :agent, j, dist_true))
            end
        end
    end
    
    # Agent-anchor edges (all agents can sense anchors)
    for i in 1:n_agents
        for j in 1:n_anchors
            dist_true = norm(agent_pos_true[i, :] - anchor_pos[j, :])
            push!(edges, (:agent, i, :anchor, j, dist_true))
        end
    end
    
    # Add measurement noise and outliers
    n_edges = length(edges)
    n_outliers = round(Int, outlier_ratio * n_edges)
    outlier_indices = randperm(n_edges)[1:n_outliers]
    
    measurements = []
    for (idx, edge) in enumerate(edges)
        type_i, i, type_j, j, dist_true = edge
        
        # Add Gaussian noise
        noise = noise_std * randn()
        dist_measured = dist_true + noise
        
        # Corrupt with outlier
        is_outlier = idx in outlier_indices
        if is_outlier
            dist_measured += outlier_scale * abs(randn())  # Large positive corruption
        end
        
        push!(measurements, (type_i, i, type_j, j, dist_true, dist_measured, is_outlier))
    end
    
    return agent_pos_true, anchor_pos, measurements
end

"""
Print summary statistics of the generated network.
"""
function print_network_summary(agent_pos_true, anchor_pos, measurements)
    n_agents, d = size(agent_pos_true)
    n_anchors = size(anchor_pos, 1)
    n_edges = length(measurements)
    n_outliers = count(m -> m[7], measurements)
    
    println("=" ^ 60)
    println("Network Summary")
    println("=" ^ 60)
    println("Dimension: $d")
    println("Agents: $n_agents")
    println("Anchors: $n_anchors")
    println("Measurements: $n_edges")
    println("Outliers: $n_outliers ($(round(100 * n_outliers / n_edges, digits=1))%)")
    println("=" ^ 60)
end

# Example usage
if abspath(PROGRAM_FILE) == @__FILE__
    agent_pos_true, anchor_pos, measurements = generate_network_data(
        n_agents=20,
        n_anchors=5,
        d=2,
        noise_std=0.1,
        outlier_ratio=0.2,
        outlier_scale=3.0,
        seed=42
    )
    
    print_network_summary(agent_pos_true, anchor_pos, measurements)
end
