using LinearAlgebra
using Statistics

include("agent_dynamics.jl")

"""
Extended Kalman Filter for agent localization.

Estimates full 9D state: [px, py, vx, vy, ax, ay, θ, ω, 0]
"""

"""
EKF Prediction step for fixed-wing aircraft.

Propagates state and covariance forward using nonlinear Dubins model.
"""
function ekf_predict!(agent::AgentState, dt::Float64, σ_process::Float64=0.1)
    # Nonlinear state propagation
    agent.x = propagate_state_nonlinear(agent.x, dt)
    
    # Linearized state transition for covariance propagation
    F = get_state_transition_matrix(dt, agent.heading, agent.speed)
    Q = get_process_noise(dt, σ_process)
    
    # Predict covariance
    agent.P = F * agent.P * F' + Q
    
    # Update derived quantities
    agent.position = agent.x[1:2]
    agent.heading = agent.x[3]
    agent.speed = agent.x[4]
    agent.turn_rate = agent.x[5]
    
    return agent
end

"""
Range measurement model: z = ||p_agent - p_target|| + noise

Returns:
    - h: predicted measurement
    - H: measurement Jacobian (1 x 5)
"""
function range_measurement_model(agent_pos::Vector{Float64}, 
                                 target_pos::Vector{Float64})
    diff = agent_pos - target_pos
    range = norm(diff)
    
    # Predicted measurement
    h = range
    
    # Jacobian: ∂h/∂x for 5D state [px, py, θ, v, ω]
    # Only position affects range measurement
    H = zeros(1, 5)
    if range > 1e-6
        H[1, 1] = diff[1] / range  # ∂h/∂px
        H[1, 2] = diff[2] / range  # ∂h/∂py
        # H[1, 3:5] = 0 (heading, speed, turn rate don't affect range)
    end
    
    return h, H
end

"""
EKF Update step with range measurement.

Updates agent state based on range measurement to a target (anchor or agent).
"""
function ekf_update_range!(agent::AgentState,
                          z_measured::Float64,
                          target_pos::Vector{Float64},
                          σ_measurement::Float64=0.1)
    
    # Measurement model
    h, H = range_measurement_model(agent.position, target_pos)
    
    # Innovation
    y = z_measured - h
    
    # Innovation covariance (convert scalar R to matrix for addition)
    R = σ_measurement^2
    S = H * agent.P * H' .+ R  # Use broadcasting to add scalar to matrix
    
    # Kalman gain (S is 1x1 matrix, convert to scalar)
    S_scalar = S[1, 1]
    K = (agent.P * H') / S_scalar  # K is 5x1 column vector
    
    # Update state (K is 5x1, y is scalar, result is 5x1)
    agent.x = agent.x + vec(K) * y
    
    # Update covariance (Joseph form for numerical stability)
    I_mat = Matrix{Float64}(I, 5, 5)
    I_KH = I_mat - K * H
    # R is scalar, need to convert for Joseph form
    agent.P = I_KH * agent.P * I_KH' + (K * K') * R
    
    # Update derived quantities
    agent.position = agent.x[1:2]
    agent.heading = agent.x[3]
    agent.speed = agent.x[4]
    agent.turn_rate = agent.x[5]
    
    return agent
end

"""
Find agents within communication range rc.
"""
function find_neighbors(agent_id::Int, 
                       agents::Vector{AgentState},
                       rc::Float64=5.0)
    neighbors = Int[]
    agent_pos = agents[agent_id].position
    
    for other in agents
        if other.id != agent_id
            dist = norm(agent_pos - other.position)
            if dist <= rc
                push!(neighbors, other.id)
            end
        end
    end
    
    return neighbors
end

"""
Covariance Intersection for fusing estimates from multiple agents.

Conservative fusion that handles unknown correlations.
For fixed-wing agents, only fuse position information.
"""
function covariance_intersection!(agent::AgentState,
                                 neighbor_x::Vector{Float64},
                                 neighbor_P::Matrix{Float64},
                                 ω::Float64=0.5)
    
    # Extract position components only for fusion
    # (Conservative: only fuse position, keep own heading/speed/turn_rate)
    
    # Add small regularization for numerical stability
    reg = 1e-6 * Matrix{Float64}(I, 2, 2)
    P_inv = inv(agent.P[1:2, 1:2] + reg)
    P_neighbor_inv = inv(neighbor_P[1:2, 1:2] + reg)
    
    # Fused covariance
    P_fused_inv = ω * P_inv + (1 - ω) * P_neighbor_inv
    P_fused = inv(P_fused_inv)
    
    # Fused mean
    x_fused = P_fused * (ω * P_inv * agent.position + 
                        (1 - ω) * P_neighbor_inv * neighbor_x[1:2])
    
    # Update agent position
    agent.x[1:2] = x_fused
    agent.position = x_fused
    agent.P[1:2, 1:2] = P_fused
    
    return agent
end

"""
Single EKF time step with multiple measurements.

Performs:
    1. Prediction
    2. Update with anchor measurements
    3. Update with neighbor range measurements
    4. Optional: Information fusion with neighbors
"""
function ekf_step!(agent::AgentState,
                  dt::Float64,
                  anchors::Matrix{Float64},
                  agents::Vector{AgentState};
                  rc::Float64=5.0,
                  σ_anchor::Float64=0.05,
                  σ_range::Float64=0.1,
                  σ_process::Float64=0.1,
                  enable_fusion::Bool=true)
    
    # 1. Prediction
    ekf_predict!(agent, dt, σ_process)
    
    # 2. Update with anchor measurements
    n_anchors = size(anchors, 1)
    for i in 1:n_anchors
        # Simulate range measurement
        true_range = norm(agent.position - anchors[i, :])
        z_measured = true_range + σ_anchor * randn()
        
        # EKF update
        ekf_update_range!(agent, z_measured, anchors[i, :], σ_anchor)
    end
    
    # 3. Find neighbors and update with their measurements
    neighbors = find_neighbors(agent.id, agents, rc)
    
    for neighbor_id in neighbors
        neighbor = agents[neighbor_id]
        
        # Range measurement to neighbor
        true_range = norm(agent.position - neighbor.position)
        z_measured = true_range + σ_range * randn()
        
        # EKF update
        ekf_update_range!(agent, z_measured, neighbor.position, σ_range)
    end
    
    # 4. Information fusion (optional)
    if enable_fusion && !isempty(neighbors)
        # Fuse with average neighbor estimate (simplified)
        for neighbor_id in neighbors
            neighbor = agents[neighbor_id]
            covariance_intersection!(agent, neighbor.x, neighbor.P, 0.7)
        end
    end
    
    return agent
end

# Example usage
if abspath(PROGRAM_FILE) == @__FILE__
    println("\n=== EKF Localization Test ===\n")
    
    # Create fixed-wing agents
    agents = [
        create_agent(1, [0.0, 0.0], 0.0, 2.0),
        create_agent(2, [3.0, 2.0], π/4, 2.0)
    ]
    
    # Anchors (static)
    anchors = [
        -5.0 5.0;
        5.0 5.0;
        5.0 -5.0;
        -5.0 -5.0
    ]
    
    println("Initial states:")
    for agent in agents
        println("  Agent $(agent.id): pos=$(agent.position), heading=$(agent.heading)")
    end
    
    # Run EKF for 10 steps
    dt = 0.1
    for step in 1:10
        for agent in agents
            ekf_step!(agent, dt, anchors, agents, rc=5.0)
        end
    end
    
    println("\nAfter 10 EKF steps (1 second):")
    for agent in agents
        println("  Agent $(agent.id): pos=$(agent.position), cov_trace=$(tr(agent.P[1:2,1:2]))")
    end
end
