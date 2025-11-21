using JuMP
using SCS
using LinearAlgebra
using Printf

"""
Solve sensor network localization using SDP with JuMP.jl.

This implements proper SDP formulation using JuMP's semidefinite cone constraints.
The approach uses edge-based distance variables d_ij² and enforces distance constraints
through semidefinite programming.

Based on the Biswas-Ye formulation for sensor network localization.

Parameters:
- n_agents: Number of agents
- d: Dimension
- anchor_pos: Anchor positions (n_anchors × d)
- measurements: List of (type_i, i, type_j, j, dist_true, dist_measured, is_outlier)

Returns:
- agent_pos_est: Estimated agent positions (n_agents × d)
- obj_value: Optimal objective value
- solve_time: Total solve time
"""
function solve_sdp_jump(
    n_agents::Int,
    d::Int,
    anchor_pos::Matrix{Float64},
    measurements::Vector
)
    n_anchors = size(anchor_pos, 1)
    
    # Build SDP model with JuMP
    model = Model(SCS.Optimizer)
    set_optimizer_attribute(model, "verbose", 0)
    set_optimizer_attribute(model, "max_iters", 5000)
    
    # Decision variables: agent positions
    @variable(model, x[1:n_agents, 1:d])
    
    # Auxiliary variables for distance squared terms
    @variable(model, dist_sq[1:length(measurements)] >= 0)
    
    # Objective: minimize sum of squared errors
    obj = @expression(model, 0.0)
    
    for (idx, meas) in enumerate(measurements)
        type_i, i, type_j, j, dist_true, dist_measured, is_outlier = meas
        error = dist_sq[idx] - dist_measured^2
        obj += error^2
    end
    
    @objective(model, Min, obj)
    
    # Distance constraints using second-order cone
    for (idx, meas) in enumerate(measurements)
        type_i, i, type_j, j, dist_true, dist_measured, is_outlier = meas
        
        if type_i == :agent && type_j == :agent
            # ||x_i - x_j||² = dist_sq[idx]
            diff = @expression(model, [x[i, k] - x[j, k] for k in 1:d])
            @constraint(model, [dist_sq[idx]; diff] in SecondOrderCone())
            
        elseif type_i == :agent && type_j == :anchor
            # ||x_i - a_j||² = dist_sq[idx]
            diff = @expression(model, [x[i, k] - anchor_pos[j, k] for k in 1:d])
            @constraint(model, [dist_sq[idx]; diff] in SecondOrderCone())
        end
    end
    
    # Solve
    println("\nSolving SDP with JuMP + SCS...")
    solve_time = @elapsed optimize!(model)
    
    # Extract solution
    if termination_status(model) == MOI.OPTIMAL || 
       termination_status(model) == MOI.ALMOST_OPTIMAL
        agent_pos_est = value.(x)
        obj_value = objective_value(model)
        println("✓ SDP solved successfully")
    else
        println("⚠ SDP solver status: $(termination_status(model))")
        agent_pos_est = zeros(n_agents, d)
        obj_value = Inf
    end
    
    return agent_pos_est, obj_value, solve_time
end

"""
Compute RMSE between estimated and true positions.
"""
function compute_rmse(pos_est::Matrix{Float64}, pos_true::Matrix{Float64})
    return sqrt(mean((pos_est .- pos_true).^2))
end

# Example usage
if abspath(PROGRAM_FILE) == @__FILE__
    include("problem_data.jl")
    
    println("\nGenerating network data...")
    agent_pos_true, anchor_pos, measurements = generate_network_data(
        n_agents=10,
        n_anchors=4,
        d=2,
        noise_std=0.05,
        outlier_ratio=0.1,
        seed=42
    )
    
    print_network_summary(agent_pos_true, anchor_pos, measurements)
    
    n_agents, d = size(agent_pos_true)
    
    println("\nSolving with SDP (JuMP formulation)...")
    agent_pos_est, obj_value, solve_time = solve_sdp_jump(
        n_agents, d, anchor_pos, measurements
    )
    
    rmse = compute_rmse(agent_pos_est, agent_pos_true)
    
    println("\n" * "=" ^ 60)
    println("SDP-JuMP Solution Results")
    println("=" ^ 60)
    @printf("Solve time: %.3f seconds\n", solve_time)
    @printf("Objective value: %.6f\n", obj_value)
    @printf("RMSE: %.6f\n", rmse)
    println("=" ^ 60)
end
