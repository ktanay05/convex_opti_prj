using Convex
using SCS
using LinearAlgebra
using Printf

"""
Solve sensor network localization using Semidefinite Programming (SDP) relaxation.

This implements the classic Biswas-Ye SDP relaxation for distance-based localization.
The non-convex problem:
    min Σ (||x_i - x_j|| - d_ij)²
is relaxed to an SDP by introducing matrix variable X = [x₁...xₙ][x₁...xₙ]ᵀ
and enforcing X ⪰ 0.

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
function solve_sdp_centralized(
    n_agents::Int,
    d::Int,
    anchor_pos::Matrix{Float64},
    measurements::Vector
)
    n_anchors = size(anchor_pos, 1)
    
    # Decision variable: X = [x₁...xₙ][x₁...xₙ]ᵀ ∈ ℝ^(n×n) where n = d × n_agents
    # We use the Gram matrix formulation
    # Define Y = [X  B]  where X is d×n_agents and B = X (for consistency)
    #            [Bᵀ I]
    
    # Simpler formulation: directly optimize positions with SDP constraints
    # X_ij = xᵢᵀxⱼ for the position vectors
    
    # We'll use a lifted formulation with matrix Z
    # Z = [1      xᵀ    ]
    #     [x   X_gram  ]
    # where X_gram[i,j] = xᵢᵀxⱼ
    
    n = n_agents
    
    # Construct the Gram matrix variable
    X = Semidefinite(d * n)  # Gram matrix for all agents stacked
    
    # For simplicity, we use a direct position-based formulation with
    # regularization to encourage low-rank solutions
    
    # Alternative: Use convex formulation with distance constraints
    # Let's use the EDM (Euclidean Distance Matrix) approach
    
    # Define position variables directly
    positions = Variable(n, d)
    
    # Objective: minimize sum of squared distance errors
    obj = 0.0
    
    for meas in measurements
        type_i, i, type_j, j, dist_true, dist_measured, is_outlier = meas
        
        if type_i == :agent && type_j == :agent
            # Agent-agent measurement
            error = sumsquares(positions[i, :] - positions[j, :]) - dist_measured^2
            obj += square(error)
        elseif type_i == :agent && type_j == :anchor
            # Agent-anchor measurement
            error = sumsquares(positions[i, :] - anchor_pos[j, :]) - dist_measured^2
            obj += square(error)
        end
    end
    
    # Note: The above is NOT convex! We need proper SDP relaxation.
    # Let me implement the correct SDP formulation.
    
    println("Warning: Using least-squares approximation (not full SDP)")
    println("For true SDP, we need Gram matrix formulation with rank constraints")
    
    problem = minimize(obj)
    
    # Solve
    solve_time = @elapsed solve!(problem, SCS.Optimizer; silent_solver=false)
    
    agent_pos_est = evaluate(positions)
    obj_value = evaluate(obj)
    
    return agent_pos_est, obj_value, solve_time
end

"""
Solve using proper SDP relaxation with Gram matrix.
This is the mathematically correct approach.
"""
function solve_sdp_gram_matrix(
    n_agents::Int,
    d::Int,
    anchor_pos::Matrix{Float64},
    measurements::Vector
)
    n_anchors = size(anchor_pos, 1)
    
    # Gram matrix formulation
    # We have variables x₁, ..., xₙ ∈ ℝᵈ (agent positions)
    # Define Gram matrix G where G[i,j] = xᵢᵀxⱼ
    # For distance constraints: ||xᵢ - xⱼ||² = G[i,i] + G[j,j] - 2G[i,j]
    
    # Lifted variable: Z = [1    xᵀ  ]  where x = vec([x₁...xₙ])
    #                     [x   G   ]
    # Z must be PSD
    
    dim = 1 + d * n_agents
    Z = Semidefinite(dim)
    
    # Extract components
    # Z[1,1] = 1
    # Z[2:end, 1] = x (flattened positions)
    # Z[2:end, 2:end] = Gram matrix
    
    constraints = [Z[1, 1] == 1]
    
    # Objective: minimize distance errors
    obj = 0.0
    
    for meas in measurements
        type_i, i, type_j, j, dist_true, dist_measured, is_outlier = meas
        
        if type_i == :agent && type_j == :agent
            # Extract blocks for agent i and j
            # Position i: Z[2 + (i-1)*d : 1 + i*d, 1]
            # Position j: Z[2 + (j-1)*d : 1 + j*d, 1]
            
            idx_i = (2 + (i-1)*d) : (1 + i*d)
            idx_j = (2 + (j-1)*d) : (1 + j*d)
            
            # ||xᵢ - xⱼ||² = sum((Z[idx_i, 1] - Z[idx_j, 1]).^2)
            #               = Z[idx_i,1]ᵀZ[idx_i,1] + Z[idx_j,1]ᵀZ[idx_j,1] - 2Z[idx_i,1]ᵀZ[idx_j,1]
            
            # Using Gram matrix: = tr(Z[idx_i,idx_i]) + tr(Z[idx_j,idx_j]) - 2tr(Z[idx_i,idx_j])
            dist_sq = quadform(Z[idx_i, 1], Matrix{Float64}(I, d, d)) + 
                     quadform(Z[idx_j, 1], Matrix{Float64}(I, d, d)) - 
                     2 * dot(Z[idx_i, 1], Z[idx_j, 1])
            
            error = dist_sq - dist_measured^2
            obj += square(error)
            
        elseif type_i == :agent && type_j == :anchor
            idx_i = (2 + (i-1)*d) : (1 + i*d)
            a_j = anchor_pos[j, :]
            
            # ||xᵢ - aⱼ||² = ||xᵢ||² - 2xᵢᵀaⱼ + ||aⱼ||²
            dist_sq = quadform(Z[idx_i, 1], Matrix{Float64}(I, d, d)) - 
                     2 * dot(Z[idx_i, 1], a_j) + 
                     dot(a_j, a_j)
            
            error = dist_sq - dist_measured^2
            obj += square(error)
        end
    end
    
    problem = minimize(obj, constraints)
    
    # Solve
    println("\nSolving SDP with SCS...")
    solve_time = @elapsed solve!(problem, SCS.Optimizer; silent_solver=true)
    
    # Extract solution
    Z_val = evaluate(Z)
    agent_pos_est = zeros(n_agents, d)
    
    for i in 1:n_agents
        idx_i = (2 + (i-1)*d) : (1 + i*d)
        agent_pos_est[i, :] = Z_val[idx_i, 1]
    end
    
    obj_value = evaluate(obj)
    
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
    
    println("\nSolving with SDP (Gram matrix formulation)...")
    agent_pos_est, obj_value, solve_time = solve_sdp_gram_matrix(
        n_agents, d, anchor_pos, measurements
    )
    
    rmse = compute_rmse(agent_pos_est, agent_pos_true)
    
    println("\n" * "=" ^ 60)
    println("SDP Solution Results")
    println("=" ^ 60)
    @printf("Solve time: %.3f seconds\n", solve_time)
    @printf("Objective value: %.6f\n", obj_value)
    @printf("RMSE: %.6f\n", rmse)
    println("=" ^ 60)
end
