using LinearAlgebra
using Printf

"""
Distributed ADMM solver for sensor network localization.

This implements consensus-based ADMM where each agent maintains local position
estimate and exchanges information only with neighbors.

Formulation:
- Each agent i maintains local position estimate xᵢ
- Consensus constraints: xᵢ = xⱼ for neighbors (i,j)
- ADMM splits this into local subproblems

ADMM Update:
1. x-update: Each agent minimizes local cost + augmented Lagrangian
2. z-update: Update consensus variables (average of neighbors)
3. u-update: Update dual variables (scaled Lagrangian multipliers)

Parameters:
- n_agents: Number of agents
- d: Dimension
- anchor_pos: Anchor positions
- measurements: Measurement list
- max_iter: Maximum ADMM iterations
- rho: Augmented Lagrangian penalty parameter
- tol: Convergence tolerance

Returns:
- agent_pos_est: Estimated positions
- residuals: Primal and dual residuals per iteration
- solve_time: Total solve time
"""
function solve_admm_distributed(
    n_agents::Int,
    d::Int,
    anchor_pos::Matrix{Float64},
    measurements::Vector;
    max_iter::Int = 100,
    rho::Float64 = 1.0,
    tol::Float64 = 1e-4
)
    n_anchors = size(anchor_pos, 1)
    
    # Build adjacency graph
    neighbors = [Int[] for _ in 1:n_agents]
    edge_list = []  # (i, j, distance_measured)
    anchor_measurements = [[] for _ in 1:n_agents]  # agent -> [(anchor_id, distance)]
    
    for meas in measurements
        type_i, i, type_j, j, dist_true, dist_measured, is_outlier = meas
        
        if type_i == :agent && type_j == :agent
            if !(j in neighbors[i])
                push!(neighbors[i], j)
                push!(neighbors[j], i)
            end
            push!(edge_list, (i, j, dist_measured))
        elseif type_i == :agent && type_j == :anchor
            push!(anchor_measurements[i], (j, dist_measured))
        end
    end
    
    # Initialize variables
    x = zeros(n_agents, d)  # Agent positions (primal)
    z = Dict{Tuple{Int,Int}, Vector{Float64}}()  # Consensus variables
    u = Dict{Tuple{Int,Int}, Vector{Float64}}()  # Dual variables
    
    # Initialize x to weighted average of anchor positions
    for i in 1:n_agents
        if !isempty(anchor_measurements[i])
            x[i, :] = mean([anchor_pos[a_id, :] for (a_id, _) in anchor_measurements[i]])
        else
            x[i, :] = randn(d)
        end
    end
    
    # Initialize z and u
    for (i, j, _) in edge_list
        key = i < j ? (i, j) : (j, i)
        z[key] = zeros(d)
        u[key] = zeros(d)
    end
    
    # ADMM iterations
    residuals = []
    
    println("\nRunning ADMM iterations...")
    solve_time = @elapsed begin
        for iter in 1:max_iter
            # x-update: Each agent solves local subproblem
            x_new = copy(x)
            
            for i in 1:n_agents
                # Local objective: distance errors + augmented Lagrangian
                
                # Gradient-based update (closed-form is complex)
                grad = zeros(d)
                step_size = 0.1 / rho
                
                # Distance error gradients (agent-anchor)
                for (a_id, d_meas) in anchor_measurements[i]
                    a_pos = anchor_pos[a_id, :]
                    diff = x[i, :] - a_pos
                    dist_current = norm(diff) + 1e-8
                    error = dist_current - d_meas
                    grad += 2 * error * diff / dist_current
                end
                
                # Distance error gradients (agent-agent)
                for j in neighbors[i]
                    # Find edge measurement
                    d_meas = 0.0
                    for (ii, jj, dd) in edge_list
                        if (ii == i && jj == j) || (ii == j && jj == i)
                            d_meas = dd
                            break
                        end
                    end
                    
                    diff = x[i, :] - x[j, :]
                    dist_current = norm(diff) + 1e-8
                    error = dist_current - d_meas
                    grad += 2 * error * diff / dist_current
                end
                
                # Augmented Lagrangian terms
                for j in neighbors[i]
                    key = i < j ? (i, j) : (j, i)
                    sign_i = i < j ? 1.0 : -1.0
                    grad += rho * sign_i * (x[i, :] - z[key] + u[key])
                end
                
                # Update
                x_new[i, :] = x[i, :] - step_size * grad
            end
            
            x_old = copy(x)
            x = x_new
            
            # z-update: Consensus (average)
            for (i, j, _) in edge_list
                key = i < j ? (i, j) : (j, i)
                z[key] = 0.5 * (x[i, :] + x[j, :] + u[key])
            end
            
            # u-update: Dual ascent
            for (i, j, _) in edge_list
                key = i < j ? (i, j) : (j, i)
                u[key] = u[key] + (x[i, :] - x[j, :]) / 2
            end
            
            # Compute residuals
            primal_res = 0.0
            dual_res = 0.0
            
            for (i, j, _) in edge_list
                key = i < j ? (i, j) : (j, i)
                primal_res += norm(x[i, :] - z[key])^2 + norm(x[j, :] - z[key])^2
            end
            primal_res = sqrt(primal_res)
            
            dual_res = rho * norm(x - x_old)
            
            push!(residuals, (primal_res, dual_res))
            
            if iter % 10 == 0
                @printf("Iter %3d: Primal res = %.6f, Dual res = %.6f\n", 
                        iter, primal_res, dual_res)
            end
            
            # Convergence check
            if primal_res < tol && dual_res < tol
                println("Converged at iteration $iter")
                break
            end
        end
    end
    
    return x, residuals, solve_time
end

# Example usage
if abspath(PROGRAM_FILE) == @__FILE__
    include("problem_data.jl")
    include("sdp_relaxation.jl")
    
    println("\nGenerating network data...")
    agent_pos_true, anchor_pos, measurements = generate_network_data(
        n_agents=15,
        n_anchors=4,
        d=2,
        noise_std=0.05,
        outlier_ratio=0.1,
        seed=42
    )
    
    print_network_summary(agent_pos_true, anchor_pos, measurements)
    
    n_agents, d = size(agent_pos_true)
    
    println("\nSolving with Distributed ADMM...")
    agent_pos_est, residuals, solve_time = solve_admm_distributed(
        n_agents, d, anchor_pos, measurements,
        max_iter=100, rho=1.0, tol=1e-4
    )
    
    rmse = compute_rmse(agent_pos_est, agent_pos_true)
    
    println("\n" * "=" ^ 60)
    println("ADMM Solution Results")
    println("=" ^ 60)
    @printf("Solve time: %.3f seconds\n", solve_time)
    @printf("RMSE: %.6f\n", rmse)
    @printf("Iterations: %d\n", length(residuals))
    println("=" ^ 60)
end
