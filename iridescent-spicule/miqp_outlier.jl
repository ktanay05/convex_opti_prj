using JuMP
using LinearAlgebra
using Printf
using Ipopt

# Check if SCIP is available for true MIQP solving
HAS_SCIP = false
try
    @eval using SCIP
    global HAS_SCIP = true
catch
    @warn "SCIP not available. MIQP solver will use continuous relaxation with Ipopt."
end

"""
Solve robust localization with outlier rejection using Mixed-Integer Programming.

Formulation:
- Binary variables bᵢⱼ ∈ {0, 1} indicate if measurement (i,j) is an outlier
- If bᵢⱼ = 1, measurement is trusted (included in objective)
- If bᵢⱼ = 0, measurement is treated as outlier (excluded)

Objective:
    min Σ bᵢⱼ (||xᵢ - xⱼ|| - dᵢⱼ)² + λ Σ (1 - bᵢⱼ)
    
where λ controls the sparsity preference (prefer to trust measurements).

This is a MIQP (Mixed-Integer Quadratic Program). We relax the binary constraints
to [0,1] and then can apply branch-and-bound or use heuristics.

Parameters:
- n_agents: Number of agents
- d: Dimension
- anchor_pos: Anchor positions
- measurements: Measurement list
- lambda_outlier: Penalty for marking measurement as outlier
- warm_start: Initial position guess (e.g., from SDP)
- use_relaxation: If true, relax binary to continuous [0,1]

Returns:
- agent_pos_est: Estimated positions
- outlier_detected: Binary vector indicating detected outliers
- obj_value: Optimal objective value
- solve_time: Total solve time
"""
function solve_miqp_outlier_rejection(
    n_agents::Int,
    d::Int,
    anchor_pos::Matrix{Float64},
    measurements::Vector;
    lambda_outlier::Float64 = 1.0,
    warm_start::Union{Nothing, Matrix{Float64}} = nothing,
    use_relaxation::Bool = !HAS_SCIP,
    max_time::Float64 = 300.0
)
    n_anchors = size(anchor_pos, 1)
    n_meas = length(measurements)
    
    # Build JuMP model
    if HAS_SCIP && !use_relaxation
        model = Model(SCIP.Optimizer)
        set_optimizer_attribute(model, "limits/time", max_time)
        set_optimizer_attribute(model, "display/verblevel", 0)
    else
        # Use Ipopt for continuous relaxation
        model = Model(Ipopt.Optimizer)
        set_optimizer_attribute(model, "print_level", 0)
        set_optimizer_attribute(model, "max_iter", 1000)
    end
    
    # Variables: agent positions
    @variable(model, x[1:n_agents, 1:d])
    
    # Binary variables: trust indicators
    if use_relaxation
        @variable(model, 0 <= b[1:n_meas] <= 1)  # Relaxed to continuous
    else
        @variable(model, b[1:n_meas], Bin)  # True binary
    end
    
    # Warm start if provided
    if !isnothing(warm_start)
        for i in 1:n_agents, j in 1:d
            set_start_value(x[i, j], warm_start[i, j])
        end
    end
    
    # Initialize b to trust all measurements
    for i in 1:n_meas
        set_start_value(b[i], 1.0)
    end
    
    # Objective: weighted distance errors + outlier penalty
    obj_expr = @expression(model, 0.0)
    
    for (idx, meas) in enumerate(measurements)
        type_i, i, type_j, j, dist_true, dist_measured, is_outlier_true = meas
        
        if type_i == :agent && type_j == :agent
            # Agent-agent distance error
            diff = @expression(model, [x[i, k] - x[j, k] for k in 1:d])
            dist_sq = @expression(model, sum(diff[k]^2 for k in 1:d))
            error = @expression(model, dist_sq - dist_measured^2)
            
            # Weighted by trust variable
            obj_expr = @expression(model, obj_expr + b[idx] * error^2)
            
        elseif type_i == :agent && type_j == :anchor
            # Agent-anchor distance error
            diff = @expression(model, [x[i, k] - anchor_pos[j, k] for k in 1:d])
            dist_sq = @expression(model, sum(diff[k]^2 for k in 1:d))
            error = @expression(model, dist_sq - dist_measured^2)
            
            # Weighted by trust variable
            obj_expr = @expression(model, obj_expr + b[idx] * error^2)
        end
        
        # Penalty for not trusting measurement
        obj_expr = @expression(model, obj_expr + lambda_outlier * (1 - b[idx]))
    end
    
    @objective(model, Min, obj_expr)
    
    # Solve
    println("\nSolving MIQP ($(use_relaxation ? "relaxed" : "exact"))...")
    solve_time = @elapsed optimize!(model)
    
    # Extract solution
    agent_pos_est = value.(x)
    outlier_detected = [value(b[i]) < 0.5 for i in 1:n_meas]
    obj_value = objective_value(model)
    
    # Statistics
    n_outliers_detected = sum(outlier_detected)
    n_outliers_true = count(m -> m[7], measurements)
    
    println("Detected $n_outliers_detected outliers (true: $n_outliers_true)")
    
    return agent_pos_est, outlier_detected, obj_value, solve_time
end

"""
Evaluate outlier detection accuracy.
"""
function evaluate_outlier_detection(
    measurements::Vector,
    outlier_detected::Vector{Bool}
)
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    for (idx, meas) in enumerate(measurements)
        is_outlier_true = meas[7]
        is_outlier_pred = outlier_detected[idx]
        
        if is_outlier_true && is_outlier_pred
            true_positives += 1
        elseif !is_outlier_true && is_outlier_pred
            false_positives += 1
        elseif !is_outlier_true && !is_outlier_pred
            true_negatives += 1
        else
            false_negatives += 1
        end
    end
    
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return (
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives
    )
end

# Example usage
if abspath(PROGRAM_FILE) == @__FILE__
    include("problem_data.jl")
    include("sdp_relaxation.jl")
    
    println("\nGenerating network data with outliers...")
    agent_pos_true, anchor_pos, measurements = generate_network_data(
        n_agents=10,
        n_anchors=4,
        d=2,
        noise_std=0.05,
        outlier_ratio=0.3,  # 30% outliers
        outlier_scale=5.0,
        seed=42
    )
    
    print_network_summary(agent_pos_true, anchor_pos, measurements)
    
    n_agents, d = size(agent_pos_true)
    
    # Get warm start from SDP (optional)
    println("\nComputing SDP warm start...")
    warm_start, _, _ = solve_sdp_gram_matrix(n_agents, d, anchor_pos, measurements)
    
    # Solve MIQP
    agent_pos_est, outlier_detected, obj_value, solve_time = solve_miqp_outlier_rejection(
        n_agents, d, anchor_pos, measurements,
        lambda_outlier=2.0,
        warm_start=warm_start,
        use_relaxation=true
    )
    
    rmse = compute_rmse(agent_pos_est, agent_pos_true)
    metrics = evaluate_outlier_detection(measurements, outlier_detected)
    
    println("\n" * "=" ^ 60)
    println("MIQP Solution Results")
    println("=" ^ 60)
    @printf("Solve time: %.3f seconds\n", solve_time)
    @printf("Objective value: %.6f\n", obj_value)
    @printf("RMSE: %.6f\n", rmse)
    println("\nOutlier Detection Metrics:")
    @printf("  Precision: %.3f\n", metrics.precision)
    @printf("  Recall: %.3f\n", metrics.recall)
    @printf("  F1-score: %.3f\n", metrics.f1)
    @printf("  TP/FP/TN/FN: %d/%d/%d/%d\n", 
            metrics.true_positives, metrics.false_positives,
            metrics.true_negatives, metrics.false_negatives)
    println("=" ^ 60)
end
