using JuMP
using LinearAlgebra
using Printf
using Ipopt

"""
Tune MIQP lambda parameter using grid search.

Performs grid search over lambda_outlier values to find optimal trade-off
between accuracy and outlier detection.

Parameters:
- n_agents, d, anchor_pos, measurements: Problem data
- agent_pos_true: Ground truth for evaluation
- lambda_values: Array of lambda values to test
- warm_start: Initial position guess

Returns:
- best_lambda: Best lambda value found
- results: Dictionary with results for each lambda
"""
function tune_miqp_lambda(
    n_agents::Int,
    d::Int,
    anchor_pos::Matrix{Float64},
    measurements::Vector,
    agent_pos_true::Matrix{Float64};
    lambda_values::Vector{Float64} = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    warm_start::Union{Nothing, Matrix{Float64}} = nothing
)
    println("\n" * "=" ^ 70)
    println("MIQP LAMBDA PARAMETER TUNING")
    println("=" ^ 70)
    println("Testing $(length(lambda_values)) lambda values: $lambda_values")
    
    results = Dict()
    best_lambda = 0.0
    best_rmse = Inf
    best_f1 = 0.0
    
    for lambda in lambda_values
        println("\n--> Testing lambda = $lambda")
        
        try
            pos_est, outliers_detected, obj_val, time = solve_miqp_outlier_rejection(
                n_agents, d, anchor_pos, measurements,
                lambda_outlier=lambda,
                warm_start=warm_start,
                use_relaxation=true,
                max_time=60.0  # Shorter time for grid search
            )
            
            # Compute metrics
            rmse = sqrt(mean((pos_est .- agent_pos_true).^2))
            metrics = evaluate_outlier_detection(measurements, outliers_detected)
            
            results[lambda] = (
                rmse=rmse,
                precision=metrics.precision,
                recall=metrics.recall,
                f1=metrics.f1,
                n_detected=sum(outliers_detected),
                time=time
            )
            
            @printf("    RMSE: %.4f, Precision: %.3f, Recall: %.3f, F1: %.3f\n",
                    rmse, metrics.precision, metrics.recall, metrics.f1)
            
            # Update best based on combined metric (weighted RMSE and F1)
            # Lower RMSE is better, higher F1 is better
            score = rmse - 2.0 * metrics.f1  # Penalize low F1
            if score < (best_rmse - 2.0 * best_f1)
                best_lambda = lambda
                best_rmse = rmse
                best_f1 = metrics.f1
            end
            
        catch e
            @warn "Lambda $lambda failed: $e"
            results[lambda] = nothing
        end
    end
    
    println("\n" * "=" ^ 70)
    println("TUNING RESULTS")
    println("=" ^ 70)
    @printf("%-10s %10s %10s %10s %10s %10s\n", 
            "Lambda", "RMSE", "Precision", "Recall", "F1", "Time(s)")
    println("-" ^ 70)
    
    for lambda in sort(collect(keys(results)))
        if !isnothing(results[lambda])
            r = results[lambda]
            marker = lambda == best_lambda ? "* " : "  "
            @printf("%s%-8.1f %10.4f %10.3f %10.3f %10.3f %10.2f\n",
                    marker, lambda, r.rmse, r.precision, r.recall, r.f1, r.time)
        end
    end
    println("=" ^ 70)
    @printf("Best lambda: %.1f (RMSE: %.4f, F1: %.3f)\n", best_lambda, best_rmse, best_f1)
    println("=" ^ 70)
    
    return best_lambda, results
end

# Include helper functions from miqp_outlier.jl
include("miqp_outlier.jl")

# Example usage
if abspath(PROGRAM_FILE) == @__FILE__
    include("problem_data.jl")
    include("sdp_jump.jl")
    
    println("\nGenerating network data...")
    agent_pos_true, anchor_pos, measurements = generate_network_data(
        n_agents=10,
        n_anchors=6,
        d=2,
        noise_std=0.05,
        outlier_ratio=0.3,
        outlier_scale=5.0,
        seed=42
    )
    
    print_network_summary(agent_pos_true, anchor_pos, measurements)
    
    n_agents, d = size(agent_pos_true)
    
    # Get warm start from SDP
    println("\nComputing SDP warm start...")
    warm_start, _, _ = solve_sdp_jump(n_agents, d, anchor_pos, measurements)
    
    # Tune lambda
    best_lambda, tuning_results = tune_miqp_lambda(
        n_agents, d, anchor_pos, measurements, agent_pos_true,
        lambda_values=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        warm_start=warm_start
    )
    
    println("\n✓ Optimal lambda found: $best_lambda")
end
