#!/usr/bin/env julia

"""
Main script for multi-agent cooperative decentralized localization.

This orchestrates all three approaches:
1. Centralized SDP Relaxation (Baseline)
2. Distributed ADMM
3. MIQP with Outlier Rejection

Usage:
    julia main.jl [--test small|large|outliers]
"""

using Printf
using Statistics

include("problem_data.jl")
include("sdp_jump.jl")  # Using JuMP-based SDP instead of Convex.jl
include("admm_distributed.jl")
include("miqp_outlier.jl")
include("visualization.jl")

"""
Run all methods and compare results.
"""
function run_comparison(;
    n_agents::Int = 20,
    n_anchors::Int = 23 * 20,  # 23 anchors per agent: 460 anchors
    d::Int = 2,
    noise_std::Float64 = 0.1,
    outlier_ratio::Float64 = 0.2,
    outlier_scale::Float64 = 3.0,
    seed::Int = 42
)
    println("\n" * "=" ^ 70)
    println("MULTI-AGENT COOPERATIVE LOCALIZATION - COMPARISON")
    println("=" ^ 70)
    
    # Generate data
    println("\n[1/4] Generating network data...")
    agent_pos_true, anchor_pos, measurements = generate_network_data(
        n_agents=n_agents,
        n_anchors=n_anchors,
        d=d,
        noise_std=noise_std,
        outlier_ratio=outlier_ratio,
        outlier_scale=outlier_scale,
        seed=seed
    )
    print_network_summary(agent_pos_true, anchor_pos, measurements)
    
    results = Dict{String, Matrix{Float64}}()
    timings = Dict{String, Float64}()
    
    # Method 1: Centralized SDP (JuMP)
    println("\n[2/4] Running Centralized SDP with JuMP...")
    try
        pos_sdp, obj_sdp, time_sdp = solve_sdp_jump(
            n_agents, d, anchor_pos, measurements
        )
        results["SDP-JuMP"] = pos_sdp
        timings["SDP-JuMP"] = time_sdp
        rmse_sdp = compute_rmse(pos_sdp, agent_pos_true)
        @printf("  ✓ SDP-JuMP: RMSE = %.6f, Time = %.3fs\n", rmse_sdp, time_sdp)
    catch e
        @warn "SDP-JuMP failed: $e"
        results["SDP-JuMP"] = zeros(n_agents, d)
        timings["SDP-JuMP"] = 0.0
    end
    
    # Method 2: Distributed ADMM
    println("\n[3/4] Running Distributed ADMM...")
    try
        pos_admm, residuals_admm, time_admm = solve_admm_distributed(
            n_agents, d, anchor_pos, measurements,
            max_iter=100, rho=1.0, tol=1e-4
        )
        results["ADMM"] = pos_admm
        timings["ADMM"] = time_admm
        rmse_admm = compute_rmse(pos_admm, agent_pos_true)
        @printf("  ✓ ADMM: RMSE = %.6f, Time = %.3fs\n", rmse_admm, time_admm)
        
        # Plot convergence
        plot_admm_convergence(residuals_admm, filename="admm_convergence.png")
    catch e
        @warn "ADMM failed: $e"
        results["ADMM"] = zeros(n_agents, d)
        timings["ADMM"] = 0.0
    end
    
    # Method 3: MIQP Outlier Rejection
    println("\n[4/4] Running MIQP with Outlier Rejection...")
    try
        # Use SDP-JuMP as warm start
        warm_start = get(results, "SDP-JuMP", nothing)
        
        pos_miqp, outliers_detected, obj_miqp, time_miqp = solve_miqp_outlier_rejection(
            n_agents, d, anchor_pos, measurements,
            lambda_outlier=2.0,
            warm_start=warm_start,
            use_relaxation=true,
            max_time=300.0
        )
        results["MIQP"] = pos_miqp
        timings["MIQP"] = time_miqp
        rmse_miqp = compute_rmse(pos_miqp, agent_pos_true)
        @printf("  ✓ MIQP: RMSE = %.6f, Time = %.3fs\n", rmse_miqp, time_miqp)
        
        # Outlier detection metrics
        metrics = evaluate_outlier_detection(measurements, outliers_detected)
        @printf("    Outlier Detection - Precision: %.3f, Recall: %.3f, F1: %.3f\n",
                metrics.precision, metrics.recall, metrics.f1)
    catch e
        @warn "MIQP failed: $e"
        results["MIQP"] = zeros(n_agents, d)
        timings["MIQP"] = 0.0
    end
    
    # Summary
    println("\n" * "=" ^ 70)
    println("RESULTS SUMMARY")
    println("=" ^ 70)
    @printf("%-15s %15s %15s\n", "Method", "RMSE", "Time (s)")
    println("-" ^ 70)
    
    for method in sort(collect(keys(results)))
        pos_est = results[method]
        rmse = compute_rmse(pos_est, agent_pos_true)
        time = timings[method]
        @printf("%-15s %15.6f %15.3f\n", method, rmse, time)
    end
    println("=" ^ 70)
    
    # Visualization
    println("\nGenerating visualizations...")
    
    # Individual plots for each method
    for (method, pos_est) in results
        plot_network(
            agent_pos_true, pos_est, anchor_pos, measurements,
            title="$method Localization",
            filename="results_$(lowercase(method)).png",
            show_outliers=true
        )
    end
    
    # Comparison plot
    plot_comparison(
        agent_pos_true, results, anchor_pos,
        filename="results_comparison.png"
    )
    
    println("\n✓ All plots saved to current directory")
    println("=" ^ 70)
    
    return results, timings
end

"""
Parse command line arguments and run tests.
"""
function main()
    test_mode = length(ARGS) >= 2 && ARGS[1] == "--test" ? ARGS[2] : "default"
    
    if test_mode == "small"
        println("\n🔬 Running SMALL network test...")
        run_comparison(
            n_agents=5,
            n_anchors=5 * 23,  # 115 anchors (23 per agent)
            noise_std=0.05,
            outlier_ratio=0.1,
            seed=42
        )
        
    elseif test_mode == "large"
        println("\n🔬 Running LARGE network test...")
        run_comparison(
            n_agents=50,
            n_anchors=50 * 23,  # 1150 anchors (23 per agent)
            noise_std=0.1,
            outlier_ratio=0.15,
            seed=42
        )
        
    elseif test_mode == "outliers"
        println("\n🔬 Running OUTLIER ROBUSTNESS test...")
        run_comparison(
            n_agents=20,
            n_anchors=20 * 23,  # 460 anchors (23 per agent)
            noise_std=0.05,
            outlier_ratio=0.3,  # 30% outliers!
            outlier_scale=5.0,
            seed=42
        )
        
    else
        println("\n🚀 Running DEFAULT configuration...")
        run_comparison(
            n_agents=20,
            n_anchors=20 * 23,  # 460 anchors (23 per agent)
            noise_std=0.1,
            outlier_ratio=0.2,
            seed=42
        )
    end
end

# Run if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
