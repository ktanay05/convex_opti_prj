#!/usr/bin/env julia

"""
Scalability testing for multi-agent localization algorithms.

Tests performance across different network sizes:
- Small: 10 agents
- Medium: 30 agents  
- Large: 50 agents
- XLarge: 100 agents

Measures:
- Solve time
- RMSE accuracy
- Memory usage
- Convergence iterations (for ADMM)
"""

using Printf
using Statistics

include("problem_data.jl")
include("sdp_jump.jl")
include("admm_distributed.jl")

"""
Run scalability test for a given network size.
"""
function run_scalability_test(
    n_agents::Int;
    n_anchors::Int = max(5, div(n_agents, 5)),
    d::Int = 2,
    noise_std::Float64 = 0.1,
    outlier_ratio::Float64 = 0.15,
    seed::Int = 42
)
    println("\n" * "=" ^ 70)
    println("SCALABILITY TEST: $n_agents agents, $n_anchors anchors")
    println("=" ^ 70)
    
    # Generate data
    agent_pos_true, anchor_pos, measurements = generate_network_data(
        n_agents=n_agents,
        n_anchors=n_anchors,
        d=d,
        noise_std=noise_std,
        outlier_ratio=outlier_ratio,
        seed=seed
    )
    
    n_measurements = length(measurements)
    println("Measurements: $n_measurements")
    
    results = Dict()
    
    # Test 1: SDP-JuMP (Centralized)
    print("\n[1/2] SDP-JuMP (Centralized)... ")
    try
        GC.gc()  # Force garbage collection
        mem_before = Base.gc_live_bytes() / 1024^2  # MB
        
        pos_sdp, obj_sdp, time_sdp = solve_sdp_jump(
            n_agents, d, anchor_pos, measurements
        )
        
        mem_after = Base.gc_live_bytes() / 1024^2  # MB
        mem_used = mem_after - mem_before
        
        rmse_sdp = sqrt(mean((pos_sdp .- agent_pos_true).^2))
        
        results["SDP-JuMP"] = (
            time=time_sdp,
            rmse=rmse_sdp,
            memory_mb=mem_used
        )
        
        @printf("✓ (%.2fs, RMSE=%.4f, Mem=%.1fMB)\n", 
                time_sdp, rmse_sdp, mem_used)
    catch e
        println("✗ Failed: $e")
        results["SDP-JuMP"] = nothing
    end
    
    # Test 2: ADMM (Distributed)
    print("[2/2] ADMM (Distributed)... ")
    try
        GC.gc()
        mem_before = Base.gc_live_bytes() / 1024^2
        
        pos_admm, residuals, time_admm = solve_admm_distributed(
            n_agents, d, anchor_pos, measurements,
            max_iter=200, rho=1.0, tol=1e-3
        )
        
        mem_after = Base.gc_live_bytes() / 1024^2
        mem_used = mem_after - mem_before
        
        rmse_admm = sqrt(mean((pos_admm .- agent_pos_true).^2))
        n_iters = length(residuals)
        
        results["ADMM"] = (
            time=time_admm,
            rmse=rmse_admm,
            memory_mb=mem_used,
            iterations=n_iters
        )
        
        @printf("✓ (%.2fs, RMSE=%.4f, Iters=%d, Mem=%.1fMB)\n",
                time_admm, rmse_admm, n_iters, mem_used)
    catch e
        println("✗ Failed: $e")
        results["ADMM"] = nothing
    end
    
    return results
end

"""
Run full scalability suite and generate report.
"""
function run_scalability_suite()
    network_sizes = [10, 30, 50, 100]
    
    println("\n" * "=" ^ 70)
    println("SCALABILITY TESTING SUITE")
    println("=" ^ 70)
    println("Testing network sizes: $network_sizes")
    
    all_results = Dict()
    
    for n_agents in network_sizes
        results = run_scalability_test(n_agents, seed=42)
        all_results[n_agents] = results
    end
    
    # Generate summary report
    println("\n\n" * "=" ^ 70)
    println("SCALABILITY SUMMARY REPORT")
    println("=" ^ 70)
    
    println("\n--- SDP-JuMP (Centralized) ---")
    @printf("%-10s %12s %12s %15s\n", "Agents", "Time (s)", "RMSE", "Memory (MB)")
    println("-" ^ 70)
    for n_agents in sort(collect(keys(all_results)))
        r = all_results[n_agents]["SDP-JuMP"]
        if !isnothing(r)
            @printf("%-10d %12.2f %12.4f %15.1f\n",
                    n_agents, r.time, r.rmse, r.memory_mb)
        else
            @printf("%-10d %12s %12s %15s\n", n_agents, "FAILED", "-", "-")
        end
    end
    
    println("\n--- ADMM (Distributed) ---")
    @printf("%-10s %12s %12s %12s %15s\n", 
            "Agents", "Time (s)", "RMSE", "Iterations", "Memory (MB)")
    println("-" ^ 70)
    for n_agents in sort(collect(keys(all_results)))
        r = all_results[n_agents]["ADMM"]
        if !isnothing(r)
            @printf("%-10d %12.2f %12.4f %12d %15.1f\n",
                    n_agents, r.time, r.rmse, r.iterations, r.memory_mb)
        else
            @printf("%-10d %12s %12s %12s %15s\n", 
                    n_agents, "FAILED", "-", "-", "-")
        end
    end
    
    # Compute speedup ratios
    println("\n--- Speedup Analysis ---")
    @printf("%-10s %20s\n", "Agents", "ADMM vs SDP Speedup")
    println("-" ^ 70)
    for n_agents in sort(collect(keys(all_results)))
        sdp_r = all_results[n_agents]["SDP-JuMP"]
        admm_r = all_results[n_agents]["ADMM"]
        if !isnothing(sdp_r) && !isnothing(admm_r) && sdp_r.time > 0
            speedup = sdp_r.time / admm_r.time
            @printf("%-10d %20.2fx faster\n", n_agents, speedup)
        end
    end
    
    println("=" ^ 70)
    
    return all_results
end

# Run if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    results = run_scalability_suite()
    println("\n✓ Scalability testing complete!")
end
