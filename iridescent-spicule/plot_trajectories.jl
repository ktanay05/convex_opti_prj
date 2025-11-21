#!/usr/bin/env julia

"""
Create focused visualization of vehicle trajectories.
"""

using Plots
using Printf
using Random
using LinearAlgebra

include("problem_data.jl")
include("agent_dynamics.jl")
include("ekf_localization.jl")

"""
Generate and plot Dubins vehicle trajectories.
"""
function plot_trajectories(;
    n_agents::Int=5,
    duration::Float64=10.0,
    dt::Float64=0.1,
    trajectory_type::Symbol=:circular,
    seed::Int=42
)
    Random.seed!(seed)
    
    # Generate initial positions
    _, anchor_pos, _ = generate_network_data(n_agents=n_agents, n_anchors=5*23, d=2, seed=seed)
    initial_positions = [10.0 * (rand(2) .- 0.5) for _ in 1:n_agents]
    
    # Create agents
    agents = [create_agent(i, initial_positions[i], 2π*rand(), 2.0) for i in 1:n_agents]
    
    # Generate trajectories
    n_steps = round(Int, duration / dt)
    trajectories = [Vector{Vector{Float64}}() for _ in 1:n_agents]
    headings = [Vector{Float64}() for _ in 1:n_agents]
    
    for step in 1:n_steps
        for i in 1:n_agents
            # Propagate
            if trajectory_type == :circular
                ω = 0.3
                agents[i].turn_rate = ω
                agents[i].x[5] = ω
                agents[i].x = propagate_state_nonlinear(agents[i].x, dt)
            elseif trajectory_type == :linear
                agents[i].turn_rate = 0.0
                agents[i].x[5] = 0.0
                agents[i].x = propagate_state_nonlinear(agents[i].x, dt)
            elseif trajectory_type == :varied
                # Varied maneuvers
                if step < 30
                    ω = 0.5  # Right turn
                elseif step < 60
                    ω = 0.0  # Straight
                else
                    ω = -0.3  # Left turn
                end
                agents[i].turn_rate = ω
                agents[i].x[5] = ω
                agents[i].x = propagate_state_nonlinear(agents[i].x, dt)
            end
            
            agents[i].position = agents[i].x[1:2]
            agents[i].heading = agents[i].x[3]
            
            push!(trajectories[i], copy(agents[i].position))
            push!(headings[i], agents[i].heading)
        end
    end
    
    return trajectories, headings, anchor_pos, initial_positions
end

"""
Create publication-quality trajectory plots.
"""
function create_trajectory_plots()
    # Plot 1: Circular trajectories
    println("Generating circular trajectories...")
    traj_circ, head_circ, anchors, init_pos = plot_trajectories(
        n_agents=5, duration=10.0, trajectory_type=:circular
    )
    
    p1 = plot(xlabel="X (m)", ylabel="Y (m)", title="Circular Trajectories (ω = 0.3 rad/s)",
              aspect_ratio=:equal, legend=:outertopright, size=(600, 600))
    
    # Plot anchors
    scatter!(p1, anchors[:, 1], anchors[:, 2], marker=:square, markersize=2, 
             color=:gray, alpha=0.2, label="Anchors (115)")
    
    # Plot trajectories
    colors = [:blue, :red, :green, :orange, :purple]
    for i in 1:length(traj_circ)
        pts = hcat(traj_circ[i]...)'
        plot!(p1, pts[:, 1], pts[:, 2], color=colors[i], linewidth=3, 
              label="Agent $i", alpha=0.8)
        
        # Start point
        scatter!(p1, [init_pos[i][1]], [init_pos[i][2]], 
                marker=:circle, markersize=8, color=colors[i], label="")
        
        # End point with heading arrow
        final_pos = traj_circ[i][end]
        final_head = head_circ[i][end]
        arrow_len = 1.0
        arrow_end = final_pos + arrow_len * [cos(final_head), sin(final_head)]
        plot!(p1, [final_pos[1], arrow_end[1]], [final_pos[2], arrow_end[2]],
              arrow=true, color=colors[i], linewidth=2, label="")
    end
    
    savefig(p1, "trajectories_circular.png")
    println("✓ Saved trajectories_circular.png")
    
    # Plot 2: Linear trajectories
    println("Generating linear trajectories...")
    traj_lin, head_lin, _, _ = plot_trajectories(
        n_agents=5, duration=10.0, trajectory_type=:linear, seed=43
    )
    
    p2 = plot(xlabel="X (m)", ylabel="Y (m)", title="Linear Trajectories (ω = 0 rad/s)",
              aspect_ratio=:equal, legend=:outertopright, size=(600, 600))
    
    scatter!(p2, anchors[:, 1], anchors[:, 2], marker=:square, markersize=2,
             color=:gray, alpha=0.2, label="Anchors")
    
    for i in 1:length(traj_lin)
        pts = hcat(traj_lin[i]...)'
        plot!(p2, pts[:, 1], pts[:, 2], color=colors[i], linewidth=3,
              label="Agent $i", alpha=0.8)
        scatter!(p2, [pts[1, 1]], [pts[1, 2]], marker=:circle, markersize=8,
                color=colors[i], label="")
    end
    
    savefig(p2, "trajectories_linear.png")
    println("✓ Saved trajectories_linear.png")
    
    # Plot 3: Varied maneuvers
    println("Generating varied maneuver trajectories...")
    traj_var, head_var, _, _ = plot_trajectories(
        n_agents=3, duration=10.0, trajectory_type=:varied, seed=44
    )
    
    p3 = plot(xlabel="X (m)", ylabel="Y (m)", title="Varied Maneuvers (turn → straight → turn)",
              aspect_ratio=:equal, legend=:outertopright, size=(600, 600))
    
    for i in 1:length(traj_var)
        pts = hcat(traj_var[i]...)'
        
        # Color segments differently
        n = size(pts, 1)
        seg1 = 1:30
        seg2 = 30:60
        seg3 = 60:n
        
        plot!(p3, pts[seg1, 1], pts[seg1, 2], color=colors[i], linewidth=3,
              linestyle=:solid, label="Agent $i (turn right)", alpha=0.8)
        plot!(p3, pts[seg2, 1], pts[seg2, 2], color=colors[i], linewidth=3,
              linestyle=:dash, label="", alpha=0.8)
        plot!(p3, pts[seg3, 1], pts[seg3, 2], color=colors[i], linewidth=3,
              linestyle=:dot, label="", alpha=0.8)
        
        scatter!(p3, [pts[1, 1]], [pts[1, 2]], marker=:circle, markersize=8,
                color=colors[i], label="")
    end
    
    savefig(p3, "trajectories_varied.png")
    println("✓ Saved trajectories_varied.png")
    
    # Combined plot
    p_combined = plot(p1, p2, p3, layout=(1, 3), size=(1800, 600))
    savefig(p_combined, "trajectories_all.png")
    println("✓ Saved trajectories_all.png")
    
    return p_combined
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    println("\n=== Generating Vehicle Trajectory Visualizations ===\n")
    
    create_trajectory_plots()
    
    println("\n✅ All trajectory plots created:")
    println("  • trajectories_circular.png - Circular paths (constant turn)")
    println("  • trajectories_linear.png - Straight paths")  
    println("  • trajectories_varied.png - Mixed maneuvers")
    println("  • trajectories_all.png - All three combined")
    println("\nThese show TRUE vehicle paths for Dubins vehicle model!")
end
