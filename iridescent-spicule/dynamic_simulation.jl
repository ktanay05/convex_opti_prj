#!/usr/bin/env julia

"""
Dynamic cooperative localization trajectory simulation.

Simulates multiple agents moving over time, each running local EKF
with range measurements to anchors and neighboring agents.
"""

using Printf
using Statistics
using Plots

include("problem_data.jl")
include("agent_dynamics.jl") 
include("ekf_localization.jl")
include("sdp_jump.jl")  # For initialization

"""
Run dynamic simulation with EKF tracking.
"""
function run_dynamic_simulation(;
    n_agents::Int = 5,
    n_anchors::Int = 5 * 23,  # 23 per agent
    duration::Float64 = 10.0,
    dt::Float64 = 0.1,
    rc::Float64 = 5.0,
    σ_anchor::Float64 = 0.05,
    σ_range::Float64 = 0.1,
    σ_process::Float64 = 0.1,
    trajectory_type::Symbol = :circular,
    use_static_init::Bool = true,
    seed::Int = 42
)
    Random.seed!(seed)
    
    println("\n" * "=" ^ 70)
    println("DYNAMIC COOPERATIVE LOCALIZATION SIMULATION")
    println("=" ^ 70)
    println("Agents: $n_agents")
    println("Anchors: $n_anchors")
    println("Duration: $(duration)s, dt: $(dt)s")
    println("Communication range (rc): $rc")
    println("Trajectory type: $trajectory_type")
    println("=" ^ 70)
    
    # Generate static anchors
    _, anchor_pos, _ = generate_network_data(
        n_agents=n_agents, n_anchors=n_anchors, d=2, seed=seed
    )
    
    # Initialize agents with static localization if requested
    if use_static_init
        println("\n[1/3] Computing initial positions with static MIQP...")
        agent_pos_true, _, measurements = generate_network_data(
            n_agents=n_agents, n_anchors=n_anchors, d=2,
            noise_std=0.01, outlier_ratio=0.0, seed=seed
        )
        
        # Use SDP for quick initialization
       pos_init, _, _ = solve_sdp_jump(n_agents, 2, anchor_pos, measurements)
        initial_positions = [pos_init[i, :] for i in 1:n_agents]
    else
        # Random initialization
        initial_positions = [10.0 * rand(2) for _ in 1:n_agents]
    end
    
    # Create agents with initial velocities
    println("\n[2/3] Initializing $(n_agents) agents...")
    agents = AgentState[]
    ground_truth_trajectories = []
    
    for i in 1:n_agents
        heading = 2π * rand()  # Random initial heading
        speed = 2.0  # Fixed speed for all agents
        agent = create_agent(i, initial_positions[i], heading, speed)
        push!(agents, agent)
        push!(ground_truth_trajectories, Vector{Vector{Float64}}())
    end
    
    # Simulation loop
    println("\n[3/3] Running simulation...")
    n_steps = round(Int, duration / dt)
    
    # Storage for results
    estimated_trajectories = [Vector{Vector{Float64}}() for _ in 1:n_agents]
    position_errors = zeros(n_agents, n_steps)
    velocity_errors = zeros(n_agents, n_steps)
    neighbor_counts = zeros(n_agents, n_steps)
    
    # True agent states (for comparison)
    true_agents = deepcopy(agents)
    
    for step in 1:n_steps
        t = (step - 1) * dt
        
        # Propagate true state using Dubins dynamics
        for i in 1:n_agents
            # Move according to trajectory type
            if trajectory_type == :circular
                # Circular trajectory: constant turn rate
                ω = 0.3  # Turn rate (rad/s)
                true_agents[i].turn_rate = ω
                true_agents[i].x[5] = ω
                
                # Propagate with nonlinear dynamics
                true_agents[i].x = propagate_state_nonlinear(true_agents[i].x, dt)
                true_agents[i].position = true_agents[i].x[1:2]
                true_agents[i].heading = true_agents[i].x[3]
                
            elseif trajectory_type == :linear
                # Straight line: zero turn rate
                true_agents[i].turn_rate = 0.0
                true_agents[i].x[5] = 0.0
                propagate_state(true_agents[i], dt, 0.0, add_noise=false)
            end
            
            push!(ground_truth_trajectories[i], copy(true_agents[i].position))
        end
        
        # EKF update for each agent
        for i in 1:n_agents
            # Run EKF step using true agent positions for generating measurements
            ekf_step!(agents[i], dt, anchor_pos, true_agents,
                     rc=rc, σ_anchor=σ_anchor, σ_range=σ_range,
                     σ_process=σ_process, enable_fusion=true)
            
            # Record results
            push!(estimated_trajectories[i], copy(agents[i].position))
            position_errors[i, step] = norm(agents[i].position - true_agents[i].position)
            
            # Velocity error (compute from heading and speed)
            est_vel = agents[i].speed * [cos(agents[i].heading), sin(agents[i].heading)]
            true_vel = true_agents[i].speed * [cos(true_agents[i].heading), sin(true_agents[i].heading)]
            velocity_errors[i, step] = norm(est_vel - true_vel)
            
            neighbors = find_neighbors(i, true_agents, rc)
            neighbor_counts[i, step] = length(neighbors)
        end
        
        if step % 10 == 0
            avg_pos_error = mean(position_errors[:, step])
            @printf("  Step %3d/%d: t=%.1fs, Avg pos error: %.4f\n",
                    step, n_steps, t, avg_pos_error)
        end
    end
    
    # Compute statistics
    final_pos_errors = position_errors[:, end]
    avg_pos_error = mean(final_pos_errors)
    max_pos_error = maximum(final_pos_errors)
    
    avg_vel_error = mean(velocity_errors[:, end])
    avg_neighbors = mean(neighbor_counts)
    
    println("\n" * "=" ^ 70)
    println("SIMULATION RESULTS")
    println("=" ^ 70)
    @printf("Final Position RMSE: %.4f\n", sqrt(mean(final_pos_errors.^2)))
    @printf("Final Position Error (avg/max): %.4f / %.4f\n", avg_pos_error, max_pos_error)
    @printf("Final Velocity Error (avg): %.4f\n", avg_vel_error)
    @printf("Average neighbors per agent: %.1f\n", avg_neighbors)
    println("=" ^ 70)
    
    return (
        true_trajectories=ground_truth_trajectories,
        est_trajectories=estimated_trajectories,
        position_errors=position_errors,
        velocity_errors=velocity_errors,
        neighbor_counts=neighbor_counts,
        anchors=anchor_pos,
        agents=agents,
        true_agents=true_agents
    )
end

"""
Visualize dynamic simulation results.
"""
function visualize_dynamic_results(results; save_path::String="dynamic_simulation.png")
    true_traj = results.true_trajectories
    est_traj = results.est_trajectories
    anchors = results.anchors
    n_agents = length(true_traj)
    
    # Create plot
    plt = plot(size=(1200, 800), layout=(2, 2))
    
    # Subplot 1: Trajectories
    plot!(plt[1], xlabel="X", ylabel="Y", title="Agent Trajectories",
          aspect_ratio=:equal, legend=:outertopright)
    
    # Plot anchors
    scatter!(plt[1], anchors[:, 1], anchors[:, 2],
            marker=:square, markersize=2, color=:green, alpha=0.3,
            label="Anchors")
    
    # Plot trajectories
    colors = [:blue, :red, :orange, :purple, :cyan]
    for i in 1:n_agents
        true_pts = hcat(true_traj[i]...)'
        est_pts = hcat(est_traj[i]...)'
        
        col = colors[mod1(i, length(colors))]
        plot!(plt[1], true_pts[:, 1], true_pts[:, 2],
              color=col, linewidth=2, alpha=0.7, label="Agent $i (true)")
        plot!(plt[1], est_pts[:, 1], est_pts[:, 2],
              color=col, linestyle=:dash, linewidth=2, label="Agent $i (est)")
    end
    
    # Subplot 2: Position errors over time
    plot!(plt[2], xlabel="Time Step", ylabel="Position Error",
          title="Position Error Evolution")
    
    for i in 1:n_agents
        col = colors[mod1(i, length(colors))]
        plot!(plt[2], results.position_errors[i, :],
              color=col, linewidth=2, label="Agent $i")
    end
    
    # Subplot 3: Velocity errors
    plot!(plt[3], xlabel="Time Step", ylabel="Velocity Error",
          title="Velocity Error Evolution")
    
    for i in 1:n_agents
        col = colors[mod1(i, length(colors))]
        plot!(plt[3], results.velocity_errors[i, :],
              color=col, linewidth=2, label="Agent $i")
    end
    
    # Subplot 4: Neighbor counts
    plot!(plt[4], xlabel="Time Step", ylabel="Number of Neighbors",
          title="Communication Graph Connectivity")
    
    for i in 1:n_agents
        col = colors[mod1(i, length(colors))]
        plot!(plt[4], results.neighbor_counts[i, :],
              color=col, linewidth=2, label="Agent $i")
    end
    
    savefig(plt, save_path)
    println("\n✓ Visualization saved to $save_path")
    
    return plt
end

# Run simulation if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    results = run_dynamic_simulation(
        n_agents=5,
        n_anchors=5 * 23,
        duration=10.0,
        dt=0.1,
        rc=5.0,
        trajectory_type=:circular,
        use_static_init=true
    )
    
    visualize_dynamic_results(results, save_path="dynamic_simulation.png")
end
