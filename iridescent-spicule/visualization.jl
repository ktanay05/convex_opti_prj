using Plots
using LinearAlgebra
using Printf

"""
Visualize network and localization results.
"""
function plot_network(
    agent_pos_true::Matrix{Float64},
    agent_pos_est::Matrix{Float64},
    anchor_pos::Matrix{Float64},
    measurements::Vector;
    title::String = "Network Localization",
    filename::Union{Nothing, String} = nothing,
    show_outliers::Bool = true
)
    # Plot setup
    plt = plot(
        size=(800, 800),
        xlabel="X",
        ylabel="Y",
        title=title,
        legend=:outertopright,
        aspect_ratio=:equal
    )
    
    # Plot measurements (edges)
    for meas in measurements
        type_i, i, type_j, j, dist_true, dist_measured, is_outlier = meas
        
        if type_i == :agent && type_j == :agent
            pos_i = agent_pos_est[i, :]
            pos_j = agent_pos_est[j, :]
            
            if show_outliers && is_outlier
                plot!(plt, [pos_i[1], pos_j[1]], [pos_i[2], pos_j[2]],
                     color=:red, alpha=0.3, linewidth=2, label="")
            else
                plot!(plt, [pos_i[1], pos_j[1]], [pos_i[2], pos_j[2]],
                     color=:gray, alpha=0.2, linewidth=1, label="")
            end
        end
    end
    
    # Plot anchors
    scatter!(plt, anchor_pos[:, 1], anchor_pos[:, 2],
            marker=:square, markersize=10, color=:green,
            label="Anchors", markerstrokewidth=2)
    
    # Plot true positions
    scatter!(plt, agent_pos_true[:, 1], agent_pos_true[:, 2],
            marker=:circle, markersize=8, color=:blue, alpha=0.5,
            label="True Positions")
    
    # Plot estimated positions
    scatter!(plt, agent_pos_est[:, 1], agent_pos_est[:, 2],
            marker=:diamond, markersize=8, color=:red,
            label="Estimated Positions")
    
    # Plot error vectors
    for i in 1:size(agent_pos_true, 1)
        plot!(plt, [agent_pos_true[i, 1], agent_pos_est[i, 1]],
             [agent_pos_true[i, 2], agent_pos_est[i, 2]],
             arrow=true, color=:orange, alpha=0.6, linewidth=2, label="")
    end
    
    if !isnothing(filename)
        savefig(plt, filename)
        println("Saved plot to $filename")
    end
    
    return plt
end

"""
Plot ADMM convergence.
"""
function plot_admm_convergence(
    residuals::Vector;
    filename::Union{Nothing, String} = nothing
)
    iters = 1:length(residuals)
    primal_res = [r[1] for r in residuals]
    dual_res = [r[2] for r in residuals]
    
    plt = plot(
        size=(800, 500),
        xlabel="Iteration",
        ylabel="Residual",
        title="ADMM Convergence",
        yscale=:log10,
        legend=:topright
    )
    
    plot!(plt, iters, primal_res, linewidth=2, label="Primal Residual", color=:blue)
    plot!(plt, iters, dual_res, linewidth=2, label="Dual Residual", color=:red)
    
    if !isnothing(filename)
        savefig(plt, filename)
        println("Saved plot to $filename")
    end
    
    return plt
end

"""
Compare multiple methods side-by-side.
"""
function plot_comparison(
    agent_pos_true::Matrix{Float64},
    results::Dict{String, Matrix{Float64}},
    anchor_pos::Matrix{Float64};
    filename::Union{Nothing, String} = nothing
)
    n_methods = length(results)
    plts = []
    
    for (method_name, agent_pos_est) in results
        rmse = sqrt(mean((agent_pos_est .- agent_pos_true).^2))
        
        plt = plot(
            size=(400, 400),
            xlabel="X",
            ylabel="Y",
            title="$method_name (RMSE: $(round(rmse, digits=4)))",
            aspect_ratio=:equal,
            legend=false
        )
        
        # Anchors
        scatter!(plt, anchor_pos[:, 1], anchor_pos[:, 2],
                marker=:square, markersize=8, color=:green)
        
        # True positions
        scatter!(plt, agent_pos_true[:, 1], agent_pos_true[:, 2],
                marker=:circle, markersize=6, color=:blue, alpha=0.5)
        
        # Estimated positions
        scatter!(plt, agent_pos_est[:, 1], agent_pos_est[:, 2],
                marker=:diamond, markersize=6, color=:red)
        
        # Error vectors
        for i in 1:size(agent_pos_true, 1)
            plot!(plt, [agent_pos_true[i, 1], agent_pos_est[i, 1]],
                 [agent_pos_true[i, 2], agent_pos_est[i, 2]],
                 arrow=true, color=:orange, alpha=0.6, linewidth=1.5)
        end
        
        push!(plts, plt)
    end
    
    combined = plot(plts..., layout=(1, n_methods), size=(400 * n_methods, 400))
    
    if !isnothing(filename)
        savefig(combined, filename)
        println("Saved comparison to $filename")
    end
    
    return combined
end

# Example usage
if abspath(PROGRAM_FILE) == @__FILE__
    include("problem_data.jl")
    include("sdp_relaxation.jl")
    
    agent_pos_true, anchor_pos, measurements = generate_network_data(seed=42)
    n_agents, d = size(agent_pos_true)
    
    agent_pos_est, _, _ = solve_sdp_gram_matrix(n_agents, d, anchor_pos, measurements)
    
    plot_network(
        agent_pos_true, agent_pos_est, anchor_pos, measurements,
        title="SDP Localization Results",
        filename="results_sdp.png"
    )
end
