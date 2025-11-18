using JuMP
using HiGHS
using Polyhedra
using CDDLib
using Graphs
using Plots
using LinearAlgebra
using Random
using Statistics
using VoronoiDelaunay

# Mapping functions
domain_min = 0.0
domain_max = 4.0
vd_min = 1.1
vd_max = 1.9

function to_vd(val)
    return vd_min + (val - domain_min) / (domain_max - domain_min) * (vd_max - vd_min)
end

function from_vd(val)
    return domain_min + (val - vd_min) / (vd_max - vd_min) * (domain_max - domain_min)
end

# ==============================================================================
# 1. Environment Setup & Decomposition
# ==============================================================================

# Helper to create a convex polygon (obstacle) using Convex Hull
function random_convex_polygon(center, scale, npoints=10)
    # Generate random points around the center
    points_cloud = [center .+ scale .* (rand(2) .- 0.5) for _ in 1:npoints]
    return points_cloud
end

function make_polyhedron(vertices)
    v = vrep(vertices)
    return polyhedron(v, CDDLib.Library())
end

function generate_environment()
    # Generate Obstacles
    obstacles = []
    max_obstacles = 5
    tries = 0
    while length(obstacles) < max_obstacles && tries < 1000
        c = [0.5 + 3.0*rand(), 0.5 + 3.0*rand()]
        # Generate random points for hull
        verts = random_convex_polygon(c, 0.8, 8)
        obs = make_polyhedron(verts)
        
        # Check if valid (non-empty, reasonable area)
        if volume(obs) > 0.01 && all(isempty(intersect(obs, o)) for o in obstacles)
            push!(obstacles, obs)
        end
        tries += 1
    end

    # Helper: Closest point on polygon to point
    function closest_point_on_polygon(pt, poly_verts)
        min_dist_sq = Inf
        closest_pt = pt
        
        # Ensure vertices are ordered for edge iteration
        cen = mean(poly_verts)
        sorted_verts = sort(poly_verts, by = v -> atan(v[2]-cen[2], v[1]-cen[1]))
        
        # Check edges
        for i in 1:length(sorted_verts)
            p1 = sorted_verts[i]
            p2 = sorted_verts[i == length(sorted_verts) ? 1 : i+1]
            
            # Project pt onto line segment p1-p2
            v = p2 - p1
            w = pt - p1
            c1 = dot(w, v)
            if c1 <= 0
                cp = p1
            else
                c2 = dot(v, v)
                if c2 <= c1
                    cp = p2
                else
                    b = c1 / c2
                    cp = p1 + b * v
                end
            end
            
            d_sq = sum((pt - cp).^2)
            if d_sq < min_dist_sq
                min_dist_sq = d_sq
                closest_pt = cp
            end
        end
        return closest_pt
    end

    # Generate seeds using Greedy Maximal Clearance (GVD approximation)
    grid_res = 40
    xs = range(0.05, 3.95, length=grid_res)
    ys = range(0.05, 3.95, length=grid_res)
    grid_pts = [[x, y] for x in xs, y in ys]
    dists = zeros(grid_res, grid_res)
    valid_mask = trues(grid_res, grid_res)
    
    for i in 1:grid_res, j in 1:grid_res
        pt = grid_pts[i, j]
        
        # Check if inside any obstacle
        in_obs = false
        for obs in obstacles
            if in(pt, obs)
                in_obs = true
                break
            end
        end
        
        if in_obs
            dists[i, j] = -1.0
            valid_mask[i, j] = false
            continue
        end
        
        # Compute min distance to any obstacle boundary
        min_d = Inf
        for obs in obstacles
            verts = collect(points(vrep(obs)))
            if isempty(verts); continue; end
            cp = closest_point_on_polygon(pt, verts)
            d = norm(pt - cp)
            if d < min_d
                min_d = d
            end
        end
        
        # Also check distance to domain bounds
        d_bounds = min(pt[1], 4.0 - pt[1], pt[2], 4.0 - pt[2])
        min_d = min(min_d, d_bounds)
        
        dists[i, j] = min_d
    end
    
    seeds = []
    num_seeds = 30 
    
    # Greedy selection
    for _ in 1:num_seeds
        max_d = -1.0
        max_idx = (-1, -1)
        
        for i in 1:grid_res, j in 1:grid_res
            if valid_mask[i, j] && dists[i, j] > max_d
                max_d = dists[i, j]
                max_idx = (i, j)
            end
        end
        
        if max_d <= 0.1; break; end
        
        seed_pt = grid_pts[max_idx[1], max_idx[2]]
        push!(seeds, seed_pt)
        
        inhibition_radius = max_d * 1.5 
        
        for i in 1:grid_res, j in 1:grid_res
            if valid_mask[i, j]
                if norm(grid_pts[i, j] - seed_pt) < inhibition_radius
                    dists[i, j] = -1.0
                end
            end
        end
    end
    
    println("Generated $(length(seeds)) seeds using GVD heuristic.")
    
    regions = []
    bbox = make_polyhedron([[0.0,0.0], [4.0,0.0], [4.0,4.0], [0.0,4.0]])
    
    for (i, site) in enumerate(seeds)
        in_obs = false
        for obs in obstacles
             if in(site, obs)
                 in_obs = true
                 break
             end
        end
        if in_obs; continue; end

        cell = bbox
        
        # 1. Voronoi Bisectors
        for (j, other) in enumerate(seeds)
            if i == j; continue; end
            normal = other - site
            rhs = 0.5 * (dot(other, other) - dot(site, site))
            hs = HalfSpace(normal, rhs)
            cell = intersect(cell, hs)
        end
        
        # 2. Separating Hyperplanes
        for obs in obstacles
            verts = collect(points(vrep(obs)))
            if isempty(verts); continue; end
            cp = closest_point_on_polygon(site, verts)
            normal = site - cp
            dist = norm(normal)
            if dist < 1e-6; continue; end
            normal = normal / dist
            margin = 0.01
            cp_safe = cp + margin * normal
            hs = HalfSpace(-normal, -dot(normal, cp_safe))
            cell = intersect(cell, hs)
        end
        
        if !isempty(cell)
            push!(regions, cell)
        end
    end
    
    return obstacles, regions, seeds
end

# Retry loop
max_retries = 20
obstacles = []
regions = []
adj_matrix = Matrix{Bool}(undef, 0, 0)
start_region_idx = 0
goal_region_idx = 0
start_pos = [0.0, 0.0]
goal_pos = [0.0, 0.0]
connected = false

for attempt in 1:max_retries
    println("Attempt $attempt...")
    global obstacles, regions, seeds = generate_environment()
    
    if isempty(regions)
        continue
    end

    global n_regions = length(regions)
    global adj_matrix = zeros(Bool, n_regions, n_regions)

    # Build graph using Delaunay Triangulation (Dual of Voronoi)
    tess = DelaunayTessellation()
    seed_map = Dict{Point2D, Int}()
    
    for (i, s) in enumerate(seeds)
        x_vd = to_vd(s[1])
        y_vd = to_vd(s[2])
        pt = Point2D(x_vd, y_vd)
        push!(tess, pt)
        seed_map[pt] = i
    end
    
    for edge in delaunayedges(tess)
        p_a = geta(edge)
        p_b = getb(edge)
        if haskey(seed_map, p_a) && haskey(seed_map, p_b)
            i = seed_map[p_a]
            j = seed_map[p_b]
            if !isempty(intersect(regions[i], regions[j]))
                adj_matrix[i, j] = true
                adj_matrix[j, i] = true
            end
        end
    end
    
    # Pick Start/Goal
    global start_region_idx = 1
    global goal_region_idx = n_regions
    
    # BFS Check
    q = [start_region_idx]
    visited = Set([start_region_idx])
    found = false
    while !isempty(q)
        u = popfirst!(q)
        if u == goal_region_idx
            found = true
            break
        end
        for v in 1:n_regions
            if adj_matrix[u, v] && !(v in visited)
                push!(visited, v)
                push!(q, v)
            end
        end
    end
    
    if found
        println("Connected environment found on attempt $attempt.")
        global connected = true
        pts_s = collect(points(vrep(regions[start_region_idx])))
        pts_g = collect(points(vrep(regions[goal_region_idx])))
        global start_pos = mean(pts_s)
        global goal_pos = mean(pts_g)
        break
    end
end

if !connected
    error("Could not generate a connected environment after $max_retries attempts.")
end

println("Generated $(length(obstacles)) obstacles and $(length(regions)) free regions.")
println("Start Region: $start_region_idx (Pos: $start_pos)")
println("Goal Region: $goal_region_idx (Pos: $goal_pos)")

# ==============================================================================
# 3. GCS Trajectory Optimization (MICP)
# ==============================================================================

# Parameters
bezier_degree = 2 # Quadratic Bezier
M = 100.0 # Big-M constant

model = Model(HiGHS.Optimizer)
set_silent(model)

# Variables
@variable(model, y[1:n_regions], Bin)
@variable(model, z[1:n_regions, 1:n_regions], Bin)
@variable(model, x[1:n_regions, 0:bezier_degree, 1:2])

# Constraints

# 1. Flow Conservation
@constraint(model, sum(z[start_region_idx, j] for j in 1:n_regions) - sum(z[j, start_region_idx] for j in 1:n_regions) == 1)
@constraint(model, sum(z[goal_region_idx, j] for j in 1:n_regions) - sum(z[j, goal_region_idx] for j in 1:n_regions) == -1)

for i in 1:n_regions
    if i != start_region_idx && i != goal_region_idx
        @constraint(model, sum(z[i, j] for j in 1:n_regions) - sum(z[j, i] for j in 1:n_regions) == 0)
    end
end

for i in 1:n_regions
    @constraint(model, y[i] >= sum(z[i, j] for j in 1:n_regions))
    @constraint(model, y[i] >= sum(z[j, i] for j in 1:n_regions))
    if i == start_region_idx || i == goal_region_idx
        @constraint(model, y[i] == 1)
    end
end

# 2. Containment
for i in 1:n_regions
    h = hrep(regions[i])
    for halfspace in halfspaces(h)
        a = halfspace.a
        b = halfspace.β
        for k in 0:bezier_degree
            @constraint(model, dot(a, x[i, k, :]) <= b + M * (1 - y[i]))
            @constraint(model, x[i, k, 1] <= M * y[i])
            @constraint(model, x[i, k, 1] >= -M * y[i])
            @constraint(model, x[i, k, 2] <= M * y[i])
            @constraint(model, x[i, k, 2] >= -M * y[i])
        end
    end
end

# 3. Continuity (C0 and C1)
for i in 1:n_regions, j in 1:n_regions
    if adj_matrix[i, j]
        # C0: x[i, degree] == x[j, 0]
        for d in 1:2
            @constraint(model, x[i, bezier_degree, d] - x[j, 0, d] <= M * (1 - z[i, j]))
            @constraint(model, x[i, bezier_degree, d] - x[j, 0, d] >= -M * (1 - z[i, j]))
        end
        
        # C1: Heading Consistency
        # Enforce continuity of the derivative direction/magnitude.
        # For quadratic Bezier: Tangent at end is proportional to (P2 - P1).
        # Tangent at start is proportional to (P1 - P0).
        # We enforce (x[i, K] - x[i, K-1]) == (x[j, 1] - x[j, 0])
        for d in 1:2
            diff_i = x[i, bezier_degree, d] - x[i, bezier_degree-1, d]
            diff_j = x[j, 1, d] - x[j, 0, d]
            
            @constraint(model, diff_i - diff_j <= M * (1 - z[i, j]))
            @constraint(model, diff_i - diff_j >= -M * (1 - z[i, j]))
        end
    else
        @constraint(model, z[i, j] == 0)
    end
end

# 4. Start and Goal Constraints
@constraint(model, x[start_region_idx, 0, 1] == start_pos[1])
@constraint(model, x[start_region_idx, 0, 2] == start_pos[2])
@constraint(model, x[goal_region_idx, bezier_degree, 1] == goal_pos[1])
@constraint(model, x[goal_region_idx, bezier_degree, 2] == goal_pos[2])

# 5. Objective: Minimize Path Length (L1 Norm)
@variable(model, t[1:n_regions, 1:bezier_degree, 1:2] >= 0)

for i in 1:n_regions
    for k in 1:bezier_degree
        for d in 1:2
            @constraint(model, t[i, k, d] >= x[i, k, d] - x[i, k-1, d])
            @constraint(model, t[i, k, d] >= -(x[i, k, d] - x[i, k-1, d]))
        end
    end
end

@objective(model, Min, sum(t))

println("Solving GCS optimization...")
optimize!(model)
println("Status: ", termination_status(model))

# ==============================================================================
# 4. Visualization
# ==============================================================================

plt = plot(title="GCS Path Planning", size=(700, 700), legend=false, aspect_ratio=:equal)

# Plot Obstacles
for obs in obstacles
    pts = collect(points(vrep(obs)))
    if !isempty(pts)
        cen = mean(pts)
        sort!(pts, by = v -> atan(v[2]-cen[2], v[1]-cen[1]))
        push!(pts, pts[1]) 
        plot!(plt, [v[1] for v in pts], [v[2] for v in pts], seriestype=:shape, color=:red, fillalpha=0.5)
    end
end

# Plot Regions
for (idx, r) in enumerate(regions)
    pts = collect(points(vrep(r)))
    if !isempty(pts)
        cen = mean(pts)
        sort!(pts, by = v -> atan(v[2]-cen[2], v[1]-cen[1]))
        push!(pts, pts[1])
        plot!(plt, [v[1] for v in pts], [v[2] for v in pts], seriestype=:shape, color=:blue, fillalpha=0.1, linealpha=0.2)
        annotate!(plt, cen[1], cen[2], text("$idx", 8, :gray))
    end
end

# Plot Edges (Graph)
for i in 1:n_regions, j in 1:n_regions
    if adj_matrix[i, j]
        c1 = mean(collect(points(vrep(regions[i]))))
        c2 = mean(collect(points(vrep(regions[j]))))
        plot!(plt, [c1[1], c2[1]], [c1[2], c2[2]], color=:gray, alpha=0.3)
    end
end

# Plot Solution
if termination_status(model) == MOI.OPTIMAL
    println("Solution found!")
    
    local current = start_region_idx
    path_regions = [current]
    
    while current != goal_region_idx
        next_node = nothing
        for j in 1:n_regions
            if value(z[current, j]) > 0.5
                next_node = j
                break
            end
        end
        if isnothing(next_node)
            break
        end
        push!(path_regions, next_node)
        current = next_node
    end
    println("Path of regions: ", path_regions)
    
    for i in path_regions
        cps = [value.(x[i, k, :]) for k in 0:bezier_degree]
        
        plot!(plt, [p[1] for p in cps], [p[2] for p in cps], 
              color=:magenta, linestyle=:dash, linewidth=1, label="")
        
        scatter!(plt, [p[1] for p in cps], [p[2] for p in cps], 
                 color=:magenta, markersize=4, marker=:square, label="")
        
        ts = range(0, 1, length=50)
        curve_pts = []
        for t in ts
            pt = (1-t)^2 .* cps[1] .+ 2*(1-t)*t .* cps[2] .+ t^2 .* cps[3]
            push!(curve_pts, pt)
        end
        
        plot!(plt, [p[1] for p in curve_pts], [p[2] for p in curve_pts], 
              color=:green, linewidth=3, label=(i==start_region_idx ? "Path" : ""))
    end
    
    scatter!(plt, [start_pos[1]], [start_pos[2]], color=:green, markersize=8, label="Start")
    scatter!(plt, [goal_pos[1]], [goal_pos[2]], color=:orange, markersize=8, label="Goal")
    
else
    println("No solution found.")
end

display(plt)
savefig(plt, "gcs_solution.png")
println("Plot saved to gcs_solution.png")
