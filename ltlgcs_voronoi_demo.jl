using Statistics
using Polyhedra
using CDDLib
using Graphs
using Plots
using Random
using LinearAlgebra
using VoronoiDelaunay
using TimerOutputs

# ---- 1. Voronoi-based Convex Partitioning ----
Random.seed!(23)
const xmax = 15.0
const ymax = 10.0
const num_partitions = 50
const label_probabilities = Dict("a"=>0.1, "b"=>0.1, "c"=>0.1, "d"=>0.1, "obs"=>0.3)

points = [[rand()*xmax, rand()*ymax] for _ in 1:num_partitions]
vd = DelaunayTessellation()
for pt in points
    push!(vd, Point2D(pt[1], pt[2]))
end

# Helper: Convert Voronoi cell to Polyhedra polytope (clipped to bounding box)
function cell_to_poly(cell, bbox)
    verts = [Point2D(v[1], v[2]) for v in cell]
    # Clip to bounding box
    verts = [Point2D(clamp(v.x, bbox[1][1], bbox[2][1]), clamp(v.y, bbox[1][2], bbox[2][2])) for v in verts]
    arr = [[v.x, v.y] for v in verts]
    v = vrep(arr)
    return polyhedron(v, CDDLib.Library())
end

bbox = ([0.0, 0.0], [xmax, ymax])
regions = Polyhedra.Polyhedron[]
region_labels = Vector{Vector{String}}()

for i in 1:num_partitions
    cell = get_voronoi_cell(vd, i)
    verts = [[v.x, v.y] for v in cell]
    poly = cell_to_poly(verts, bbox)
    # Assign random labels
    labels = String[]
    for (label, prob) in label_probabilities
        if rand() < prob
            push!(labels, label)
        end
    end
    push!(regions, poly)
    push!(region_labels, labels)
end

# ---- 2. Transition System Construction ----
n = length(regions)
g = Graphs.SimpleDiGraph(n)
for i = 1:n, j = 1:n
    if i != j && !isempty(intersect(regions[i], regions[j]))
        Graphs.add_edge!(g, i, j)
    end
end

# ---- 3. Visualization ----
plt = plot(title="LTL-GCS Voronoi Demo", size=(800,600), legend=false)

# Plot regions, color by label
for (i, region) in enumerate(regions)
    verts = [point for point in Polyhedra.collect(Polyhedra.points(Polyhedra.vrep(region)))]
    xs = [v[1] for v in verts]; ys = [v[2] for v in verts]
    xs = push!(xs, xs[1]); ys = push!(ys, ys[1])
    color = :white
    if "obs" in region_labels[i]
        color = :red
    elseif "a" in region_labels[i]
        color = :green
    elseif "b" in region_labels[i]
        color = :blue
    elseif "c" in region_labels[i]
        color = :yellow
    elseif "d" in region_labels[i]
        color = :magenta
    end
    plot!(xs, ys, color=color, st=:shape, fillalpha=0.4, lw=2)
end

# Plot region centers
centers = [vec(mean(reduce(hcat, [v for v in Polyhedra.collect(Polyhedra.points(Polyhedra.vrep(r)))]), dims=2)) for r in regions]
for c in centers
    scatter!([c[1]], [c[2]], marker=:circle, color=:black)
end

# Plot edges between regions
for e in Graphs.edges(g)
    i, j = Graphs.src(e), Graphs.dst(e)
    x_coords = [centers[i][1], centers[j][1]]
    y_coords = [centers[i][2], centers[j][2]]
    plot!(x_coords, y_coords, lw=0.5, lc=:gray)
end

display(plt)
savefig(plt, "ltlgcs_voronoi_demo.png")

println("Voronoi-based partitioning and random labeling complete.")
println("Number of regions: $n")

# ---- 4. Placeholder for LTL, DFA, Product, and GCS ----
println("LTL to DFA, product automaton, and GCS construction not implemented in this demo.")
# You would need to:
# - Parse LTL spec (no direct Julia package, use Python interop or placeholder)
# - Build DFA, take product with TS, build GCS
# - Implement path planning/optimization (e.g., with JuMP)
# - Visualize path as in Python reference
