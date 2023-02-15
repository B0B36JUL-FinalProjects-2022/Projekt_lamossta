using Statistics: mean
using DataStructures: counter

include("prepare_titanic_data.jl")
include("distance_functions.jl")
    

function find_nearest_neighbours(k::Int64, distance_func::Any, train_x::Matrix{Float64}, train_y::Vector{Int64}, test_x::Matrix{Float64})
    rows, _ = size(test_x)
    predictions = Array{Int64}(undef, rows)

    for i in 1:rows
        cur_vec = test_x[i, :]
        distances = [distance_func(collect(cr), cur_vec) for cr in eachrow(train_x)]
        best_k_idxs = sortperm(distances)[1:k]
        best_k_labels = train_y[best_k_idxs]
        best_label = findmax(counter(best_k_labels))[2]
        predictions[i] = best_label
    end
    
    return predictions
end

function predict(k::Int64=3, distance_fnc::String="euclidean")
    @info "K value: $k"
    
    train_x, train_y = get_train_features()
    test_x, test_y = get_test_features()
    distance_fnc_var = nothing
    
    if distance_fnc == "euclidean"
        distance_fnc_var = euclidean_distance
    elseif distance_fnc == "manhattan"
        distance_fnc_var = manhattan_distance
    else
        distance_fnc_var = chebyshev_distance        
    end
    
    predictions = find_nearest_neighbours(k, distance_fnc_var, train_x, train_y, test_x)
    acc_value = evaluate(predictions, test_y)
    @info "Accuracy: $acc_value"

    return k, acc_value
end    

function evaluate(predictions::Vector{Int64}, truth_labels::Vector{Int64})
    acc_value = mean(predictions .== truth_labels)    
    return acc_value
end
