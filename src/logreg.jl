include("prepare_titanic_data.jl")
using Statistics: mean

export logistic_loss_gradient, logistic_loss_gradient_descent, lr_classify, zero_to_negative_labels, negative_to_zero_label, run_lr_on_titanic 


function logistic_loss_gradient(w::Vector{Float64}, x_data::Matrix{Float64}, labels_y::Vector{Int64})
    rows, cols = size(x_data)
    labeled_data =  labels_y
    output_mat = Array{Float64}(undef, rows, cols)

    index = 1
    for cur_row in eachrow(x_data)
        labeled_data = (cur_row .* w) .* labels_y[index]
        first_data = 1 ./ (exp.(labeled_data) .+ 1) 
        second_data = -1 .* (cur_row .* labels_y[index])
        output_mat[index, :] = (first_data .* second_data)
        
        index += 1
    end

    return sum(output_mat, dims=1) .* (1 / rows) 
end

function logistic_loss_gradient_descent(training_x::Matrix{Float64}, training_y::Vector{Int64}; max_iter::Int64=17000, learning_rate::Float64=0.0001)
    rows, cols = size(training_x)
    w = ones(Float64, cols)
    
    for i in 1:max_iter 
        w = w .- (logistic_loss_gradient(w, training_x, training_y) .* learning_rate)'
        w = reshape(w, cols)
    end

    return w
end

function lr_classify(features::Vector{Float64})
    class_labels = Vector{Int64}(undef, length(features))
    
    for i in eachindex(features)
        if features[i] > 0
            class_labels[i] = 1
        else
            class_labels[i] = -1    
        end
    end
    
    return negative_to_zero_label(class_labels)
end

function zero_to_negative_labels(y_labels::Vector{Int64})
    for i in eachindex(y_labels)
        if y_labels[i] == 0
            y_labels[i] = -1
        end
    end
    
    return y_labels
end

function negative_to_zero_label(y_labels::Vector{Int64})
    for i in eachindex(y_labels)
        if y_labels[i] == -1
            y_labels[i] = 0
        end
    end
    
    return y_labels
end

function run_lr_on_titanic(maxiter::Int64, lr::Float64)
    train_x, train_y = get_train_features()
    test_x, test_y = get_test_features()

    w = logistic_loss_gradient_descent(train_x, zero_to_negative_labels(train_y), max_iter=maxiter, learning_rate=lr)

    return (mean(lr_classify(test_x * w) .== test_y))
end
