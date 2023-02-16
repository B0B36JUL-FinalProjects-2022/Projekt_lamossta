using MyClassification
using Test
using Flux

@testset "MyClassification.jl" begin
    
    @testset "distance_functions.jl" begin
        distance_vector_x1 = [1.5, 5.47, 7.87]
        distance_vector_x2 = [0, 47.87, 250.874]

        @test euclidean_distance(distance_vector_x1, distance_vector_x2) == 246.6798613912372
        @test euclidean_distance(distance_vector_x1, distance_vector_x2) isa Float64
        
        @test manhattan_distance(distance_vector_x1, distance_vector_x2) == 286.904
        @test manhattan_distance(distance_vector_x1, distance_vector_x2) isa Float64

        @test chebyshev_distance(distance_vector_x1, distance_vector_x2) == 243.004
        @test chebyshev_distance(distance_vector_x1, distance_vector_x2) isa Float64
    end

    @testset "knn.jl" begin
        predictions = [1, 0, 0, 0]
        truth_labels = [1, 0, 1, 1]

        train_x = [1.5 5.47 7.87 7.59; 7.47 2.54 6.47 8.98; 4.74 8.74 2.22 1.47; 4.89 2.0 0.0 0.0]
        train_y = [1, 0, 1, 1]
        test_x = [14.5 17.8 19.74 78.47; 4.7 5.2 7.8 2.4; 1.1 1.2 1.3 4.0; 0.0 0.0 2.1 2.0]
        k = 3

        @test evaluate(predictions, truth_labels) == 0.5
        @test evaluate(predictions, truth_labels) isa Float64

        @test find_nearest_neighbours(k, euclidean_distance, train_x, train_y, test_x) == [1, 1, 1, 1]
        @test find_nearest_neighbours(k, manhattan_distance, train_x, train_y, test_x) isa Vector{Int64}
        @test length(find_nearest_neighbours(k, manhattan_distance, train_x, train_y, test_x)) == length(train_y)
    end

    @testset "nn.jl" begin
        vec_y = Int32.([1, 2, 3])
        onehot_batch = Flux.onehotbatch([1, 2, 3], 1:3)
        y_predictions = Float32.([0.541 0.412 0.147; 0.4780 0.0 0.541; 0.748 0.478 0.895])
        y_true = Flux.onehotbatch([2, 1, 1], 1:3)
        
        @test onehot_y(vec_y) == onehot_batch
        @test length(onehot_y(vec_y)) == (length(vec_y) * 3)
        @test onehot_y(vec_y) isa Flux.OneHotMatrix

        @test accuracy_onehot(y_predictions, y_true) == 0.0
        @test accuracy_onehot(y_predictions, y_true) isa Float64
    end

    @testset "prepare_titanic_data.jl" begin
        missing_val = missing
        nan_val = NaN
        replace_val = "replace_val"
        replace_val_num = 24.5

        @test notdefined_replacement(missing_val, replace_val) == replace_val
        @test notdefined_replacement(nan_val, replace_val_num) == 24.5
        @test notdefined_replacement(replace_val, replace_val_num) == replace_val
    end

    @testset "logreg.jl" begin
        negative_labels = [1,-1,-1,1]
        zero_labels = [1,0,0,1]
        predictions = [2.5, -1, -5.4, 47.78]
        w = ones(Float64, 3)
        x_datas = [4.5 6.2 4.0; 5.8 7.4 2.9]
        y_data_labels = [1, 0]

        @test zero_to_negative_labels(zero_labels) == [1, -1, -1, 1]
        @test zero_to_negative_labels(zero_labels) isa Vector{Int64}

        @test negative_to_zero_label(negative_labels) == [1, 0, 0, 1]
        @test negative_to_zero_label(negative_labels) isa Vector{Int64}

        @test lr_classify(predictions) == [1, 0, 0, 1]
        @test lr_classify(predictions) isa Vector{Int64}

        @test logistic_loss_gradient(w, x_datas, y_data_labels) == [-0.024720620918834653 -0.006278493206054634 -0.03597241992418312]
        @test logistic_loss_gradient(w, x_datas, y_data_labels) isa Matrix{Float64}

        @test logistic_loss_gradient_descent(x_datas, y_data_labels, max_iter=1) == [1.000002472062092, 1.0000006278493205, 1.0000035972419925]
        @test logistic_loss_gradient_descent(x_datas, y_data_labels, max_iter=1) isa Vector{Float64}
    end
end
