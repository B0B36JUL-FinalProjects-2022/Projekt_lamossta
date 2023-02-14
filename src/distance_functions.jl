export euclidean_distance, manhattan_distance, chebyshev_distance

function euclidean_distance(vector_x1::Vector{Float64}, vector_x2::Vector{Float64})
    if length(vector_x1) == length(vector_x2)
        return sqrt(sum((vector_x1 - vector_x2) .^ 2))
    else
        @error "Vector size must match to compute euclidean distance"    
    end
end

function manhattan_distance(vector_x1::Vector{Float64}, vector_x2::Vector{Float64})
    if length(vector_x1) == length(vector_x2)
        return sum([abs(cur_el) for cur_el in (vector_x1 - vector_x2)])
    else
        @error "Vector size must match to compute manhattan distance"
    end
end

function chebyshev_distance(vector_x1::Vector{Float64}, vector_x2::Vector{Float64})
    if length(vector_x1) == length(vector_x2)
        return maximum([abs(cur_el) for cur_el in (vector_x1 - vector_x2)]) 
    else
        @error "Vector size must match to compute chebyshev distance"
    end
end
