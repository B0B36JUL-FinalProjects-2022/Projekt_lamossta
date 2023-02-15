export euclidean_distance, manhattan_distance, chebyshev_distance

"""
    euclidean_distance(vector_x1::Vector{Float64}, vector_x2::Vector{Float64})

Computes euclidean distance between two vectors.

# Arguments
- `vector_x1`: first feature vector
- `vector_x2`: second feature vector
"""
function euclidean_distance(vector_x1::Vector{Float64}, vector_x2::Vector{Float64})
    if length(vector_x1) == length(vector_x2)
        return sqrt(sum((vector_x1 - vector_x2) .^ 2))
    else
        @error "Vector size must match to compute euclidean distance"    
    end
end

"""
    manhattan_distance(vector_x1::Vector{Float64}, vector_x2::Vector{Float64})

Computes manhattan distance between two vectors.

# Arguments
- `vector_x1`: first feature vector
- `vector_x2`: second feature vector
"""
function manhattan_distance(vector_x1::Vector{Float64}, vector_x2::Vector{Float64})
    if length(vector_x1) == length(vector_x2)
        return sum([abs(cur_el) for cur_el in (vector_x1 - vector_x2)])
    else
        @error "Vector size must match to compute manhattan distance"
    end
end

"""
    chebyshev_distance(vector_x1::Vector{Float64}, vector_x2::Vector{Float64})

Computes chebyshev distance between two vectors.

# Arguments
- `vector_x1`: first feature vector
- `vector_x2`: second feature vector
"""
function chebyshev_distance(vector_x1::Vector{Float64}, vector_x2::Vector{Float64})
    if length(vector_x1) == length(vector_x2)
        return maximum([abs(cur_el) for cur_el in (vector_x1 - vector_x2)]) 
    else
        @error "Vector size must match to compute chebyshev distance"
    end
end
