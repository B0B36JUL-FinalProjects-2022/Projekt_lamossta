using Embeddings
using DataFrames


const SENTENCE_SIZE = 30
const EMBEDDING_DIM = 100
const IOB_DIM = 3

@info "Loading embeddings to memory"

my_embtable = load_embeddings(GloVe{:en}, "./data/glove.6B.100d.txt") # or load_embeddings(FastText_Text) or ...
my_get_word_index = Dict(word=>ii for (ii,word) in enumerate(my_embtable.vocab))
my_emtable_vocab_values = values(my_embtable.vocab)

@info "Embeddings has been loaded to memory"

function load_dataset_to_matrix(path::String, separator)
    @info "Tokenizing words from dataset"
    
    dataset_file = open(path)
    all_lines_tmp = readlines(dataset_file)
    all_lines = filter((i) -> i != "" && in(lowercase(split(i, separator)[1]), emtable_vocab_values), all_lines_tmp) # && (lowercase(i) in emtable_vocab_values) == true
    total_sentences = div(length(all_lines), SENTENCE_SIZE) + ((length(all_lines) % SENTENCE_SIZE) > 0) 
    x_data = Array{Float64}(undef, total_sentences, SENTENCE_SIZE, EMBEDDING_DIM) #embeddings_arr
    y_data = Array{Int}(undef, total_sentences, SENTENCE_SIZE, IOB_DIM)

    @info "Converting tokens to embeddings representation"

    line_counter = 0
    sentence_counter = 1
    for cur_idx in eachindex(all_lines)
        line = all_lines[cur_idx]
        line_counter = line_counter + 1
        splitted_line = split(line, separator)
        cur_word = first(splitted_line)
        entity_label = last(splitted_line)
        iob_label = zeros(IOB_DIM)

        if entity_label == "O"
            iob_label[1] = 1
        
        elseif entity_label[1] == 'B'
            iob_label[2] = 1
        
        else
            iob_label[3] = 1    
        end

        y_data[sentence_counter, line_counter, :] = iob_label
        x_data[sentence_counter, line_counter, :] = get_embedding(cur_word)
        
        if line_counter == SENTENCE_SIZE
            sentence_counter = sentence_counter + 1
            line_counter = 0
        
        elseif cur_idx == length(all_lines) && (length(all_lines) % SENTENCE_SIZE) > 0
            x_data = pad_last_subarray_x(x_data, line_counter + 1)
            y_data = pad_last_subarray_y(y_data, line_counter + 1)
        end
    end

    @info "Data successfully converted to embeddings"
    
    return x_data, y_data
end

function pad_last_subarray_x(arr::Array, start_index::Int)
    for idx in start_index:SENTENCE_SIZE
        arr[end, idx, :] = zeros(EMBEDDING_DIM)
    end
    
    return arr
end

function pad_last_subarray_y(arr::Array, start_index::Int)
    for idx in start_index:SENTENCE_SIZE
        arr[end, idx, :] = zeros(IOB_DIM)
    end
    
    return arr
end

function get_embedding(word)
    word = lowercase(word)
    ind = my_get_word_index[word]
    emb = my_embtable.embeddings[:,ind]
    return emb
end

ontonotes_data_path = "./data/ontonotes5.0/train.conll"
ontonotes_x, ontonotes_y = load_dataset_to_matrix(ontonotes_data_path, '\t')

conll_data_path = "./data/conll2003/train.txt"
conll_x, conll_y = load_dataset_to_matrix(conll_data_path, ' ')
