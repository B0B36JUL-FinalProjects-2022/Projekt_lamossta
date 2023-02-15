using Embeddings
using DataFrames

export prepare_conll_dataset, prepare_ontonotes_dataset, EMBEDDING_DIM, IOB_DIM, SENTENCE_SIZE, load_dataset_to_matrix, get_embedding   

const SENTENCE_SIZE = 30
const EMBEDDING_DIM = 100
const IOB_DIM = 3
ASSETS_PATHE = chop(dirname(@__FILE__), tail=3) 

@info "Loading embeddings to memory"

my_embtable = load_embeddings(GloVe{:en}, joinpath(ASSETS_PATHE, "data", "glove.6B.100d.txt"))
my_get_word_index = Dict(word=>ii for (ii,word) in enumerate(my_embtable.vocab))
my_emtable_vocab_values = values(my_embtable.vocab)

@info "Embeddings has been loaded to memory"

"""
    load_dataset_to_matrix(path::String, separator::Char)

Function to load data from conll or OntoNotes dataset and return it in the form of x features and y labels.
    
# Arguments
- `path`: Path to dataset file
- `separator`: Separator char of individual data   
"""
function load_dataset_to_matrix(path::String, separator::Char)
    @info "Tokenizing words from dataset"
    
    dataset_file = open(path)
    all_lines_tmp = readlines(dataset_file)
    all_lines = filter((i) -> i != "" && in(lowercase(split(i, separator)[1]), my_emtable_vocab_values), all_lines_tmp)
    total_sentences = div(length(all_lines), SENTENCE_SIZE) + ((length(all_lines) % SENTENCE_SIZE) > 0) 
    x_data = Array{Float32}(undef, EMBEDDING_DIM, length(all_lines)) 
    y_data = Vector{Int32}(undef, length(all_lines))

    @info "Converting tokens to embeddings representation"

    line_counter = 0
    sentence_counter = 1
    for cur_idx in eachindex(all_lines)
        line = all_lines[cur_idx]
        line_counter = line_counter + 1
        splitted_line = split(line, separator)
        cur_word = first(splitted_line)
        entity_label = last(splitted_line)
        iob_label = 0

        if entity_label == "O"
            iob_label = 1
        
        elseif entity_label[1] == 'B'
            iob_label = 2
        
        else
            iob_label = 3    
        end

        y_data[cur_idx] = iob_label
        x_data[:, cur_idx] = get_embedding(cur_word)
        
        if line_counter == SENTENCE_SIZE
            sentence_counter = sentence_counter + 1
            line_counter = 0
        end
    end

    @info "Data successfully converted to embeddings"
    
    return x_data, y_data
end

"""
    get_embedding(word::SubString{String})

Function to get embedding from word.
    
# Arguments
- `word`: Word to get embedding from
"""
function get_embedding(word::SubString{String})
    word = lowercase(word)
    ind = my_get_word_index[word]
    emb = my_embtable.embeddings[:,ind]
    
    return emb
end

"""
    prepare_conll_dataset()

Function to prepare data from conll2003 dataset.
"""
function prepare_conll_dataset()
    conll_train_path = joinpath(ASSETS_PATHE, "data", "conll2003", "train.txt")#"../data/conll2003/train.txt"
    conll_test_path = joinpath(ASSETS_PATHE, "data", "conll2003", "test.txt")#"../data/conll2003/test.txt"
    conll_valid_path = joinpath(ASSETS_PATHE, "data", "conll2003", "valid.txt")#"../data/conll2003/valid.txt"
    
    train_x, train_y = load_dataset_to_matrix(conll_train_path, ' ')
    test_x, test_y = load_dataset_to_matrix(conll_test_path, ' ')
    valid_x, valid_y = load_dataset_to_matrix(conll_valid_path, ' ')

    return train_x, train_y, test_x, test_y, valid_x, valid_y
end

"""
    prepare_ontonotes_dataset()

Function to prepare data from ontonotes5.0 dataset.
"""
function prepare_ontonotes_dataset()
    ontonotes_train_path = joinpath(ASSETS_PATHE, "data", "ontonotes5.0", "train.conll")   #"data/ontonotes5.0/train.conll"
    ontonotes_test_path = joinpath(ASSETS_PATHE, "data", "ontonotes5.0", "test.conll")     #"data/ontonotes5.0/test.conll"
    ontonotes_valid_path = joinpath(ASSETS_PATHE, "data", "ontonotes5.0", "development.conll") #"data/ontonotes5.0/development.conll"

    train_x, train_y = load_dataset_to_matrix(ontonotes_train_path, '\t')
    test_x, test_y = load_dataset_to_matrix(ontonotes_test_path, '\t')
    valid_x, valid_y = load_dataset_to_matrix(ontonotes_valid_path, '\t')

    return train_x, train_y, test_x, test_y, valid_x, valid_y 
end
