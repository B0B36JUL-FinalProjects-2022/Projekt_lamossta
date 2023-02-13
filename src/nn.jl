module Ner_Lstm


#TODO prejmenovat argumenty v evaluate na test...
#TODO zapisovani do souboru loss na validation pri treninku
#TODO pridat pocitani tohoto http://juliaml.github.io/MLMetrics.jl/latest/fractions/f_score/

using Flux
using Flux.Losses
using Flux: onehotbatch, onehot
using Statistics: mean

include("./prepare_ner_data.jl")
using .Ner_Data

export run_on_conll, run_on_ontonotes

const BATCH_SIZE = 32
const EPOCHS = 10
const LSTM_OUTPUT_DIM = 32

function onehot_y(y_data)
    return onehotbatch(y_data, 1:3)
end

function train(x_data, y_data, valid_x, valid_y)
    dataset = Flux.Data.DataLoader((x_data, y_data), batchsize=BATCH_SIZE, shuffle=true)

    model = Chain(
        LSTM(Ner_Data.EMBEDDING_DIM => LSTM_OUTPUT_DIM),
        Dense(LSTM_OUTPUT_DIM => Ner_Data.IOB_DIM),
        softmax
    )

    function loss(x, y)
        b = size(x,2)
        predictions = [model(x[:,i]) for i=1:b]
        value = crossentropy(hcat(predictions...), y)
        Flux.reset!(model)
        
        return value
    end

    evalcb = function()
        b = size(valid_x,2)
        new_predictions_tmp = [model(valid_x[:,i]) for i=1:b]
        new_predictions = hcat(new_predictions_tmp...)
        loss_val = crossentropy(new_predictions, valid_y)
        accuracy = accuracy_onehot(new_predictions, valid_y)

        @info "Accuracy on validation set: $accuracy"
        @info "Loss on validation set: $loss_val"
    end    

    optimizer = ADAM()

    for epoch in 1:EPOCHS
        @info "Epoch #$epoch"
        Flux.train!(loss, Flux.params(model), dataset, optimizer, cb=Flux.throttle(evalcb, 20))
        println("\n")
    end    

    return model
end    

function run_on_conll()
    train_x, train_y, test_x, test_y, valid_x, valid_y = Ner_Data.prepare_conll_dataset()
    model = train(train_x, onehot_y(train_y), valid_x, onehot_y(valid_y))

    evaluate(model, test_x, onehot_y(test_y), "conll2003")
end

function run_on_ontonotes()
    train_x, train_y, test_x, test_y, valid_x, valid_y = Ner_Data.prepare_ontonotes_dataset()
    model = train(train_x, onehot_y(train_y), valid_x, onehot_y(valid_y))

    evaluate(model, test_x, onehot_y(test_y), "OntoNotes5.0")
end

function accuracy_onehot(y_pred, y)
    return mean(Flux.onecold(y_pred) .== Flux.onecold(y))
end

function evaluate(model, train_x, train_y, dataset_name)
    println("\n")
    println("NER classification report on $dataset_name")
    
    _, cols = size(train_x)
    println("Number of words: $cols")
    println("Number of labels: 3")
    println("Labels: I, O, B")

    b = size(train_x, 2)
    new_predictions_tmp = [model(train_x[:, i]) for i=1:b]
    new_predictions = hcat(new_predictions_tmp...)
    loss_value = crossentropy(new_predictions, train_y)
    accuracy = accuracy_onehot(new_predictions, train_y)

    println("Loss on training set: $loss_value")
    println("Accuracy on training set: $accuracy")
end    

run_on_ontonotes()

end
