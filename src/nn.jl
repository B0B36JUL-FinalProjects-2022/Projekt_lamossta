using Flux


const EPOCHS = 10

function train()
    model = Chain(
        LSTM(),
        Dense()
    )

    for epoch in 1:EPOCHS

    end    

end

function predict()

end

function evaluate()

end    