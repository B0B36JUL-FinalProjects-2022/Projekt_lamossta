# Final Julia project

The project is divided into two parts:
1. Named Entity Recognition using LSTM performed on conll2003 and OntoNotes5.0 datasets
2. Titanic survival classification using KNN and Logistic regression

# Project Installation
Package is not officially registered but you can download it using following command:
```julia
(@v1.8) pkg> add https://github.com/B0B36JUL-FinalProjects-2022/Projekt_lamossta
```

# NER structure
- [`src/nn.jl`](src/nn.jl) => implemented classification LSTM net
- [`src/prepare_ner_data.jl`](src/prepare_ner_data.jl) => dataset loading, embedding conversion and feature vector construction

# Titanic structure
- [`src/knn.jl`](src/knn.jl) => implementation of KNN classification algorithm
- [`src/logreg.jl`](src/logreg.jl) => implementation of Logistic regression
- [`src/prepare_titanic_data.jl`](src/prepare_titanic_data.jl) => CSV loading, missing data augmentation and feature vector construction

# Examples and tests
- [`examples/example.jl`](examples/example.jl) => example classification using all implemented algorithms mentioned above
- [`examples/ner_ntb.ipynb`](examples/ner_ntb.ipynb) => notebook with showcase of NER classification
- [`examples/titanic_ntb.ipynb`](examples/titanic_ntb.ipynb) => notebook with showcase of Titanic classification
- [`tests/runtests.jl`](tests/runtests.jl) => Unit tests for various functions    