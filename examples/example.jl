using MyClassification

#RNN on conll and ontonotes datasets
run_on_conll()
run_on_ontonotes()

#KNN with euclidean distance function
predict(5, "euclidean")

#KNN with manhattan distance function
predict(5, "manhattan")

#KNN with chebyshev distance function
predict(5, "chebyshev")

#Logistic regression
lr_method_acc = run_lr_on_titanic(maxiter=17000, lr=0.0001)
println("Logistic regression with maxiter=17000 and lr=0.0001 => Accuracy: $lr_method_acc") 