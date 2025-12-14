def accuracy(nn, X_test, y_test):
    # feed the test dataset to the predict method
    nn_y = nn.predict(X_test)
    # compare the resulting class labels with real ones are return the fraction
    # of correctly classified objects. Everything greater than 0.5 should be considered a 1,
    # everything less - a 0.
    return # TODO
