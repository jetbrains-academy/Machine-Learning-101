def accuracy(nn, X_test, y_test):
    # Feed the test dataset to the predict method
    nn_y = nn.predict(X_test)
    # Compare the resulting class labels with the real ones and return the fraction
    # of correctly classified objects. Values greater than 0.5 should be rounded to 1,
    # values less than 0.5 should be rounded to 0.
    return # TODO
