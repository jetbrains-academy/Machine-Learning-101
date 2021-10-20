def accuracy(nn, X_test, y_test):
    nn_y = nn.predict(X_test)
    return ((nn_y > 0.5).astype(int) == y_test).sum() / len(y_test)
