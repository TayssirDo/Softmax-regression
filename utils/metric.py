
def accuracy(y, predictions):
    """ Compute accuracy metric"""
    correct = 0
    for i in range(len(y)):
        if y[i] == predictions[i]:
            correct += 1
    return correct / len(y)
