def exercise_1():
    # TODO: Use GridSearchCV to tune n_neighbors and weights
    # TODO: Train best KNN model and evaluate on test set
    # Goal: >97% accuracy
    pass


def exercise_2():
    # TODO: Write a function that can shift an MNIST image in any direction (left, right, up, or down) by one pixel.
    # TODO: For each image in the training set, create four shifted copies (one per direction) and add them to the training set.
    # TODO: Train your best model on this expanded training set and measure its accuracy on the test set.
    pass


def exercise_3():
    # TODO: Tackle the Titanic dataset.
    # TODO: Download the data from https://homl.info/titanic.tgz and unzip it.
    # TODO: Load train.csv and test.csv using pandas.read_csv().
    # TODO: Train a classifier that predicts the Survived column based on other columns.
    pass


def exercise_4():
    # a. Download examples of spam and ham from Apache SpamAssassin’s public datasets.
    # b. Unzip the datasets and familiarize yourself with the data format.
    # c. Split the data into a training set and a test set.
    # d. Write a data preparation pipeline to convert each email into a feature vector.
    #    Your preparation pipeline should transform an email into a (sparse) vector that indicates the presence or absence of each possible word.
    #    Example: ["Hello", "how", "are", "you"] → "Hello you Hello Hello you" → [1, 0, 0, 1] (binary) or [3, 0, 0, 2] (counts).
    #    Add hyperparameters to control: stripping headers, lowercasing, removing punctuation, replacing URLs/numbers, stemming.
    # e. Try out several classifiers and build a great spam classifier with both high recall and high precision.
    pass


if __name__ == "__main__":
    pass