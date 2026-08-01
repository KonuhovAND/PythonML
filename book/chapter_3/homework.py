def exercise_1():
    from sklearn.datasets import load_digits
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # TODO: Use GridSearchCV to tune n_neighbors and weights
    # TODO: Train best KNN model and evaluate on test set
    # Goal: >97% accuracy
    data = load_digits(as_frame=False)
    x_data, y_target = data.data, data.target
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_target, test_size=0.2)
    pipline = Pipeline(
        [
            ("s", StandardScaler()),
            ("k", KNeighborsClassifier()),
        ]
    )
    param_grid = {
        "k__n_neighbors": [1, 3, 5, 7, 9, 11],
        "k__weights": ["uniform", "distance"],
        "k__n_jobs": [2, 4, 6],
    }
    searcher = GridSearchCV(pipline, scoring="accuracy", param_grid=param_grid, cv=3)
    searcher.fit(x_train, y_train)
    preds = searcher.predict(x_test)
    print(accuracy_score(y_test, preds))
    print(searcher.best_params_, searcher.best_score_)
    print("=" * 30)
    print("task 2")

    x_train_huge, x_test_huge, y_train_huge, y_test_huge = exercise_2()
    searcher.fit(x_train_huge, y_train_huge)
    preds = searcher.predict(x_test_huge)
    print(accuracy_score(y_test_huge, preds))


def exercise_2():
    # TODO: Write a function that can shift an MNIST image in any direction (left, right, up, or down) by one pixel.
    # TODO: For each image in the training set, create four shifted copies (one per direction) and add them to the training set.
    # TODO: Train your best model on this expanded training set and measure its accuracy on the test set.
    from sklearn.datasets import load_digits
    from scipy.ndimage import shift
    from sklearn.model_selection import train_test_split
    import numpy as np

    Data = load_digits(as_frame=False)
    x, y = Data.images, Data.target
    data_north = shift(x, (0, 1, 0), cval=0)
    data_south = shift(x, (0, -1, 0), cval=0)
    data_east = shift(x, (0, 0, 1), cval=0)
    data_west = shift(x, (0, 0, -1), cval=0)

    x_huge = np.vstack(
        [
            x.reshape(-1, 64),
            data_north.reshape(-1, 64),
            data_south.reshape(-1, 64),
            data_east.reshape(-1, 64),
            data_west.reshape(-1, 64),
        ]
    )
    y_huge = np.hstack([y, y, y, y, y])
    x_data, y_target = x_huge, y_huge
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_target, test_size=0.2)
    return x_train, x_test, y_train, y_test


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
    exercise_1()
