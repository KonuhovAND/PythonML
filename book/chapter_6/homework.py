from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
    StackingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def exercise_1():
    # TODO: If you have trained five different models on the exact same training data, and
    # they all achieve 95% precision, is there any chance that you can combine these
    # models to get better results? If so, how? If not, why?
    pass


def exercise_2():
    # TODO: What is the difference between hard and soft voting classifiers?
    pass


def exercise_3():
    # TODO: Is it possible to speed up training of a bagging ensemble by distributing it across
    # multiple servers? What about pasting ensembles, boosting ensembles, random forests, or stacking ensembles?
    pass


def exercise_4():
    # TODO: What is the benefit of out-of-bag evaluation?
    pass


def exercise_5():
    # TODO: What makes extra-trees ensembles more random than regular random forests?
    # How can this extra randomness help? Are extra-trees classifiers slower or faster than regular random forests?
    pass


def exercise_6():
    # TODO: If your AdaBoost ensemble underfits the training data, which hyperparameters should you tweak, and how?
    pass


def exercise_7():
    # TODO: If your gradient boosting ensemble overfits the training set, should you increase or decrease the learning rate?
    pass


def exercise_8():
    # TODO: Load the MNIST dataset (introduced in Chapter 3), and split it into a training set,
    # a validation set, and a test set (e.g., use 50,000 instances for training, 10,000 for validation, and 10,000 for testing).
    # Then train various classifiers, such as a random forest classifier, an extra-trees classifier, and an SVM classifier.
    # Next, try to combine them into an ensemble that outperforms each individual classifier on the validation set,
    # using soft or hard voting. Once you have found one, try it on the test set.
    # How much better does it perform compared to the individual classifiers?
    pass


def exercise_9():
    # TODO: Run the individual classifiers from the previous exercise to make predictions on the validation set,
    # and create a new training set with the resulting predictions: each training instance is a vector
    # containing the set of predictions from all your classifiers for an image, and the target is the image’s class.
    # Train a classifier on this new training set. Congratulations—you have just trained a blender,
    # and together with the classifiers it forms a stacking ensemble!
    # Now evaluate the ensemble on the test set. For each image in the test set, make predictions with all your classifiers,
    # then feed the predictions to the blender to get the ensemble’s predictions.
    # How does it compare to the voting classifier you trained earlier?
    # Now try again using a StackingClassifier instead. Do you get better performance? If so, why?
    pass


if __name__ == "__main__":
    # Вызывай функции по мере выполнения.
    # Если я увижу пустые функции в итоговой сдаче — работа не будет принята.
    pass
