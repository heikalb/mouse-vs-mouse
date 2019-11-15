"""
Classify texts containing the word 'mouse' whether they refer to a mouse the
animal or the computer mouse.
Heikal Badrulhisham, 2019 <heikal93@gmail.com>
"""
import csv
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


def get_data():
    """
    Get text data from .csv files. Associate each text with a label ('animal'
    vs. 'computer')
    :return: list of texts, list of labels
    """
    data = []

    for label in ['animal', 'computer']:
        with open(f'datasets/{label}.csv', 'r') as f:
            reader = csv.reader(f)
            data += [(r[0], label) for r in reader if r][1:]

    random.shuffle(data)
    texts = [d[0] for d in data]
    labels = [d[1] for d in data]

    return texts, labels


def get_vectors(texts):
    """
    Derive feature vectors from texts, using Term Frequency times inverse
    document frequency (TFIDF).
    :param texts: list of texts
    :return: list of TFIDF vectors
    """
    vectorizer = CountVectorizer()
    count_vectors = vectorizer.fit_transform(texts)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(count_vectors)

    return tfidf


def run_test(classifier, tfidf, labels):
    """
    Get predictions of classifier, display its accuracy.
    :param classifier: a Naive Bayes classifier
    :param tfidf: list of TFIDF feature vectors
    :param labels: list of correct labels to predict
    """
    # Make predictions
    pred = classifier.predict(tfidf)

    num_correct = 0

    # Display actual and predicted labels
    for i in range(len(pred)):
        print(f'Actual: {pred[i]}\tPrediction: {labels[i]}')

        # Tally correct predictions
        if pred[i] == labels[i]:
            num_correct += 1

    # Display accuracy
    proportion = f'{num_correct}/{len(pred)}'
    percentage = round((num_correct/len(pred))*100, 2)
    print(f'Accuracy: {proportion}, {percentage}%')


def main():
    """
    Get data, derive feature vectors, train a Naive Bayes classifier, test
    said classifier.
    """
    # Get data
    texts, labels = get_data()

    # Turn texts into vectors
    tfidf = get_vectors(texts)

    # Separate training and testing data
    split = int(len(texts)*0.8)
    train_tfidf, test_tfidf = tfidf[:split], tfidf[split:]
    train_labels, test_labels = labels[:split], labels[split:]

    # Train Naive Bayes classifier
    classifier = MultinomialNB().fit(train_tfidf, train_labels)

    # Test classifier
    run_test(classifier, test_tfidf, test_labels)


if __name__ == '__main__':
    main()
    exit(0)
