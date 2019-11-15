"""

Heikal Badrulhisham, 2019 <heikal93@gmail.com>
"""
import csv
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


def get_data():
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
    vectorizer = CountVectorizer()
    count_vectors = vectorizer.fit_transform(texts)
    transformer = TfidfTransformer()

    return transformer.fit_transform(count_vectors)


def run_test(classifier, tfidf, labels):
    pred = classifier.predict(tfidf)

    num_correct = 0
    for i in range(len(pred)):
        print(f'Actual: {pred[i]}\tPrediction: {labels[i]}')

        if pred[i] == labels[i]:
            num_correct += 1

    proportion = f'{num_correct}/{len(pred)}'
    percentage = round((num_correct/len(pred))*100, 2)
    print(f'Accuracy: {proportion}, {percentage}%')


def main():
    """
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
