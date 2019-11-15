"""

Heikal Badrulhisham, 2019 <heikal93@gmail.com>
"""
import csv
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


def main():
    """
    """
    # Get data
    with open('datasets/animal.csv', 'r') as f:
        reader = csv.reader(f)
        animal_data = [(r[0], 'animal') for r in reader if r][1:]

    with open('datasets/computer.csv', 'r') as f:
        reader = csv.reader(f)
        computer_data = [(r[0], 'computer') for r in reader if r][1:]

    data = animal_data + computer_data
    random.shuffle(data)
    train_data = data[:-1000]

    # Get features
    vectorizer = CountVectorizer()
    count_vectors = vectorizer.fit_transform([d[0] for d in train_data])
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(count_vectors)

    # Train Naive Bayes classifier
    classifier = MultinomialNB().fit()
    pred = classifier.predict()

    for p in pred:
        print(p)

    return


if __name__ == '__main__':
    main()
    exit(0)
