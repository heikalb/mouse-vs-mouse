# Is it a mouse or a mouse?

The dataset of this project consists of texts containing the word 'mouse'.
This project is about classifying such texts as either referring to
a mouse the animal or a computer mouse. Texts are converted into feature
vectors, using TF-IDF (Term Frequency times inverse document frequency).
The feature vectors are then used to train a Naive Bayes classifier. 
All of this and the testing of the classifier are implemented in classify.py.
The text data are stored in `datasets/`

Requirement(s): sklearn
Data source: https://www.kaggle.com/werty12121/animal-mouse-vs-computer-mouse-text-dataset#animal.csv