# Naive-Bayes-Classifier---Movie-protagonists

# Gender Attribution in Movie Dialogues – Naive Bayes Classifier

This system is designed to classify the gender category (Men, Women, or Children) of movie protagonists based on their dialogue text, using a Naive Bayes Classifier (NBC). The classifier learns patterns in function words and content words, and evaluates performance at word-level and at various n-gram-length settings.

## Requirements

To run this project, you need Python and the docopt-ng package for command-line parsing:

    python3 -m venv myenv
    source myenv/bin/activate
    pip install docopt-ng spacy
    python -m spacy download en_core_web_sm


## Run the code

To train and evaluate the classifier, you can choose between function or content word features and optionally choose word or n-gram tokenization and the length of the n-grams.

    python3 Code/attribution_content.py --words
    python3 Code/attribution_function.py --words

Or alternatively:

    python3 Code/attribution_content.py --chars 3
    python3 Code/attribution_function.py --chars 3
 
The model loops ten times, asking you to chose how many files you want to use for each category.
The output of the code tells you which operations the classifier is currently performing: computing prior probabilities, conditional probabilities, etc. Then, for illustration, it outputs the 10 features with highest conditional probability for the class under consideration (i.e. for each author). It gives you an idea of which words / ngrams are most important for each gender. Finally we obtain a table with the results for the three metric (F1, precision, and recall) as well as the accuracy. The results are the averages calculated over the ten runs of the loop. A csv file with the results will be generated and stored in a "Results" folder along with a confusion matrix.

## Experiment Design

The classifier was tested in two primary ways:

- Balanced train/test split (80/20) across all categories ;

- Imbalanced (pseudo-random) split to simulate real-world imbalance in gender representation in films
