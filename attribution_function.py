"""Gender Attribution

    Usage:
    attribution_function.py --words
    attribution_function.py --chars=<n>
    attribution_function.py (-h | --help)
    attribution_function.py --version

    Options:
    -h --help     Show this screen.
    --version     Show version.
    --words
    --chars=<n>  Length of char ngram.

    """

import os
import math
from collections import Counter
import numpy as np
from docopt import docopt
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score

from utils_NBC import (
    process_document_words,
    process_document_ngrams,
    get_documents,
    extract_vocab,
    top_cond_probs_by_gender
)

RESULTS_FUNCTION_DIR = "/home/clotilde/Documents/Cours/S2/Human Language Processing/project/Results"
os.makedirs(RESULTS_FUNCTION_DIR, exist_ok=True)

def plot_and_save_metric(all_metrics, metric_name, classes, RESULTS_FUNCTION_DIR):
    import matplotlib.pyplot as plt

    runs = range(1, len(all_metrics) + 1)
    plt.figure(figsize=(8, 5))

    for i, cls in enumerate(classes):
        scores = [run_metrics[i] for run_metrics in all_metrics]
        plt.plot(runs, scores, marker='o', label=cls)

    plt.title(f"{metric_name} over 10 Runs")
    plt.xlabel("Run")
    plt.ylabel(metric_name)
    plt.ylim(0, 1)
    plt.xticks(runs)
    plt.legend(title="Gender")
    plt.grid(True)
    plt.tight_layout()

    filename = os.path.join(RESULTS_FUNCTION_DIR, f"{metric_name.lower().replace(' ', '_')}_over_runs.png")
    plt.savefig(filename)
    plt.close()
    print(f"📈 {metric_name} plot saved to '{filename}'")


def count_docs(documents):
    return len(documents)


def count_docs_in_class(documents, c):
    count=0
    for values in documents.values():
        if values[0] == c:
            count+=1
    return count

def concatenate_text_of_all_docs_in_class(documents, c):
    words_in_class = Counter()
    for values in documents.values():
        if values[0] == c:
            words_in_class.update(values[2])
    return words_in_class

def train_naive_bayes(classes, documents):
    vocabulary = extract_vocab(documents)

    conditional_probabilities = {}
    for t in vocabulary:
        conditional_probabilities[t] = {}
    priors = {}

    print("\n*** Calculating priors and conditional probabilities ***")

    uniform_prior = 1 / len(classes)
    for c in classes:
        priors[c] = uniform_prior
        print(f"Prior for {c}: {priors[c]:.4f}")
        words_in_class = concatenate_text_of_all_docs_in_class(documents, c)
            
        print("Calculating conditional probabilities for the vocabulary.")

        denominator = sum(words_in_class.values())
        for t in vocabulary:
            if t in words_in_class:
                conditional_probabilities[t][c] = (words_in_class[t] + alpha) / (denominator * (1 + alpha))
            else:
                conditional_probabilities[t][c] = (0 + alpha) / (denominator * (1 + alpha))
        
    return vocabulary, priors, conditional_probabilities


def apply_naive_bayes(filepath, actual_class, feature_type, ngram_size, classes, priors, conditional_probabilities):
        
    # extract counts for this document

    if feature_type == "chars":
        _, _, word_counts = process_document_ngrams(filepath, ngram_size, actual_class)
    elif feature_type == "words":
        _, _, word_counts = process_document_words(filepath, actual_class)


    # compute log-score per class
    scores = {}

    for c in classes:
        scores[c] = math.log(priors[c])
        for t in word_counts:
            if t in conditional_probabilities:
                for i in range(word_counts[t]):
                    scores[c] += math.log(conditional_probabilities[t][c])

    # return best class + full score dict
    return max(scores, key=scores.get), scores


# Evaluatig the model

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# ... (keep all your imports and other code)

def evaluate_model(test_documents, feature_type, ngram_size, classes, priors, conditional_probabilities):
    true_labels = []
    predicted_labels = []

    for filepath, (actual_class, _, _) in test_documents.items():
        predicted_class, score_dict = apply_naive_bayes(filepath, actual_class, feature_type, ngram_size,
                                                        classes, priors, conditional_probabilities)
        
        true_labels.append(actual_class)
        predicted_labels.append(predicted_class)
        print(f"🧾 {os.path.basename(filepath)} → Actual: {actual_class} | Predicted: {predicted_class}")

        # Custom per-class classification report (no micro/weighted avg)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, labels=classes, zero_division=0)
    print("\n📊 Classification Report (per class):\n")
    for i, cls in enumerate(classes):
        print(f"{cls}:")
        print(f"  Precision: {precision[i]:.4f}")
        print(f"  Recall:    {recall[i]:.4f}")
        print(f"  F1-score:  {f1[i]:.4f}")

    # Overall accuracy
    acc = accuracy_score(true_labels, predicted_labels)
    print(f"\nOverall Accuracy: {acc:.4f}")

    # Confusion Matrix plotting and saving (same as before)
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(true_labels, predicted_labels, labels=classes)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.xlabel("Predicted gender")
    plt.ylabel("Actual gender")
    plt.title("Confusion Matrix - Naive Bayes")

    # Inside evaluate_model(), replace saving part with:
    if feature_type == "words":
        filename = f"confusion_function_matrix_{feature_type}.png"
    elif feature_type == "chars":
        filename = f"confusion_function_matrix_{feature_type}_{ngram_size}.png"

    filepath = os.path.join(RESULTS_FUNCTION_DIR, filename)
    plt.savefig(filepath, bbox_inches="tight")
    print(f"📁 Confusion matrix saved to '{filepath}'")




if __name__ == '__main__':
    arguments = docopt(__doc__, version='Gender Attribution 1.1')

    if arguments["--words"]:
        feature_type = "words"
        ngram_size = -1
    if arguments["--chars"]:
        feature_type = "chars"
        ngram_size = int(arguments["--chars"])

    alpha = 0.1
    classes = ["Women", "Men", "Children"]    
    datapath = "Data/preprocessed/function_wd/Gender_based_texts"

    all_precisions = []
    all_recalls = []
    all_f1s = []
    all_accuracies = []

    for run in range(10):
        print(f"\n🚀 Run {run + 1}/10")

        train_documents, test_documents = get_documents(datapath, feature_type, ngram_size)
        vocabulary, priors, conditional_probabilities = train_naive_bayes(classes, train_documents)

        for c in classes:
            print(f"\nBest features for {c}") 
            top_cond_probs_by_gender(conditional_probabilities, c, 10)

        true_labels = []
        predicted_labels = []

        for filepath, (actual_class, _, _) in test_documents.items():
            predicted_class, _ = apply_naive_bayes(filepath, actual_class, feature_type, ngram_size,
                                                   classes, priors, conditional_probabilities)
            true_labels.append(actual_class)
            predicted_labels.append(predicted_class)

        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, labels=classes, zero_division=0)
        acc = accuracy_score(true_labels, predicted_labels)

        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)
        all_accuracies.append(acc)

    print("\n📈 AVERAGED METRICS OVER 10 RUNS\n")

    for i, cls in enumerate(classes):
        print(f"{cls}:")
        print(f"  Precision: {np.mean([p[i] for p in all_precisions]):.4f}")
        print(f"  Recall:    {np.mean([r[i] for r in all_recalls]):.4f}")
        print(f"  F1-score:  {np.mean([f[i] for f in all_f1s]):.4f}")
    print(f"\nOverall Accuracy: {np.mean(all_accuracies):.4f}")

    # ✅ Save averaged metrics to CSV with ngram size in name
    # When saving the CSV summary in main:
    if feature_type == "chars":
        csv_filename = f"evaluation_summary_function_chars_{ngram_size}.csv"
    else:
        csv_filename = "evaluation_summary_function_words.csv"

    csv_filepath = os.path.join(RESULTS_FUNCTION_DIR, csv_filename)

    with open(csv_filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Gender", "Avg Precision", "Avg Recall", "Avg F1-score"])

        for i, cls in enumerate(classes):
            avg_precision = np.mean([p[i] for p in all_precisions])
            avg_recall = np.mean([r[i] for r in all_recalls])
            avg_f1 = np.mean([f[i] for f in all_f1s])
            writer.writerow([cls, f"{avg_precision:.4f}", f"{avg_recall:.4f}", f"{avg_f1:.4f}"])

        mean_precision = np.mean([np.mean(p) for p in all_precisions])
        mean_recall = np.mean([np.mean(r) for r in all_recalls])
        mean_f1 = np.mean([np.mean(f) for f in all_f1s])
        writer.writerow(["Average", f"{mean_precision:.4f}", f"{mean_recall:.4f}", f"{mean_f1:.4f}"])

    print(f"\n📁 Summary written to {csv_filepath}")
