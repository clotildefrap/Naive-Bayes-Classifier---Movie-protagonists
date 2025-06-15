import os
import random


def get_train_test_size(datapath, cl):
#parse and check user input
    files = [os.path.join(datapath, f) for f in os.listdir(datapath) if os.path.isfile(os.path.join(datapath, f))]
    tot = len(files)

    s = '{} has {} docs. '.format(cl, str(tot)) + \
        'How many for training? '
    err = 'Too many. Please enter a lower number.\n'

    firsttime = True
    while True:
        n = int(input(s if firsttime else err + s))
        firsttime = False
        if n < tot:
            break
        
    random.shuffle(files) #check that it does not interfere with the test set choice
    train_files = files[:n]
    test_files = files[n:] # The rest are test files

    return train_files, test_files            


def get_documents(datapath, feature_type, ngram_size):

    train_documents = {}
    test_documents = {}
    
    for gender_folder in  os.listdir(datapath) :
        cl = gender_folder
        folder_path = os.path.join(datapath, gender_folder)

        train_files_for_class, test_files_for_class = get_train_test_size(folder_path, cl)
        
        for f in train_files_for_class:
            if feature_type == "chars":
                gender, doc_length, counts = process_document_ngrams(f, ngram_size, cl)
            elif feature_type == "words":
                gender, doc_length, counts = process_document_words(f, cl)
            train_documents[f] = [gender, doc_length, counts]

        # Process testing documents
        for f in test_files_for_class:
            if feature_type == "chars":
                gender, doc_length, counts = process_document_ngrams(f, ngram_size, cl)
            elif feature_type == "words":
                gender, doc_length, counts = process_document_words(f, cl)
            test_documents[f] = [gender, doc_length, counts]

    return train_documents, test_documents


def extract_vocab(documents):
    vocabulary = []
    for values in documents.values():
        vocabulary += list(values[2].keys())
    #print("First 20 words in the vocabulary:",vocabulary[:20])
    return vocabulary

def process_document_words(filename, cl):
    words={}
    doc_length = 0
    f=open(filename,'r', encoding='utf-8')
    c = 0
    for l in f:
        l=l.rstrip('\n')
        for w in l.split():
            if w in words:
                words[w]+=1
            else:
                words[w]=1
            doc_length += 1
        c+=1
    f.close()
    return cl, doc_length, words

def process_document_ngrams(filename, n, cl):
    ngrams={}
    doc_length = 0
    f=open(filename,'r', encoding='utf-8')
    c = 0
    for l in f:
        l=l.rstrip('\n')
        for i in range(len(l)-n+1):
            ngram=l[i:i+n]
            if ngram in ngrams:
                ngrams[ngram]+=1
            else:
                ngrams[ngram]=1
            doc_length += 1
        c+=1
    f.close()
    return cl, doc_length, ngrams

def top_cond_probs_by_gender(conditional_probabilities,cl,n):
    cps = {}
    for term,probs in conditional_probabilities.items():
        cps[term] = probs[cl]
    
    c = 0
    for term in sorted(cps, key=cps.get, reverse=True):
        if c < n:
            print(c,term,"score:",cps[term])
            c+=1
        else:
            break
        
