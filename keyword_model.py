"""
Linear event-tweet filtration baseline model. Uses keyword frequency 
 
Author: Justin Sech, jsech1@jhu.edu
"""
import os
import argparse
import pandas as pd
import logging
import pickle
import numpy as np

from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

from littlebird import BERTweetTokenizer


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def parse_args():
     parser = argparse.ArgumentParser()
     parser.add_argument("--input-file", required=True, help="CSV with columns <tweet_id>,<tweet>,<label>")
     parser.add_argument("--output-file", required=True, type=str)
     parser.add_argument("--keywords-file", type=str)
     parser.add_argument("--learning-rate", default=0.01, type=float)
     parser.add_argument("--seed", type=int, help="Use this flag to specify a manual seed for train/test split")
     return parser.parse_args()


def countKeywords(text, keywords):
     key_count = dict.fromkeys(keywords,0)
     for word in text.lower().split():
          if word in key_count:
               key_count[word] += 1
     return list(key_count.values())


if __name__ == "__main__":
     args = parse_args()

     # Read in Tweets and labels
     tweets_df = pd.read_csv(args.input_file)
     
     if args.keywords_file:
         with open(args.keywords_file, 'r') as f:
              keywords = [i.strip() for i in f.readlines()]
     else:
        keywords = None

     # Create keyword vector for each tweet
     tokenizer = BERTweetTokenizer()
     X = tweets_df["text"].values
     y = tweets_df["label"].values
     logging.info(f"Loaded data")

     # Initialize vectorizer for ngram features
     vectorizer = CountVectorizer(vocabulary=keywords, tokenizer=tokenizer.tokenize)
     
     # Make pipeline for model
     model = LogisticRegression()
     clf = Pipeline([
        ("ngram", vectorizer),
        ("model", model)
     ])

     scoring = {'accuracy' : make_scorer(accuracy_score),
                'precision' : make_scorer(precision_score),
                'recall' : make_scorer(recall_score),
                'f1_score' : make_scorer(f1_score)}

     # Run 5-fold cross validation
     logging.info(f"Running cross validation")
     skf = StratifiedKFold(n_splits=5, random_state=args.seed, shuffle=True)
     scores = cross_validate(clf, X, y, cv=skf, scoring=scoring)

     # Log results
     logging.info(f"Accuracy:  {np.mean(scores['test_accuracy']):.3}  std: {np.std(scores['test_accuracy']):.3}")
     logging.info(f"Precision: {np.mean(scores['test_precision']):.3}  std: {np.std(scores['test_precision']):.3}")
     logging.info(f"Recall:    {np.mean(scores['test_recall']):.3}  std: {np.std(scores['test_recall']):.3}")
     logging.info(f"F1:        {np.mean(scores['test_f1_score']):.3}  std: {np.std(scores['test_f1_score']):.3}")

     # Train model on entire dataset
     clf.fit(X, y)
     # Save model
     with open(args.output_file, 'wb') as out:
          pickle.dump(clf, out)
     logging.info(f"Model saved to {args.output_file}")
