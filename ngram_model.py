"""
Linear event-tweet filtration baseline model. Uses token count frequency.
 
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
from sklearn.preprocessing import StandardScaler

from littlebird import BERTweetTokenizer


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def parse_args():
     parser = argparse.ArgumentParser()
     parser.add_argument("--input-file", required=True, help="CSV with columns <tweet_id>,<tweet>,<label>")
     parser.add_argument("--save-model-path", type=str, required=True,
                         help="Location to save model. Should be a pickle file (.pkl)")
     parser.add_argument("--results-file", type=str, required=True,
                         help="Location to results from cross-validation. Should be a pickle file (.pkl)")
     parser.add_argument("--keywords-file", type=str)
     parser.add_argument("--scale-data", action="store_true")
     parser.add_argument("--max-iter", type=int, default=100)
     parser.add_argument("--seed", type=int, help="Use this flag to specify a manual seed for train/test split")
     return parser.parse_args()


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
     pipeline = [("ngram", vectorizer)]
     
     if args.scale_data:
         # Scale data to help convergence
         scaler = StandardScaler(with_mean=False)
         pipeline.append(("scaler", scaler))

     # Model
     model = LogisticRegression(max_iter=args.max_iter)
     pipeline.append(("model", model))
     
     # Make pipeline for model
     clf = Pipeline(pipeline)

     scoring = {'accuracy' : make_scorer(accuracy_score),
                'precision' : make_scorer(precision_score),
                'recall' : make_scorer(recall_score),
                'f1_score' : make_scorer(f1_score)}

     # Run 5-fold cross validation
     logging.info(f"Running cross validation")
     skf = StratifiedKFold(n_splits=5, random_state=args.seed, shuffle=True)
     scores = cross_validate(clf, X, y, cv=skf, scoring=scoring)

     # Save and log results
     results = {
          "accuracy":  {
               "avg": np.mean(scores['test_accuracy']),
               "std": np.std(scores['test_accuracy'])
          },
          "precision": {
               "avg": np.mean(scores['test_precision']),
               "std": np.std(scores['test_precision'])
          },
          "recall": {
               "avg": np.mean(scores['test_recall']),
               "std": np.std(scores['test_recall'])
          },
          "f1": {
               "avg": np.mean(scores['test_f1_score']),
               "std": np.std(scores['test_f1_score'])
          }
     }
     with open(args.results_file, 'wb') as out:
          pickle.dump(results, out)
     results_str = "\n".join([f"{metric}: {d['avg']:.3} ({d['std']:.3})" for metric, d in results.items()])
     logging.info(results_str)

     # Train model on entire dataset
     clf.fit(X, y)
     logging.info(f"Fit data in {clf['model'].n_iter_} iterations")

     # Save model
     with open(args.save_model_path, 'wb') as out:
          pickle.dump(clf, out)
     logging.info(f"Model saved to {args.save_model_path}")
