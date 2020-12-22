"""
Filtration Model

Given raw tweets (CSV in tweet_id,text,label format),
predict which tweets are related to events. 

Authors: Justin Sech, Alexandra DeLucia
"""
# Standard
import os
import argparse
import logging
from types import SimpleNamespace
from typing import Dict, Any, Iterable
import pickle

# Third-party
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold 
from transformers import RobertaConfig, RobertaModel
from transformers import PreTrainedTokenizer
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary

# Custom packages
from littlebird import BERTweetTokenizer as TweetNormalizer
from littlebird import TweetReader
from BERTweet_utils_2 import Batcher, BERTweetWrapper

# Set up logging
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True, help="CSV with columns <tweet_id>,<tweet>,<label>")
    parser.add_argument("--save-model-path",
                        help="Location to save model. should be torch file (.pt). "
                             "Only saves when cross-validation option is not used.")
    parser.add_argument("--results-file", help="Location to results from cross-validation. Should be a pickle file (.pkl)")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument("--cross-validate", action="store_true",help="Indicates if you want to cross validate results or train on entire dataset")
    parser.add_argument("--save-preds",action="store_true",help="Flag to save y_preds, must also indicate --cross-validate")
    parser.add_argument("--BERTweet-model-path",
        default="/home/aadelucia/files/minerva/src/feature_engineering/BERTweet_base_transformers",
        help="Path to BERTweet_base_transformers folder")
    parser.add_argument("--batch-size", default=20, type=int, help="Batch size")
    parser.add_argument("--num-epochs", default=100, type=int)
    parser.add_argument("--learning-rate", default=0.01, type=float)
    parser.add_argument("--seed", default=42, type=int, help="Use this flag to specify a manual seed for train/test split")
    return parser.parse_args()

 
class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(768, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

    def predict(self, x):
        model_out = self.forward(x)
        return (model_out > 0.5).int()


if __name__ == "__main__":
    args = parse_args()

    # Set CPU/GPU device and claim it
    if args.cpu:
        device = "cpu"
        torch.device("cpu")
    else:
        device = "cuda"
        torch.device("cuda")
        torch.ones(1).to("cuda")

    # Initialize debugging if selected
    if args.debug:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

    # Load model and configurations
    try:
        BERTweetWrapper = BERTweetWrapper(args.BERTweet_model_path, device)
    
    except FileNotFoundError as err:
        logging.error(f"Check path exists: {args.BERTweet_model_path}\n{err}")
        sys.exit(1)

    # Read in Tweets
    tweets_df = pd.read_csv(args.input_file)
    # Separate data and labels
    data = tweets_df.text.values
    labels = tweets_df.label.values

    # Get BERTweet feature representations of each tweet prior to training
    logging.info(f"Collecting BERTweet feature representations")
    features = BERTweetWrapper.get_BERTweet_representation(data)
    logging.info(f"Created {len(features)} tweet represenations")


    if args.cross_validate:
        # Use cross-validation
        # Store results for each fold
        results = {}
        skf = StratifiedKFold(n_splits=5, random_state=args.seed, shuffle=True)
        loss_dict = {}
        for fold, (train_index, test_index) in enumerate(skf.split(data, labels)): 
            # Initialize model, loss, and optimizer
            model = LogisticRegression().to(device)
            criterion = torch.nn.MSELoss().to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
            #optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            
            # Initialize Batcher
            batcher = Batcher(X=features[train_index], y=labels[train_index], batch_size=args.batch_size)

            for epoch in range(args.num_epochs):
                # Log progress
                logging.info(f"On epoch {epoch}")
                for batch_iter, (X, y) in enumerate(batcher.batchify()):
                    logging.debug(f"batch_iter: {batch_iter}\tX: {X}\ty: {y}")

                    # Batch descent
                    model.train()
                    optimizer.zero_grad()
                    
                    # Slice all features for current indecies                    
                    y_pred = model(X)
                    y = torch.reshape(torch.FloatTensor(y), y_pred.size()).to(device)
                    loss = criterion(y_pred, y)
                    loss.backward()
                    optimizer.step()

                    if args.debug:
                        # End program after first batch for debugging
                        break

                    
            # After final epoch, test the model
            # Get BERTweet representation for test data
            test_X = data[test_index]
            test_y = labels[test_index]
            test_features = BERTweetWrapper.get_BERTweet_representation(test_X)

            y_pred = model.predict(test_features).cpu()
            acc = accuracy_score(y_pred, test_y)
            f1 = f1_score(y_pred, test_y)
            prec = precision_score(y_pred, test_y)
            rec = recall_score(y_pred, test_y)

            if args.save_preds:
                results_df = tweets_df['label'].iloc[test_index].to_frame().reset_index()
                results_df['predicted'] = y_pred
                results_df.to_csv("y_preds_bert.csv")
            
            logging.info(f"""Results from fold {fold}
            Accuracy: {acc}
            Precision: {prec}
            Recall: {rec}
            F1:  {f1}
            """)

            # Save model and results
            #REMOVED model from dict
            results[fold] = {
                    "accuracy": acc,
                    "f1": f1,
                    "recall": rec,
                    "precision": prec
            }
            
        # Average the folds for the final score
        final_acc, final_f1, final_rec, final_prec = [], [], [], []
        for fold, res in results.items():
            final_acc.append(res["accuracy"])
            final_f1.append(res["f1"])
            final_rec.append(res["recall"])
            final_prec.append(res["precision"])

        results["accuracy"] = {
            "avg": np.average(final_acc),
            "std": np.std(final_acc)
        }
        results["f1"] = {
            "avg": np.average(final_f1),
            "std": np.std(final_f1)
        }
        results["recall"] = {
            "avg": np.average(final_rec),
            "std": np.std(final_rec)
        }
        results["precision"] = {
            "avg": np.average(final_prec),
            "std": np.std(final_prec)
        }
        logging.info(f"""Final results
        Accuracy: {results["accuracy"]}
        Precision: {results["precision"]}
        Recall: {results["recall"]}
        F1:  {results["f1"]}
        """)

        results['batch_size']=args.batch_size
        results['learning_rate']=args.learning_rate
        
        # Save models and results
        with open(args.results_file, 'wb') as f:
            pickle.dump(results, f)
        logging.info(f"Results saved to {args.results_file}")
        quit()

    #####
    # Train final model on entire dataset
    #####
    # Initialize model
    model = LogisticRegression().to(device)
    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    batcher = Batcher(features, labels, batch_size=args.batch_size)

    for epoch in range(args.num_epochs):
        # Log progress
        logging.info(f"On epoch {epoch}")
        for batch_iter, (X, y) in enumerate(batcher.batchify()):
            logging.debug(f"batch_iter: {batch_iter}\tX: {X}\ty: {y}")
            # Get BERTweet representation of tweets
                
            # Batch descent
            model.train()
            optimizer.zero_grad()
            y_pred = model(X)
            y = torch.reshape(torch.FloatTensor(y), y_pred.size()).to(device)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

    # Save model
    torch.save(model, args.save_model_path)
    logging.info(f"Model saved to {args.save_model_path}")
