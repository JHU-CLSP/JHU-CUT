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


class BERTweetTokenizer():
    """Tokenizer to use with BERTweet model. Modeled after the Hugging Face tokenizers"""
    def __init__(self, 
        model_path: str,
        bos_token: str="<s>",
        eos_token: str="</s>",
        sep_token: str="</s>",
        cls_token: str="<s>",
        unk_token: str="<unk>",
        mask_token: str="<mask>",
        pad_token: str="<pad>",
        pad_token_id: int=1,
        add_prefix_space: bool=False):

        self.model_path = model_path
        self.bpe = fastBPE(SimpleNamespace(bpe_codes=f"{self.model_path}/bpe.codes"))
        self.vocab = Dictionary()
        self.vocab.add_from_file(f"{self.model_path}/dict.txt")
        self.normalizer = TweetNormalizer()
        self.mask_token = mask_token
        self.add_prefix_space = add_prefix_space
        self.pad_token = pad_token
        self.pad_token_id = pad_token_id
        self.bos_token=bos_token
        self.eos_token=eos_token
        self.unk_token=unk_token

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.vocab)

    def get_vocab(self):
        """Get the vocabulary Dictionary"""
        return self.vocab

    def encode(self, text: str) -> (Iterable[int], Iterable[int]):
        """Encode a single string"""
        # 1. Tokenize with BERTweet normalizer
        text = " ".join(self.normalizer.tokenize(text))
        # 2. Encode with fastBPE
        # Keep list of first subword positions
        subwords = f"{self.bos_token} {self.bpe.encode(text)} {self.eos_token}"
        first_subword_pos = [i for i, w in enumerate(subwords.split()) if "@@" not in w]
        # 3. Encode with vocab dict 
        input_ids = self.vocab.encode_line(
            subwords,
            append_eos=False, 
            add_if_not_exist=False).long()
        return input_ids, first_subword_pos

    def batch_encode(self, text_sequence: Iterable[str]) -> (Iterable[int], Iterable[int], Iterable[int]):
        """Encode a list of strings"""
        # 1. Get encoding for the text
        # Keep the positions of the first subword tokens for later decoding
        all_input_ids, all_position_ids = [], []
        for text in text_sequence:
            inputs, pos = self.encode(text)
            all_input_ids.append(inputs)
            all_position_ids.append(pos)

        # 2. Pad to max length
        padded_input_ids, mask = self._pad_to_max_length(
            all_input_ids,
            padding_value=self.pad_token_id)
        
        # 3. Return
        return padded_input_ids, mask, all_position_ids

    def _pad_to_max_length(self, sequences, padding_value: int):
        """
        Padding solution from https://discuss.pytorch.org/t/how-to-do-padding-based-on-lengths/24442/3
        :param sequences: list of input_ids from batch encoding
        :return: Tensor of size B x T x * where T is the length of the longest sequence
        """
        num = len(sequences)
        max_len = max([s.size(0) for s in sequences])
        out_dims = (num, max_len)
        out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
        mask = sequences[0].data.new(*out_dims).fill_(0)
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            out_tensor[i, :length] = tensor
            mask[i, :length] = 1
        return out_tensor, mask

    def og_tokenizer(self, text):
        """Sanity check"""
        text = " ".join(self.normalizer.tokenize(text))
        subwords = '<s> ' + self.bpe.encode(text) + ' </s>'
        input_ids = self.vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False).long().tolist()
        all_input_ids = torch.tensor([input_ids], dtype=torch.long)
        return all_input_ids

 
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


class Batcher:
    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size
    
    def batchify(self, indices):
        # Limit data to specified indices
        X = self.X[indices]
        y = self.y[indices]

        # Calculate number of batches
        num_batches = int(np.ceil(len(X) / self.batch_size))
        for i in range(num_batches):
            start = i * self.batch_size
            end = start + self.batch_size
            end = min(start + self.batch_size, len(X))
            batch_X = X[start:end]
            batch_y = y[start:end]
            if len(batch_X) < self.batch_size:
                logging.debug(f"Expected batch size of {self.batch_size} but generated batch with {len(batch_X)} item(s) with indices [{start}:{end}].")
            yield batch_X, batch_y


def create_BERTweet_features(BERTweetModel, tokenizer, X, device):
    """Create BERTweet representation of provided input data (tweet text)"""
    # Get BERTweet representation of tweets
    # Use an attention mask to hide the padded tokens
    input_ids, attention_mask, first_subword_positions = tokenizer.batch_encode(X)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    with torch.no_grad():
        features = BERTweetModel(
            input_ids,
            attention_mask=attention_mask,
        )[0]  # Only want the embeddings
    logging.debug(f"{features.size()}\n{features}")

    # Represent Tweets as the average of their subwords (first subword positions only)
    # Using just the first token gives very bad results
    avg_features = []
    for indices, subword_embeddings in zip(first_subword_positions, features):
        logging.debug(f"""
indices: {len(indices)}
embedding {subword_embeddings.size()}
AVG:{subword_embeddings[indices].size()}
{torch.mean(subword_embeddings[indices], 0).size()}""")
        avg_features.append(
            torch.mean(subword_embeddings[indices], 0)
        )
    features = torch.stack(avg_features).to(device)
    logging.debug(f"Features: {features.size()}")
    return features


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
        config = RobertaConfig.from_pretrained(f"{args.BERTweet_model_path}/config.json")
        BERTweetModel = RobertaModel.from_pretrained(
            f"{args.BERTweet_model_path}/model.bin",
            config=config)\
            .to(device)
        tokenizer = BERTweetTokenizer(args.BERTweet_model_path)
    except FileNotFoundError as err:
        logging.error(f"Check path exists: {args.BERTweet_model_path}\n{err}")
        sys.exit(1)

    # Read in Tweets
    tweets_df = pd.read_csv(args.input_file)
    # Separate data and labels
    data = tweets_df.text.values
    labels = tweets_df.label.values

    if args.cross_validate:
        # Use cross-validation
        # Store results for each fold
        results = {}
        skf = StratifiedKFold(n_splits=5, random_state=args.seed, shuffle=True)
        for fold, (train_index, test_index) in enumerate(skf.split(data, labels)): 
            # Initialize model, loss, and optimizer
            model = LogisticRegression().to(device)
            criterion = torch.nn.MSELoss().to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

            # Initialize Batcher
            batcher = Batcher(data, labels, args.batch_size)

            for epoch in range(args.num_epochs):
                # Log progress
                logging.info(f"On epoch {epoch}")
                for batch_iter, (X, y) in enumerate(batcher.batchify(train_index)):
                    logging.debug(f"batch_iter: {batch_iter}\tX: {X}\ty: {y}")
                    # Get BERTweet representation of tweets
                    features = create_BERTweet_features(BERTweetModel, tokenizer, X, device)

                    # Batch descent
                    model.train()
                    optimizer.zero_grad()
                    y_pred = model(features)
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
            features = create_BERTweet_features(BERTweetModel, tokenizer, test_X, device)

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
            results[fold] = {
                    "model": model,
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
    batcher = Batcher(data, labels, args.batch_size)

    for epoch in range(args.num_epochs):
        # Log progress
        logging.info(f"On epoch {epoch}")
        for batch_iter, (X, y) in enumerate(batcher.batchify(tweets_df.index)):
            logging.debug(f"batch_iter: {batch_iter}\tX: {X}\ty: {y}")
            # Get BERTweet representation of tweets
            features = create_BERTweet_features(BERTweetModel, tokenizer, X, device)
                
            # Batch descent
            model.train()
            optimizer.zero_grad()
            y_pred = model(features)
            y = torch.reshape(torch.FloatTensor(y), y_pred.size()).to(device)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

    # Save model
    torch.save(model, args.save_model_path)
    logging.info(f"Model saved to {args.save_model_path}")
