from torch.functional import block_diag
from torch.utils import data
import utils
import trainer
import model
import dataset
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
import random
import argparse
random.seed(0)


argp = argparse.ArgumentParser()
argp.add_argument('function',
                  help="Whether to pretrain, finetune or evaluate a model",
                  choices=["pretrain", "finetune", "evaluate"])
argp.add_argument('variant',
                  help="Which variant of the model to run ('vanilla' or 'synthesizer')",
                  choices=["vanilla", "synthesizer"])
argp.add_argument('pretrain_corpus_path',
                  help="Path of the corpus to pretrain on", default=None)
argp.add_argument('--reading_params_path',
                  help="If specified, path of the model to load before finetuning/evaluation",
                  default=None)
argp.add_argument('--writing_params_path',
                  help="Path to save the model after pretraining/finetuning", default=None)
argp.add_argument('--finetune_corpus_path',
                  help="Path of the corpus to finetune on", default=None)
argp.add_argument('--eval_corpus_path',
                  help="Path of the corpus to evaluate on", default=None)
argp.add_argument('--outputs_path', default=None)
args = argp.parse_args()


device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

block_size = 128
text = open(args.pretrain_corpus_path).read()
pretrain_dataset = dataset.CharCorruptionDataset(text, block_size)


mconf = model.GPTConfig(pretrain_dataset.vocab_size, pretrain_dataset.block_size,
                        n_layer=4, n_head=8, n_embd=256)


if args.variant == 'vanilla':
    pass  # TODO [part c]: Make some model here
    model = model.GPT(mconf).to(device)
elif args.variant == 'synthesizer':
    pass  # TODO [part g]: Make some other model here
    mconf.synthesizer = True
    model = model.GPT(mconf).to(device)


if args.function == 'pretrain':
    assert args.pretrain_corpus_path is not None
    assert args.writing_params_path is not None
    text = open(args.pretrain_corpus_path).read().strip()
    char_corruption_pretrain_dateset = dataset.CharCorruptionDataset(
        text, block_size)
    tconf = trainer.TrainerConfig(
        max_epochs=650,
        batch_size=128,
        learning_rate=6e-3,
        lr_decay=True,
        warmup_tokens=512*20,
        final_tokens=200*len(pretrain_dataset)*block_size,
        num_workers=4
    )
    tconf.ckpt_path = args.writing_params_path
    trainer = trainer.Trainer(
        model, char_corruption_pretrain_dateset, None, tconf)
    trainer.train()


elif args.function == 'finetune':
    assert args.writing_params_path is not None
    assert args.finetune_corpus_path is not None

    text = open(args.finetune_corpus_path).read()
    finetune_dataset = dataset.NameDataset(pretrain_dataset, text)
    #   Hyperparameters for finetuning WITHOUT a pretrained model:
    tconf = trainer.TrainerConfig(
        max_epochs=75,
        batch_size=256,
        learning_rate=6e-4,
        lr_decay=True,
        warmup_tokens=512*20,
        final_tokens=200*len(pretrain_dataset)*block_size,
        num_workers=4
    )

    # 1. If args.reading_params_path is specified, load these parameters into the model
    if args.reading_params_path is not None:
        model.load_state_dict(torch.load(args.reading_params_path))
        #   Hyperparameters for finetuning WITH a pretrained model:
        tconf = trainer.TrainerConfig(
            max_epochs=10,
            batch_size=256,
            learning_rate=6e-4,
            lr_decay=True,
            warmup_tokens=512*20,
            final_tokens=200*len(pretrain_dataset)*block_size,
            num_workers=4
        )

    #     3. Save the resulting model in args.writing_params_path
    tconf.ckpt_path = args.writing_params_path

    trainer = trainer.Trainer(model, finetune_dataset, None, tconf)
    trainer.train()

elif args.function == 'evaluate':
    assert args.outputs_path is not None
    assert args.reading_params_path is not None
    assert args.eval_corpus_path is not None
    model.load_state_dict(torch.load(args.reading_params_path))
    correct = 0
    total = 0
    with open(args.outputs_path, 'w') as fout:
        predictions = []
        for line in tqdm(open(args.eval_corpus_path)):
            x = line.split('\t')[0]
            x = x + '⁇'
            x = torch.tensor([pretrain_dataset.stoi[s]
                             for s in x], dtype=torch.long)[None, ...].to(device)
            pred = utils.sample(model, x, 32, sample=False)[0]
            completion = ''.join([pretrain_dataset.itos[int(i)] for i in pred])
            pred = completion.split('⁇')[1]
            predictions.append(pred)
            fout.write(pred + '\n')
        total, correct = utils.evaluate_places(
            args.eval_corpus_path, predictions)
    if total > 0:
        print('Correct: {} out of {}: {}%'.format(
            correct, total, correct/total*100))
    else:
        print('Predictions written to {}; no targets provided'
              .format(args.outputs_path))
