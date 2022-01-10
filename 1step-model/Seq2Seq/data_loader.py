import os
import torchtext.legacy.data as data
from torchtext.legacy.data import Field, BucketIterator, Iterator
import json
import logging
import pickle

logger = logging.getLogger(__name__)


def build_dataset(model_type, args, device, use_pretrained_vectors):
    batch_first = False
    if model_type == 'Transformer':
        batch_first = True

    # data 특성에 맞게 tokenizer 변경해주기
    G_tokenizer = lambda x: x.split()
    P_tokenizer = lambda x: x.split()

    SRC = Field(
        init_token='<sos>',
        eos_token='<eos>',
        tokenize=G_tokenizer,
        batch_first=batch_first)
    TRG = Field(
        init_token='<sos>',
        eos_token='<eos>',
        tokenize=P_tokenizer,
        batch_first=batch_first)

    train_data, val_data, test_data = CustomDataset.splits(args.data_dir, SRC, TRG, args)

    if use_pretrained_vectors:
        from torchtext.vocab import Vectors

        vectors = Vectors(name='embedding/glove.txt')
        SRC.build_vocab(train_data, val_data, test_data, vectors=vectors)
    else:
        SRC.build_vocab(train_data, val_data, test_data)

    TRG.build_vocab(train_data, val_data, test_data)

    try:
        os.makedirs(args.save_model_dir)
    except FileExistsError:
        pass

    try:
        with open(os.path.join(args.save_model_dir, 'vocabs.pkl'), 'rb') as f:
            vocabs = pickle.load(f)

    except FileNotFoundError:
        vocabs = {'src': SRC.vocab, 'trg': TRG.vocab}

        with open(os.path.join(args.save_model_dir, 'vocabs.pkl'), 'wb') as f:
            pickle.dump(vocabs, f)

    logger.info(vars(train_data.examples[0]))
    logger.info(vars(val_data.examples[0]))

    train_iter = BucketIterator(train_data, batch_size=args.train_batch_size, train=True, device=device, shuffle=True)
    val_iter = Iterator(val_data, batch_size=args.eval_batch_size, train=False, device=device, shuffle=False,
                        sort=False)
    test_iter = Iterator(test_data, batch_size=args.eval_batch_size, train=False, device=device, shuffle=False,
                         sort=False)

    return (train_iter, val_iter, test_iter), vocabs


class CustomDataset(data.Dataset):
    def __init__(self, elements, SRC, TRG):
        fields = [('src', SRC), ('trg', TRG)]
        examples = []
        for elem in elements:
            s = elem['src']
            t = elem['trg']
            examples.append(data.Example.fromlist([s, t], fields))
        self.sort_key = lambda x: len(x.src)
        super().__init__(examples, fields)

    @classmethod
    def splits(cls, data_dir, SRC, TRG, args):
        # Train Data
        train_file = args.train_data

        with open(os.path.join(data_dir, train_file), 'r', encoding='utf-8') as f:
            train_data = json.load(f)

        # Dev Data
        dev_file = args.dev_data

        with open(os.path.join(data_dir, dev_file), 'r', encoding='utf-8') as f:
            valid_data = json.load(f)

        # Test Data
        test_file = args.test_data
        with open(os.path.join(data_dir, test_file), 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        train_dataset = cls(train_data, SRC, TRG)
        val_dataset = cls(valid_data, SRC, TRG)
        test_dataset = cls(test_data, SRC, TRG)

        logger.info(f"Number of training examples: {len(train_dataset.examples)}")
        logger.info(f"Number of validation examples: {len(val_dataset.examples)}")
        logger.info(f"Number of testing examples: {len(test_dataset.examples)}")

        return (train_dataset, val_dataset, test_dataset)
