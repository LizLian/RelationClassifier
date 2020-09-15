# coding: utf-8

import argparse, logging
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn
from mxnet.gluon.data import DataLoader
import gluonnlp as nlp
from load_data import load_dataset, BasicTransform
from model import RelationClassifier
from utils import logging_config
from typing import List


def get_parser():
    parser = argparse.ArgumentParser(description='Train a simple binary relation classifier')
    # training args
    parser.add_argument('--train_file', type=str, help='File containing file representing the input TRAINING data')
    parser.add_argument('--val_file', type=str, help='File containing file representing the input VALIDATION data', default=None)
    parser.add_argument('--test_file', type=str, help='File containing file representing the input TEST data', default=None)
    parser.add_argument('--log_dir', type=str, default='.', help='Output directory for log file')
    parser.add_argument('--out_file', type=str, help='File containing the output predictions')
    parser.add_argument('--debug', action='store_true', help='Run the model on a small dataset for debugging purpose')
    parser.add_argument('--max_len', type=int, default=100, help='Input sequence maximum length')
    parser.add_argument('--context', type=str, help='cpu or gpu')

    # model hyper-parameter args
    parser.add_argument('--epochs', type=int, default=10, help='Upper epoch limit')
    parser.add_argument('--optimizer',type=str, help='Optimizer (adam, sgd, etc.)', default='adam')
    parser.add_argument('--lr',type=float, help='Learning rate', default=0.1)
    parser.add_argument('--batch_size',type=int, help='Training batch size', default=64)
    parser.add_argument('--dropout', type=float, help='Dropout ratio', default=0.5)

    # model embedding args
    parser.add_argument('--embedding_source', type=str, default='freebase-vectors-skipgram1000-en', help='Pre-trained embedding source name')
    parser.add_argument('--fixed_embedding', action='store_true', help='Fix the embedding layer weights')
    parser.add_argument('--random_embedding', action='store_true', help='Use random initialized embedding layer')
    return parser


def classify_test_data(model: RelationClassifier, data_test: BasicTransform, ctx=mx.cpu()) -> List[int]:
    """
    Generate predictions on the test data and returns the predictions in a list
    :param model: trained model
    :param data_test: data loader
    :param ctx: cpu or gpu, default for cpu
    :return: a list of predictions
    """
    preds = []
    for i, x in enumerate(data_test):
        data, inds, label = x
        data = data.as_in_context(ctx)
        inds = inds.as_in_context(ctx)
        score = model(data, inds)
        predictions = mx.nd.argmax(score, axis=1)
        preds.extend(predictions)
    return preds


def train_classifier(vocabulary: nlp.Vocab, transformer: BasicTransform, data: mx.ndarray,
                     ctx=mx.cpu(), debug=True) -> RelationClassifier:
    """
    Main loop for training a classifier
    :param vocabulary: vocabulary of the model
    :param transformer: data structure that holds the training data and their labels
    :param data: raw data to be fed in transformer
    :param data_test: test data to be fed in transformer
    :param ctx: cpu or gpu, default to cpu
    :return: a trained model and a list of predictions for test dataset. returns an empty list if test data is none
    """

    emb_input_dim, emb_output_dim = vocabulary.embedding.idx_to_vec.shape if vocabulary.embedding else (len(vocabulary), 128)

    num_classes = 19 # number of classes derived from training set
    model = RelationClassifier(emb_input_dim, emb_output_dim, num_classes=num_classes)
    differentiable_params = []

    ## initialize model parameters on the context ctx
    model.initialize(mx.init.Xavier(magnitude=2.34), ctx=ctx, force_reinit=True)
    if not args.random_embedding:
        ## set the embedding layer parameters to pre-trained embedding
        model.embedding.weight.set_data(vocabulary.embedding.idx_to_vec)
    elif args.fixed_embedding:
        model.embedding.collect_params().setattr('grad_req', 'null')

    # Do not apply weight decay on LayerNorm and bias terms
    for _, v in model.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0

    # collect parameters for gradient decent
    for p in model.collect_params().values():
        if p.grad_req != 'null':
            differentiable_params.append(p)

    # wd - weight decay (regularization)
    trainer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': args.lr, 'wd': 0.0001})
    distance_loss = DistanceLoss(ctx)

    # use a small dataset if the debug mode is true
    if debug:
        data_train = data[0:1000]
        data_val = data[1000: 1200]
    else:
        data_train = data[:int(len(data)*0.8)]
        data_val = data[int(len(data)*0.8):]
    data_train = gluon.data.SimpleDataset(data_train).transform(transformer)
    data_val = gluon.data.SimpleDataset(data_val).transform(transformer)

    for epoch in range(args.epochs):
        train_dataloader = mx.gluon.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
        val_dataloader = mx.gluon.data.DataLoader(data_val, batch_size=args.batch_size, shuffle=False)

        epoch_loss = 0
        for i, x in enumerate(train_dataloader):
            data1, inds, label = x
            data1 = data1.as_in_context(ctx)
            label = label.as_in_context(ctx)
            inds = inds.as_in_context(ctx)
            with autograd.record():
                score = model(data1, inds)
                # use the distance loss from the acnn paper
                l = distance_loss(score, label)
            l.backward()
            grads = [p.grad(ctx) for p in differentiable_params]
            gluon.utils.clip_global_norm(grads, 1)
            trainer.step(1) ## step = 1 since we took the mean of the loss over the batch
            epoch_loss += l.asscalar()
        logging.info(f"Epoch{epoch} loss = {epoch_loss}")
        # evaluate on the dev dataset
        val_acc = _eval(model, val_dataloader, ctx)
        # evaluate on the training dataset
        train_acc = _eval(model, train_dataloader, ctx)
        logging.info(f"Train Acc = {train_acc}, Validation Acc = {val_acc}")
    # save the best model parameters
    model.save_parameters('base.params')
    return model


def predict(model: RelationClassifier, data_test: mx.ndarray, transformer: BasicTransform, ctx=mx.cpu()) -> mx.ndarray:
    """
    use the trained model for test data inference
    :param model: trained RelationClassifier model
    :param data_test: test dataset
    :param transformer: data structure that holds the test data without their labels
    :return: a list of predictions for the test set
    """
    data_test = gluon.data.SimpleDataset(data_test).transform(transformer)
    test_dataloader = mx.gluon.data.DataLoader(data_test, batch_size=args.batch_size, shuffle=False)
    test_preds = classify_test_data(model, test_dataloader, ctx)
    return test_preds


def _eval(model: RelationClassifier, dataloader: DataLoader, ctx=mx.cpu()) -> float:
    """
    Get predictions on the dataloader items from model
    Return metrics (accuracy, etc.)
    :param model: the trained RelationClassifier model
    :param dataloader: test data in the dataloader structure. there is no labels for test data sets
    :param ctx: cpu or gpu, default to cpu
    :return: the accuracy score
    """

    total_correct, total = 0, 0
    for i, (data, inds, label) in enumerate(dataloader):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        inds = inds.as_in_context(ctx)
        score = model(data, inds)
        predictions = mx.nd.argmax(score, axis=1)
        correctPreds = predictions.astype('int').as_in_context(ctx) == \
                       label.transpose()[0, :].astype('int').as_in_context(ctx)
        total_correct += mx.nd.sum(correctPreds).asscalar()
        total += len(data)
    acc = total_correct / float(total)
    return acc


class DistanceLoss(nn.Block):
    """
    this is the ranking loss function implemented from paper https://www.aclweb.org/anthology/P15-1061.pdf
    Like some other ranking approaches that only update two classes/examples at every training round ,
    this ranking approach can efficiently train the network for tasks which have a very large number of classes.
    """

    def __init__(self, ctx):
        super(DistanceLoss, self).__init__()
        self.ctx = ctx

    def forward(self, score: mx.ndarray, label: mx.ndarray, mplus: float=2.5, mNeg: float=0.5, gamma: int=2) -> mx.ndarray:
        """
        :param score: a list of predicted scores/probabilities over all labels
        :param label: ground truth labels
        :param mplus: a parameter (positive loss) for the distance loss function
        :param mNeg: a parameter (negative loss) for the distance loss function
        :param gamma: a scaling factor that magnifies the difference between the score and the margin. It helps more
        with penalizing the prediction errors
        :return: loss for the batch
        """
        rows = mx.nd.array(list(range(len(score))))
        # ground truth score
        gt_score = score[rows, label.transpose()[0,:]].as_in_context(self.ctx)
        gt_score = mx.nd.log(1 + mx.nd.exp(gamma * (mplus - gt_score))) + mx.nd.log(
            1 + mx.nd.exp(gamma * (-100 + gt_score)))  # positive loss

        # top two scores for each batch, return the score that's different from ground truth
        val, inds = mx.nd.topk(score, axis=1, k=2, ret_typ='both')
        predT = inds[:, 0].astype('int').as_in_context(self.ctx) == label.transpose((1,0)).astype('int').as_in_context(self.ctx)
        predF = inds[:, 0].astype('int').as_in_context(self.ctx) != label.transpose((1,0)).astype('int').as_in_context(self.ctx)
        predT = predT[0, :]
        predF = predF[0, :]

        # negative loss
        part2 = mx.nd.log(1 + mx.nd.exp(gamma * (mNeg + val))) + mx.nd.log(
            1 + mx.nd.exp(gamma * (-100 - val)))
        # positive loss
        part2 = mx.nd.dot(predT.astype('float'), part2[:, 1].astype('float')) + \
                mx.nd.dot(predF.astype('float'), part2[:, 0].astype('float'))

        # include other loss
        loss = mx.nd.sum(gt_score.astype('float')) + part2.astype('float')
        return loss/label.shape[0]


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    # log training process
    logging_config(args.log_dir, 'train', level=logging.INFO)
    # load dataset from an input file
    vocab, dataset, transform = load_dataset(args.train_file, args.max_len)

    # set word embedding
    if args.embedding_source:
        # specify word embedding
        embeddings = nlp.embedding.create('word2vec', source=args.embedding_source)
        vocab.set_embedding(embeddings)
    emb_dim = vocab.embedding.idx_to_vec.shape[1]

    # cpu/gpu
    ctx = mx.gpu() if args.context=='gpu' else mx.cpu()

    # train model
    model = train_classifier(vocab, transform, dataset, ctx, args.debug)

    # create test dataset and inference if the test file is available
    if args.test_file:
        _, test_dataset, _ = load_dataset(args.train_file, args.max_len)
        preds = predict(model, test_dataset, transform, ctx)
        label_map = {transform.label_map[label]: label for label in transform.label_map}
        pred_labels = [pred.asscalar() for pred in preds]
        # write predictions to an output file
        with open(args.out_file, "w") as outf:
            for label in pred_labels:
                outf.write(label_map[label]+"\n")
