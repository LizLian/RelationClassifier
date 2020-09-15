# RelationClassifier
Relation classification is a task of identifying the semantic relation holding between two nominal entities in text.

Example: Fizzy [drinks] and meat cause heart disease and [diabetes].

With the annotated entities: entity1 -> drinks; entity2 -> diabetes, the goal is to automatically recognize the input sentence express a cause-effect relation between entity1 and entity2.

There are 19 relation types in the training set. They are:
`Component-Whole, Component-Whole-Inv, Instrument-Agency, Instrument-Agency-Inv, Member-Collection, Member-Collection-Inv, Cause-Effect, Cause-Effect-Inv, Entity-Destination, Entity-Destination-Inv, Content-Container, Content-Container-Inv, Message-Topic, Message-Topic-Inv, Product-Producer, Product-Producer-Inv, Entity-Origin, Entity-Origin-Inv, Other`

This project is an implementation of the paper "Relation Classification via Multi-Level Attention CNNs"[1] using MXNet.
https://www.aclweb.org/anthology/P16-1123.pdf

Embedding - this model uses pre-trained embedding from gluonnlp APIs. You can choose anything come with MNXet.
Read more here - https://gluon-nlp.mxnet.io/api/modules/embedding.html

The model consists of input word embedding, relative position encodings, multiple CNN layers as the sliding window to recognize bigrams, trigrams etc., a max pooling layer, and  a dense layer at the end.
This paper proposed a novel loss function  L = log(1 + exp(γ(m+ − sθ(x)y+ )) + log(1 + exp(γ(m− + sθ(x)c− ). It is referred to as a ranking method in the paper. Like some other ranking approaches that only update two classes/examples at every training round, this ranking approach can efficiently train the network for tasks which have a very large number of classes. The implementation of this loss function can be found in DistanceLoss class in train.py.

# How to run
Ensure all packages in requirements.txt are installed. This can be done by running:
```
pip install -r requirements.txt
```
To train the model, run:
```
python3 train.py --train_file [train_file]
```

Following flags help specify file paths and model hyper-parameters

`--train_file` specifies the train file path, required

`--test_file` specifies the test_file path, optional, default=None

`--out_file` file containing the output predictions, required when a test file is given

`--epoch` upper epoch limit, optional, default=10

`--lr` learning rate, optional, default=0.1

`--batch_size` training batch size, optional, default=64

`--dropout` dropout rate, optional, default=0.5

`--embedding-source` pre-trained embedding source name, optional, default=freebase-vectors-skipgram1000-en

`--log-dir` output directory for log file, optional, default='.'

`--context` run the model on a cpu or gpu device, optional, default to cpu

`--max_len` input sequence maximum length, optional, default=100

`--debug` run the model on a small dataset under a debug mode, action='store_true', default=False

# Code structure
model.py - model implementation for Relation Classifier

    This model uses relative position encoding to measure the relative distance
    between entities and each word in the input sentence. Relative distances are
    later mapped to all positive distances.

    Another feature from the model is CNN layers as sliding windows to
    recognize bigrams, trigrams etc., following a max pooling layer to extract
    the most important features.

    This model also incorporates an input attention layer so that it can compare
    the entities with with each word embedding to calculate similarities.

    The last layer of the model is a dense layer for outputting the probabilities
    across all labels.

    The detailed implementation can be seen in the RelationClassifier class.

train.py - main loop for training the classifier

    Initializes model, serves as the main loop for training the classifier.
    Predict and evaluate the results.
    Inference is currently done through `train.py` if a test file is provided.
    May want to separate the inference by creating a predict.py file later.

distance_loss.py - contains the implementation of a novel distance loss function.
    
    The distance loss function was propose by dos Santos et al. and enhanced by 
    Wang, LinLin, et al. 
    at every training round, this ranking approach can efficiently train the
    network for tasks which have a very large number of classes. The detailed
    implementation can be seen in class `DistanceLoss`.

load_data.py - reads the input file and convert unstructured data into token IDs.

    use to preprocess files before training the classifier. You only need to
    provide a train data file. Dataset will be split into train and val datasets.

utils.py - utility file

    It's a logging file, adapted from existing code.

# Results
Sentence length 100, learning rate 0.3, epoch 25, dropout 0.5, word2vec embedding, including “other” loss, cross validation
Epoch 1: training accuracy: 54% \
Epoch 5: training accuracy: 78% \
Epoch 10: training accuracy: 87% \
Epoch 15: training accuracy: 94% \
Epoch 20: training accuracy: 97% \
Epoch 25: training accuracy: 99% 

Test accuracy: 76%

# References
1. Wang, Linlin, et al. "Relation classification via multi-level attention cnns." Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2016.
2. dos Santos, Cicero, Bing Xiang, and Bowen Zhou. "Classifying Relations by Ranking with Convolutional Neural Networks." Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). 2015.
