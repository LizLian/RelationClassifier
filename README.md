# RelationClassifier
Relation classification is a task of identifying the semantic relation holding between two nominal entities in text.

Example: Fizzy [drinks] and meat cause heart disease and [diabetes].

With the annotated entities: entity1 -> drinks; entity2 -> diabetes, the goal is to automatically recognize the input sentence express a cause-effect relation between entity1 and entity2.

There are 19 relation types in the train dataset. They are: Component-Whole, Component-Whole-Inv, Instrument-Agency, Instrument-Agency-Inv, Member-Collection, Member-Collection-Inv, Cause-Effect, Cause-Effect-Inv, Entity-Destination, Entity-Destination-Inv, Content-Container, Content-Container-Inv, Message-Topic, Message-Topic-Inv, Product-Producer, Product-Producer-Inv, Entity-Origin, Entity-Origin-Inv, Other

This project is an implementation of the paper "Relation Classification via Multi-Level Attention CNNs" https://www.aclweb.org/anthology/P16-1123.pdf https://www.aclweb.org/anthology/P15-1061.pdf

The model consists of input word embedding, relative position encodings, multiple CNN layers as the sliding window to recognize bigrams, trigrams etc., a max pooling layer, and  a dense layer at the end.
This paper proposed a novel loss function  L = log(1 + exp(γ(m+ − sθ(x)y+ )) + log(1 + exp(γ(m− + sθ(x)c− ). It is referred to as a ranking method in the paper. Like some other ranking approaches that only update two classes/examples at every training round, this ranking approach can efficiently train the network for tasks which have a very large number of classes. The implementation of this loss function can be found in DistanceLoss class in train.py.

# How to run
Ensure all packages in requirements.txt are installed. This can be done by running:
```
pip install -r requirements.txt
```
To train the model, run:
```
python3 train.py [train_file]
```

Following flags help specify file paths and model hyper-parameters

`--train_file` specifies the train file path

`--test_file` specifies the test_file path

`--epoch` upper epoch limit

`--lr` learning rate

`--batch_size` training batch size

`--dropout` dropout rate

`--embedding-source` Pre-trained embedding source name

`--log-dir` Output directory for log file

`--fixed_embedding` Fix the embedding layer weights

`--random_embedding` Use random initialized embedding layer

`--out_file` File containing the output predictions

`--context` cpu or gpu

`--max_len` Input sequence maximum length

`--debug` Run the model on a small dataset under a debug mode

# Code structure
model.py - model implementation for Relation Classifier

    This model uses relative position encoding to measure the relative distance
    between entities and each word in the input sentence.

    Another feature from the model is CNN layers as sliding windows to
    recognize bigrams, trigrams etc., following a max pooling layer to extract
    the most important features.

    This model also incorporates an input attention layer so that it can compare
    the entities with with each word embedding to calculate similarities.

    The last layer of the model is a dense layer for outputting the probabilities
    across all labels.

    The detailed implementation can be seen in the RelationClassifier class.

train.py - main loop for training the classifier

    Other than serving as the main loop for training the classifier, it also contains the implementation of novel distance loss function. this is the ranking loss function implemented from paper https://www.aclweb.org/anthology/P15-1061.pdf
    Like some other ranking approaches that only update two classes/examples at every training round, this ranking approach can efficiently train the
    network for tasks which have a very large number of classes. The detailed implementation can be seen in class DistanceLoss.
    Inference is currently done through trian.py if a test file is provided. May want to separate the inference by creating a predict.py file later.

load_data.py - read the input file and convert unstructured data into token ids.

    use to preprocess files before training the classifier. You only need to provide a train data file. Dataset will be split into train and val
    datasets.

utils.py - utility file

    adapted from existing code

# References
1. Wang, Linlin, et al. "Relation classification via multi-level attention cnns." Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2016.
