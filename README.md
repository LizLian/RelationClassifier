# RelationClassifier
Relation classification is a task of identifying the semantic relation holding between two nominal entities in text.
Example: Fizzy [drinks] and meat cause heart disease and [diabetes].
with the annotated entities: entity1 -> drinks; entity2 -> diabetes, the goal is to automatically recognize the input sentence express a cause-effect relation between entity1 and entity2.

This project is an implementation of the paper "Relation Classification via Multi-Level Attention CNNs" https://www.aclweb.org/anthology/P16-1123.pdf https://www.aclweb.org/anthology/P15-1061.pdf
The model consists of input word embedding, relative position encodings, multiple CNN layers as the sliding window to recognize bigrams, trigrams etc, a max pooling layer, and  a dense layer at the end. This paper proposed a novel loss function  L = log(1 + exp(γ(m+ − sθ(x)y+ )) + log(1 + exp(γ(m− + sθ(x)c− ), which is mentioned as a ranking method in the paper . Like some other ranking approaches that only update two classes/examples at every training round, this ranking approach can efficiently train the network for tasks which have a very large number of classes. The implementation of this loss function can be found in DistanceLoss class in train.py.

# Code structure
model.py - the model implementation for Relation Classifier
train.py - the main loop for training the classifier
load_data.py - reads the input file and convert unstructured data into token ids.
utils.py - a utility file
