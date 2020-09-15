from mxnet.gluon import nn
import mxnet as mx


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
