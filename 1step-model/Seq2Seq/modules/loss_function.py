import torch
import torch.nn as nn


class LabelSmoothingLoss(nn.Module):
    def __init__(self, ignore_index, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        classes = pred.size(-1)
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)  # [(trg len - 1) * batch size, output dim] 크기만큼 0을 채워서 만들어줌
            true_dist.fill_(self.smoothing / (classes - 1))  # self.smoothing / (classes - 1) 값만큼 모든 원소에 채워줌
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            # 정답인 인덱스에 0.9를 채워줌
            # e.g.)
            # target.data.unsqueeze(1) = [[3],
            #                             ...]
            # true_dist = [[0.0002, 0.0002, 0.0002, 0.9, 0.0002 ...], [...], ...]

            true_dist[:, self.ignore_index] = 0
            # [[0.0002, *0.0000, 0.0002,  ..., 0.0002, 0.0002, 0.0002], => 첫 문장의 첫 단어에 대한 라벨 == 정답인 인덱스에만 1인 원핫 벡터인데 스무딩 중
            # [0.0002, *0.0000, 0.0002,  ..., 0.0002, 0.0002, 0.0002], => 첫 문장의 두 번째 단어에 대한 라벨
            # [0.0002, *0.0000, 0.0002,  ..., 0.0002, 0.0002, 0.0002], => 첫 문장의 세 번째 단어에 대한 라벨
            # ...,
            # self.ignore_index(==1) 은 padding index이므로 loss 계산할 필요가 없으므로 모두 0으로 바꾸어준다.

            mask = torch.nonzero(target.data == self.ignore_index, as_tuple=True)[0]
            # target.data==self.ignore_index
            # [False, False, False, False, False, False, False, False, False, False,  => target 문장에서 padding 된 위치에 True 반환
            # False,  True,  True,  True,  True,  True, False, False, False,  True,
            #  True,  True,  True,  True,  True,  True, ... ]
            # True인 위치의 인덱스만 뽑아낼 때, as_tuple=True 이면 tuple of 1-D tensors 반환하므로 [0] 추가

            if mask.dim() > 0:  # padding 한 위치가 있으면
                true_dist.index_fill_(0, mask, 0.0)  # loss 계산하지 않도록 0으로 바꾸어준다.

        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
