import torch
import torch.nn as nn


class CoxLoss(nn.Module):
    def __init__(self, do_sigmoid=False):
        super(CoxLoss, self).__init__()
        self.do_sigmoid = do_sigmoid

    def forward(self, risk_scores, time, event):
        '''
            risk_scores: model output (higher = riskier)
            time: recurrence time
            event: 1 = BCR, 0 = censored
        '''

        if self.do_sigmoid:
            risk_scores = torch.sigmoid(risk_scores)

        # sort by descending time
        order = torch.argsort(time, descending=True)
        risk = risk_scores[order]
        event = event[order]

        # cumulative hazard
        log_cum_hazard = torch.log(torch.cumsum(torch.exp(risk), dim=0))

        # loss = -torch.mean((risk - log_cum_hazard) * event)

        loss = -torch.sum(event * (risk - log_cum_hazard)) / (event.sum() + 0.00001)

        return loss


