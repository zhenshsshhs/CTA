import torch
from torch import nn


# 下游任务模型
class CTASCHModel(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.linear1 = nn.Linear(768, 80)
        self.pretrained = pretrained

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = self.pretrained(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)

        out = self.linear1(out.last_hidden_state[:, 0])
        out = out.softmax(dim=1)
        return out


