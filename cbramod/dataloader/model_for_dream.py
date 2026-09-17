import torch
import torch.nn as nn

from .cbramod import CBraMod


class Model(nn.Module):
    """
    DREAM only has ~120 training trials. The first version of this head
    pooled over time/patches only, leaving a Linear(257*200=51400 -> 2000)
    first layer -- ~103M parameters for ~120 training examples, which
    overfit catastrophically (train loss -> ~0 for both pretrained and
    random) and left "best checkpoint" selection on the noisy 42-sample
    val set essentially picking a coin flip that then collapsed to a
    constant single-class prediction on test.

    Fix: pool over BOTH the channel dimension and the time/patch dimension
    (down to a single 200-dim vector per trial), then a small MLP with
    dropout. This only changes our own added head, not the pretrained
    backbone, and cuts the head to a few thousand parameters -- a much
    more reasonable ratio for a ~120-example dataset.
    """
    def __init__(self, param):
        super(Model, self).__init__()
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8
        )
        if param.use_pretrained_weights:
            map_location = torch.device(f'cuda:{param.cuda}')
            self.backbone.load_state_dict(torch.load(param.foundation_dir, map_location=map_location))
        self.backbone.proj_out = nn.Sequential()
        self.classifier = nn.Sequential(
            nn.Linear(200, 64),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        bz, ch_num, seq_len, patch_size = x.shape
        feats = self.backbone(x)  # (bz, ch_num, seq_len, 200)
        feats = feats.mean(dim=2)  # average over the 20 one-second patches -> (bz, ch_num, 200)
        feats = feats.mean(dim=1)  # average over channels -> (bz, 200)
        out = self.classifier(feats)
        out = out.contiguous().view(bz)
        return out