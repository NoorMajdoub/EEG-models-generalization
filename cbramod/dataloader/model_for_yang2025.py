import torch
import torch.nn as nn

from .cbramod import CBraMod


class Model(nn.Module):
    """
    Yang2025 / WBCIC-SHU binary motor imagery head. 59 channels x 6 patches
    x 200 = 70800-dim flattened backbone output -- same sizing pattern as
    model_for_clinical.py/model_for_karaone.py (flatten + shrink), not
    DREAM's aggressive channel+patch pooling, since 70800 is a much more
    reasonable ratio against Yang2025's larger (~51-subject) training set
    than DREAM's 51400-dim flatten was against its ~120 trials.
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
            nn.Linear(59 * 6 * 200, 6 * 200),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(6 * 200, 200),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(200, 1)
        )

    def forward(self, x):
        bz, ch_num, seq_len, patch_size = x.shape
        feats = self.backbone(x)  # (bz, 59, 6, 200)
        feats = feats.contiguous().view(bz, -1)
        out = self.classifier(feats)
        out = out.contiguous().view(bz)
        return out