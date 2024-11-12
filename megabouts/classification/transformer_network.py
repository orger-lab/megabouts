import torch
from torch.utils.data import Dataset
from torch import nn
import numpy as np


class BoutsDataset(Dataset):
    def __init__(self, X, t_sample, sampling_mask, device=None, precision=None):
        device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        precision = (
            precision
            if precision
            else (torch.float64 if self.device.type == "cuda" else torch.float32)
        )

        self.X = torch.from_numpy(np.swapaxes(X, 1, 2)).to(dtype=precision).to(device)
        self.t_sample = (
            torch.from_numpy(t_sample[:, :, np.newaxis]).to(dtype=precision).to(device)
        )
        self.sampling_mask = (
            torch.from_numpy(sampling_mask).to(dtype=torch.bool).to(device)
        )

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :, :], self.t_sample[idx, :], self.sampling_mask[idx, :]


class ContinuousPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(ContinuousPositionalEncoding, self).__init__()
        self.d, self.T = d_model, max_seq_length
        denominators = torch.pow(
            10000, 2 * torch.arange(0, self.d // 2) / self.d
        )  # 10000^(2i/d_model), i is the index of embedding
        self.register_buffer("denominators", denominators)

    def forward(self, x, t):
        pe = torch.zeros((x.shape[0], self.T, self.d), device=x.device)
        pe[:, :, 0::2] = torch.sin(t / self.denominators)  # sin(pos/10000^(2i/d_model))
        pe[:, :, 1::2] = torch.cos(t / self.denominators)  # cos(pos/10000^(2i/d_model))
        return x + pe


class TransAm(nn.Module):
    def __init__(
        self,
        mapping_label_to_sublabel,
        feature_size=64,
        num_layers=3,
        dropout=0.0,
        nhead=8,
    ):
        super(TransAm, self).__init__()
        self.model_type = "Transformer"
        self.input_embedding = nn.Linear(11, feature_size)
        self.pos_encoder = ContinuousPositionalEncoding(
            d_model=feature_size, max_seq_length=140
        )
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_size, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_size))
        self.feature_fill = nn.Parameter(torch.randn(1, 1, 11) - 10)
        self.dense_bout_cat = nn.Linear(feature_size, 18, bias=True)
        self.dense_bout_sign = nn.Linear(feature_size, 2, bias=True)
        self.dense_peak_loc_1 = nn.Linear(feature_size, feature_size, bias=True)
        self.dense_peak_loc_2 = nn.Linear(feature_size, 1, bias=True)
        self.mapping_label_to_sublabel = mapping_label_to_sublabel

    def forward(self, input, t, mask):
        body_angle = input[:, :, 9]
        input[:, :, 9] = torch.cos(body_angle)
        input = torch.cat([input, torch.sin(body_angle[:, :, None])], axis=2)
        mask_feature = torch.isnan(input)
        feature_filler = torch.broadcast_to(self.feature_fill, input.shape).to(
            dtype=input.dtype
        )  # Match dtype with input
        input[mask_feature] = feature_filler[mask_feature]
        output = self.input_embedding(
            input
        )  # linear transformation before positional embedding
        output = self.pos_encoder(output, t)
        output = torch.cat(
            [self.cls_token.expand(output.shape[0], -1, -1), output], dim=1
        )
        mask = torch.cat(
            [
                torch.zeros((output.shape[0], 1), dtype=torch.bool, device=mask.device),
                mask,
            ],
            dim=1,
        )
        output = self.transformer_encoder(
            output, src_key_padding_mask=mask
        )  # ,self.src_mask)
        output_CLS = output[:, 0, :]
        output_bout_cat = self.dense_bout_cat(output_CLS)
        output_bout_sign = self.dense_bout_sign(output_CLS)
        output_t_peak = nn.Sigmoid()(self.dense_peak_loc_1(output_CLS))
        output_t_peak = self.dense_peak_loc_2(output_t_peak)

        # To add to pytorch
        logit_sublabel = output_bout_cat
        logit_label = torch.zeros(
            (output_bout_cat.shape[0], 13),
            dtype=output_bout_cat.dtype,
            device=output_bout_cat.device,
        )
        for i in range(13):
            id = self.mapping_label_to_sublabel[i]
            logit_label[:, i] = torch.log(
                torch.sum(torch.exp(logit_sublabel[:, id]), axis=1)
            )

        # return output_bout_cat,out_firstHB
        return logit_label, logit_sublabel, output_bout_sign, output_t_peak
