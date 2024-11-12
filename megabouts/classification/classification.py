import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .transformer_network import BoutsDataset, TransAm
from ..tracking_data.tracking_data import TrackingConfig
from ..config.segmentation_config import SegmentationConfig


class TailBouts:
    def __init__(self, *, segments, classif_results, tail_array=None, traj_array=None):
        # Segmentation Info:
        self.onset = segments.onset
        self.offset = segments.offset
        self.HB1 = segments.HB1
        # Classif Info:
        self.category = classif_results["cat"]
        self.subcategory = classif_results["subcat"]
        self.sign = classif_results["sign"]
        self.proba = classif_results["proba"]

        # Dataframe:
        self.df = self._create_dataframe()

        # Tail or Traj_array:
        if tail_array is not None:
            self.tail = tail_array
        if traj_array is not None:
            self.traj = traj_array

    def _create_dataframe(self):
        data = np.vstack(
            (
                self.onset,
                self.offset,
                self.HB1,
                self.category,
                self.subcategory,
                self.sign,
                self.proba,
            )
        ).T

        columns = [
            ("location", "onset"),
            ("location", "offset"),
            ("location", "HB1"),
            ("label", "category"),
            ("label", "subcategory"),
            ("label", "sign"),
            ("label", "proba"),
        ]
        columns = pd.MultiIndex.from_tuples(columns)

        return pd.DataFrame(data, index=range(data.shape[0]), columns=columns)


class BoutClassifier:
    def __init__(
        self,
        tracking_cfg: TrackingConfig,
        segmentation_cfg: SegmentationConfig,
        exclude_CS: bool = False,
        device=None,
        precision=None,
    ):
        self.tracking_cfg = tracking_cfg
        self.segmentation_cfg = segmentation_cfg
        self.exclude_CS = exclude_CS
        # Set device and precision dynamically based on arguments
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.precision = (
            precision
            if precision
            else (torch.float64 if self.device.type == "cuda" else torch.float32)
        )
        self.net = self.load_classifier()
        self.input_len = 140

    def load_classifier(self):
        mapping_label_to_sublabel = {
            0: [0],
            1: [1],
            2: [2],
            3: [3],
            4: [4, 5, 6],
            5: [7, 8],
            6: [9],
            7: [10],
            8: [11],
            9: [12],
            10: [13, 14],
            11: [15],
            12: [16, 17],
        }
        net = (
            TransAm(mapping_label_to_sublabel, num_layers=3, nhead=8)
            .to(dtype=self.precision)
            .to(self.device)
        )
        transformer_weights_path = os.path.join(
            os.path.dirname(__file__), "transformer_weights.pt"
        )
        net.load_state_dict(
            torch.load(transformer_weights_path, map_location=torch.device(self.device))
        )
        return net

    def prepare_tensor_input(self, **kwargs):
        X = self.extract_bouts(**kwargs)
        t_sampling, sampling_mask = self.compute_sampling_input(X.shape[0])
        sampling_mask = (
            1 - sampling_mask
        )  # the positions with the value of True will be ignored
        data = BoutsDataset(
            X, t_sampling, sampling_mask, device=self.device, precision=self.precision
        )
        data_loader = DataLoader(data, batch_size=50, shuffle=False)
        return data, data_loader

    def extract_bouts(self, **kwargs):
        if self.tracking_cfg.tracking == "full_tracking":
            return self.extract_bouts_full_tracking(**kwargs)
        elif self.tracking_cfg.tracking == "head_tracking":
            return self.extract_bouts_head_tracking(**kwargs)
        else:
            raise ValueError(
                f"No implemented yet tracking mode: {self.tracking_cfg.tracking}"
            )

    def extract_bouts_full_tracking(self, *, tail_array, traj_array):
        num_samples, T = tail_array.shape[0], tail_array.shape[2]
        X = np.full((num_samples, 10, self.input_len), np.nan)
        X[:, :7, :T] = tail_array[:, :7, :]
        X[:, 7:, :T] = traj_array
        return X

    def extract_bouts_head_tracking(self, *, traj_array):
        num_samples, T = traj_array.shape[0], traj_array.shape[2]
        X = np.full((num_samples, 10, self.input_len), np.nan)
        X[:, 7:, :T] = traj_array
        return X

    def compute_sampling_input(self, num_samples):
        t_sampling = np.zeros(self.input_len)
        t_ = np.linspace(
            0,
            self.segmentation_cfg.bout_duration_ms,
            self.segmentation_cfg.bout_duration,
            endpoint=False,
        )
        t_sampling[: len(t_)] = t_
        t_sampling = np.repeat(t_sampling[None, :], num_samples, axis=0)

        sampling_mask = np.zeros((num_samples, self.input_len))
        sampling_mask[:, : len(t_)] = 1

        return t_sampling, sampling_mask

    def run_classification(self, **kwargs):
        data, data_loader = self.prepare_tensor_input(**kwargs)
        results = []

        with torch.no_grad():
            for inputs in data_loader:
                feature, t, mask = inputs
                results.append(self.net(feature, t, mask))

        classif_results = self.process_results(results, len(data))
        return classif_results

    def process_results(self, results, num_samples):
        bout_cat = np.zeros(num_samples)
        bout_subcat = np.zeros(num_samples)
        bout_sign = np.zeros(num_samples)
        proba = np.zeros(num_samples)
        t_peak_ms = np.zeros(num_samples)

        k = 0
        for result in results:
            batch_size = result[0].shape[0]
            id_batch = np.arange(k, k + batch_size)
            logit_label, logit_sublabel, logit_bout_sign, t_p = [
                x.cpu().detach().numpy() for x in result
            ]
            logit_label, logit_sublabel = self.filter_logit(logit_label, logit_sublabel)
            proba[id_batch] = np.max(
                np.exp(logit_label)
                / np.sum(np.exp(logit_label), axis=1, keepdims=True),
                axis=1,
            )
            bout_cat[id_batch] = logit_label.argmax(1)
            bout_subcat[id_batch] = logit_sublabel.argmax(1)
            bout_sign[id_batch] = logit_bout_sign.argmax(1)
            t_peak_ms[id_batch] = t_p[:, 0]
            k += batch_size

        # Convert first half-beat location from ms to frame unit:
        first_half_beat = np.round(t_peak_ms * self.tracking_cfg.fps / 1000)
        # Convert bout sign from {0,1} to {-1,1}:
        bout_sign = 2 * bout_sign - 1
        bout_sign = bout_sign.astype("int")
        return {
            "cat": bout_cat,
            "subcat": bout_subcat,
            "sign": bout_sign,
            "proba": proba,
            "first_half_beat": first_half_beat,
        }

    def filter_logit(self, logit_label, logit_sublabel):
        if self.exclude_CS:
            CS_cat = np.array([3, 4])
            CS_subcat = np.array([3, 4, 5, 6])
            logit_label[:, CS_cat] = -np.inf
            logit_sublabel[:, CS_subcat] = -np.inf
        return logit_label, logit_sublabel
