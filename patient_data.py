from __future__ import annotations
from dataclasses import dataclass, asdict, field, fields
from pathlib import Path
from typing import Dict, List, Any

import h5py
import numpy as np
import torch

@dataclass
class PatientRecord:
    """All relevant outputs for *one* patient."""
    true_label: int = field(
        metadata={"role": "Ground-truth class index"}
    )
    pred_label: int = field(
        metadata={"role": "Predicted class index"}
    )
    latent: np.ndarray = field(
        metadata={"role": "Bag-level representation (embedding_dim,)"}
    )
    attention: np.ndarray = field(
        metadata={"role": "Per-instance attention scores"}
    )
    image_paths: List[str] = field(
        metadata={"role": "Full paths of the images in this bag"}
    )
    prediction_vector: np.ndarray = field(
        metadata={"role": "Raw logits for every class"}
    )
    loss: float = field(
        metadata={"role": "Cross-entropy loss for this patient"}
    )

    @staticmethod
    def from_tensors(
        true_label: int,
        pred_label: int,
        latent: torch.Tensor,
        attention: torch.Tensor,
        image_paths: List[str],
        prediction_vector: torch.Tensor,
        loss: torch.Tensor | float,
    ) -> "PatientRecord":
        """Convenience factory that takes PyTorch tensors directly."""
        return PatientRecord(
            true_label=true_label,
            pred_label=pred_label,
            latent=latent.detach().cpu().numpy(),
            attention=attention.detach().cpu().numpy(),
            image_paths=list(image_paths),
            prediction_vector=prediction_vector.detach().cpu().numpy(),
            loss=float(loss) if isinstance(loss, (int, float)) else float(loss.item()),
        )

    @classmethod
    def schema(cls) -> Dict[str, str]:
        """Return {field_name: role_description} as a plain dict."""
        return {f.name: f.metadata.get("role", "") for f in fields(cls)}

    @classmethod
    def describe(cls, width: int = 14) -> None:
        """Nicely print the schema to stdout."""
        print("\nPatientRecord schema")
        print("-" * (width + 3 + 60))
        for name, role in cls.schema().items():
            print(f"{name:<{width}} : {role}")
        print("-" * (width + 3 + 60))


class DataMatrix:
    """
    Collects `PatientRecord`s and can persist / reload them as *one* `.h5` file.

    Internal layout in HDF5:
    ├─ /patients/<patient_id>   (HDF5 group)
    │   ├─ latent               (dataset, float32, gzip)
    │   ├─ attention            (dataset, float32, gzip)
    │   ├─ prediction_vector    (dataset, float32)
    │   ├─ image_paths          (dataset, vlen-str)
    │   └─ attrs: true_label, pred_label, loss
    └─ /meta
        └─ attrs: n_patients, n_classes
    """

    def __init__(self, n_classes: int | None = None) -> None:
        self.records: Dict[str, PatientRecord] = {}
        self.n_classes = n_classes            # helpful metadata

    # ------------------------------------------------------------------ #
    # Add / access
    # ------------------------------------------------------------------ #
    def add_patient(self, patient_id: str, record: PatientRecord) -> None:
        #if patient_id in self.records:
            #raise KeyError(f"Patient '{patient_id}' already added.")
        self.records[patient_id] = record

    def __getitem__(self, patient_id: str) -> PatientRecord:
        return self.records[patient_id]

    def __len__(self) -> int:
        return len(self.records)
    
    def get_patient_names(self) -> List[str]:
        """Returns a list of patient IDs."""
        return list(self.records.keys())


    def save_hdf5(self, file_path: str | Path, *, compression: str = "gzip") -> None:
        file_path = Path(file_path).with_suffix(".h5")
        with h5py.File(file_path, "w") as h5f:
            # Global metadata
            meta = h5f.create_group("meta")
            meta.attrs["n_patients"] = len(self)
            if self.n_classes is not None:
                meta.attrs["n_classes"] = self.n_classes

            # One HDF5 group per patient
            str_dt = h5py.string_dtype(encoding="utf-8")
            for pid, rec in self.records.items():
                grp = h5f.create_group(f"patients/{pid}")

                # Attributes
                grp.attrs["true_label"] = rec.true_label
                grp.attrs["pred_label"] = rec.pred_label
                grp.attrs["loss"] = rec.loss

                # Dense data
                grp.create_dataset("latent",
                                   data=rec.latent.astype(np.float32),
                                   compression=compression)
                grp.create_dataset("attention",
                                   data=rec.attention.astype(np.float32),
                                   compression=compression)
                grp.create_dataset("prediction_vector",
                                   data=rec.prediction_vector.astype(np.float32))

                # Variable-length strings for image paths
                grp.create_dataset("image_paths",
                                   data=np.asarray(rec.image_paths, dtype=str_dt),
                                   dtype=str_dt)

        print(f"✅ Saved {len(self)} patients to '{file_path}'.")

    @classmethod
    def load_hdf5(cls, file_path: str | Path) -> "DataMatrix":
        data = cls()
        with h5py.File(file_path, "r") as h5f:
            data.n_classes = int(h5f["meta"].attrs.get("n_classes", -1))
            patients_grp = h5f["patients"]

            for pid in patients_grp:
                grp = patients_grp[pid]
                rec = PatientRecord(
                    true_label=int(grp.attrs["true_label"]),
                    pred_label=int(grp.attrs["pred_label"]),
                    latent=grp["latent"][...],
                    attention=grp["attention"][...],
                    image_paths=np.concatenate([p for p in grp["image_paths"][...]]).astype(str).tolist(),
                    prediction_vector=grp["prediction_vector"][...],
                    loss=float(grp.attrs["loss"]),
                )
                data.records[pid] = rec
        print(f"📂 Loaded {len(data)} patients from '{file_path}'.")
        return data