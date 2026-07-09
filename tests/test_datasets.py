import tempfile
import unittest
from pathlib import Path

import pandas as pd

from image_moderation_poc.datasets import build_dataset_splits


class DatasetTest(unittest.TestCase):
    def test_dataset_builder_excludes_golden_groups(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            source = pd.DataFrame(
                [
                    {
                        "picture_url": "https://x/D_NQ_NP_1-MLA111_012026-F.webp",
                        "infraction_detected": True,
                        "labels_detected": "GRATIS",
                        "ocr_text": "gratis",
                    },
                    {
                        "picture_url": "https://x/D_NQ_NP_2-MLA222_012026-F.webp",
                        "infraction_detected": False,
                        "labels_detected": "",
                        "ocr_text": "producto normal",
                    },
                    {
                        "picture_url": "https://x/D_NQ_NP_3-MLA333_012026-F.webp",
                        "infraction_detected": True,
                        "labels_detected": "Envio inmediato",
                        "ocr_text": "envio inmediato",
                    },
                    {
                        "picture_url": "https://x/D_NQ_NP_4-MLA444_012026-F.webp",
                        "infraction_detected": False,
                        "labels_detected": "",
                        "ocr_text": "",
                    },
                ]
            )
            golden = pd.DataFrame(
                [
                    {
                        "picture_url": "https://x/D_NQ_NP_1-MLA111_012026-F.webp",
                        "leakage_group_id": "MLA111",
                    }
                ]
            )
            source_path = tmp_path / "source.csv"
            golden_path = tmp_path / "golden.csv"
            source.to_csv(source_path, index=False)
            golden.to_csv(golden_path, index=False)

            paths = build_dataset_splits(source_path, golden_path, tmp_path / "out")
            combined = pd.concat(
                [
                    pd.read_csv(paths["train"]),
                    pd.read_csv(paths["validation"]),
                    pd.read_csv(paths["test_internal"]),
                ],
                ignore_index=True,
            )

            self.assertNotIn("MLA111", set(combined["leakage_group_id"]))
            group_splits = combined.groupby("leakage_group_id")["split"].nunique()
            self.assertEqual(group_splits.max(), 1)


if __name__ == "__main__":
    unittest.main()
