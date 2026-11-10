import tempfile
import unittest
from pathlib import Path
import nibabel as nib
import numpy as np
from prepare_volumes import extract_slices


class VolumeTests(unittest.TestCase):
    def test_finite_bounded_slices(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fictional.nii.gz'
            data = np.random.default_rng(42).uniform(1, 100, (24, 24, 40)).astype(np.float32)
            nib.save(nib.Nifti1Image(data, np.eye(4)), path)
            slices = extract_slices(path)
            self.assertEqual(len(slices), 8)
            self.assertTrue(all(a.shape == (24, 24) and a.dtype == np.uint8 for _, a in slices))

    def test_constant_image_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'constant.nii.gz'
            nib.save(nib.Nifti1Image(np.ones((16, 16, 16)), np.eye(4)), path)
            with self.assertRaises(ValueError):
                extract_slices(path)
