import tempfile
import unittest
from pathlib import Path
import nibabel as nib
import numpy as np
from ixi_age import features, score, intervals, paired_gain


class AgeTests(unittest.TestCase):
    def test_age_metrics(self):
        result = score([20, 40, 60], [22, 38, 60])
        self.assertAlmostEqual(result['mae_years'], 4 / 3)
        self.assertEqual(intervals(np.array([20, 40]), np.array([20, 40])), [0., 0.])
        comparison = paired_gain(np.array([20, 40]), np.array([20, 40]), np.array([30, 30]))
        self.assertEqual(comparison['mae_reduction_years'], 10.)
        self.assertEqual(comparison['paired_bootstrap_95'], [10., 10.])

    def test_fixed_features(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.nii.gz'
            data = np.random.default_rng(4).uniform(0, 10, (20, 24, 28)).astype(np.float32)
            nib.save(nib.Nifti1Image(data, np.eye(4)), path)
            vector = features(path)
            self.assertEqual(vector.shape, (4096,))
            self.assertTrue(np.isfinite(vector).all())
