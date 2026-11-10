import unittest
from train import validate, participant_metrics, bootstrap_intervals


def rows():
    return [dict(participant_id=f'{s}-{label}', image_path=f'{s}-{label}.png', label=label, split=s)
            for s in ('train', 'validation', 'test') for label in ('0', '1')]


class EvaluationTests(unittest.TestCase):
    def test_participant_leakage_rejected(self):
        manifest = rows()
        validate(manifest)
        manifest[-1]['participant_id'] = manifest[0]['participant_id']
        with self.assertRaises(ValueError):
            validate(manifest)

    def test_duplicate_paths_rejected(self):
        manifest = rows()
        manifest[-1]['image_path'] = manifest[0]['image_path']
        with self.assertRaises(ValueError):
            validate(manifest)

    def test_missing_class_rejected(self):
        with self.assertRaises(ValueError):
            validate(rows()[:-1])

    def test_perfect_predictions(self):
        metrics = participant_metrics([0, 0, 1, 1], [.1, .2, .8, .9])
        self.assertTrue(all(value == 1 for value in metrics.values()))
        ci = bootstrap_intervals([0, 0, 1, 1], [.1, .2, .8, .9])
        self.assertEqual(ci['intervals_95']['auroc'], [1., 1.])

    def test_majority_baseline(self):
        metrics = participant_metrics([0, 0, 0, 1], [0, 0, 0, 0])
        self.assertEqual(metrics['balanced_accuracy'], .5)
        self.assertEqual(metrics['auroc'], .5)
