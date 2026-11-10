import unittest
from split_manifest import partition


def fixture():
    return [dict(participant_id=f'synthetic-{label}-{i}',
                 image_path=f'synthetic/{label}/{i}/{j}.png', label=label)
            for label in ('a', 'b') for i in range(10) for j in range(3)]


class PartitionTests(unittest.TestCase):
    def test_disjoint_and_stratified(self):
        rows = fixture()
        result = partition(rows)
        self.assertEqual(len(result), len(rows))
        sets = {s: {r['participant_id'] for r in result if r['split'] == s}
                for s in ('train', 'validation', 'test')}
        self.assertFalse(sets['train'] & sets['test'])
        self.assertFalse(sets['train'] & sets['validation'])
        self.assertFalse(sets['test'] & sets['validation'])
        for s in sets:
            self.assertEqual({r['label'] for r in result if r['split'] == s}, {'a', 'b'})
        self.assertTrue(all('split' not in r for r in rows))

    def test_order_independent(self):
        a = {r['participant_id']: r['split'] for r in partition(fixture())}
        b = {r['participant_id']: r['split'] for r in partition(list(reversed(fixture())))}
        self.assertEqual(a, b)

    def test_reject_bad_data(self):
        for rows in ([], fixture() + [fixture()[0]], [dict(participant_id='x', label='a')]):
            with self.assertRaises(ValueError):
                partition(rows)
        rows = fixture()
        rows[1]['label'] = 'conflict'
        with self.assertRaises(ValueError):
            partition(rows)
        with self.assertRaises(ValueError):
            partition(fixture()[:3])
