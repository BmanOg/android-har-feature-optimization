import unittest

import numpy as np

from run_stage3 import feature_domain, feature_sensor, metric_bundle


class MetricTests(unittest.TestCase):
    def test_perfect_predictions_score_one(self):
        y = np.arange(1, 13)
        metrics = metric_bundle(y, y)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["daily_macro_f1"], 1.0)
        self.assertEqual(metrics["transition_macro_f1"], 1.0)

    def test_overall_macro_f1_is_equal_group_average(self):
        y_true = np.tile(np.arange(1, 13), 3)
        y_pred = y_true.copy()
        y_pred[[1, 8, 19, 30]] = [3, 7, 12, 4]
        metrics = metric_bundle(y_true, y_pred)
        expected = (
            metrics["daily_macro_f1"] + metrics["transition_macro_f1"]
        ) / 2
        self.assertAlmostEqual(metrics["macro_f1"], expected, places=12)

    def test_feature_family_labels(self):
        self.assertEqual(feature_domain("tBodyAcc-Mean-1"), "time")
        self.assertEqual(feature_domain("fBodyGyro-STD-1"), "frequency")
        self.assertEqual(feature_sensor("tBodyAcc-Mean-1"), "accelerometer")
        self.assertEqual(feature_sensor("fBodyGyro-STD-1"), "gyroscope")


if __name__ == "__main__":
    unittest.main()
