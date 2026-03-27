"""
Tests for the distillation mapping module.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optipfair.distillation.mapping import (
    create_layer_map_uniform,
    create_layer_map_last,
    resolve_layer_map,
    MAPPING_UNIFORM,
    MAPPING_LAST,
)


class TestCreateLayerMapUniform(unittest.TestCase):

    def test_same_size(self):
        """When student == teacher, mapping should be 1:1."""
        result = create_layer_map_uniform(n_student=4, n_teacher=4)
        self.assertEqual(result, [0, 1, 2, 3])

    def test_student_smaller_than_teacher(self):
        """Standard case after depth pruning."""
        result = create_layer_map_uniform(n_student=3, n_teacher=6)
        self.assertEqual(result, [0, 2, 4])

    def test_output_length_equals_n_student(self):
        result = create_layer_map_uniform(n_student=5, n_teacher=18)
        self.assertEqual(len(result), 5)

    def test_all_indices_within_teacher_range(self):
        n_teacher = 18
        result = create_layer_map_uniform(n_student=14, n_teacher=n_teacher)
        self.assertTrue(all(0 <= idx < n_teacher for idx in result))


class TestCreateLayerMapLast(unittest.TestCase):

    def test_same_size(self):
        """When student == teacher, mapping should be 1:1."""
        result = create_layer_map_last(n_student=4, n_teacher=4)
        self.assertEqual(result, [0, 1, 2, 3])

    def test_student_smaller_than_teacher(self):
        """Student maps to the last N teacher blocks."""
        result = create_layer_map_last(n_student=3, n_teacher=6)
        self.assertEqual(result, [3, 4, 5])

    def test_output_length_equals_n_student(self):
        result = create_layer_map_last(n_student=5, n_teacher=18)
        self.assertEqual(len(result), 5)

    def test_last_student_maps_to_last_teacher(self):
        n_student, n_teacher = 14, 18
        result = create_layer_map_last(n_student, n_teacher)
        self.assertEqual(result[-1], n_teacher - 1)


class TestResolveLayerMap(unittest.TestCase):

    def setUp(self):
        """Create mock models with config."""

        class MockConfig:
            def __init__(self, num_hidden_layers):
                self.num_hidden_layers = num_hidden_layers

        class MockModel:
            def __init__(self, num_layers):
                self.config = MockConfig(num_layers)

        self.student = MockModel(14)
        self.teacher = MockModel(18)

    def test_uniform_strategy(self):
        result = resolve_layer_map(self.student, self.teacher, MAPPING_UNIFORM)
        expected = create_layer_map_uniform(14, 18)
        self.assertEqual(result, expected)

    def test_last_strategy(self):
        result = resolve_layer_map(self.student, self.teacher, MAPPING_LAST)
        expected = create_layer_map_last(14, 18)
        self.assertEqual(result, expected)

    def test_invalid_strategy_raises(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_layer_map(self.student, self.teacher, "original_indices")
        self.assertIn("Unsupported", str(ctx.exception))

    def test_constants_values(self):
        self.assertEqual(MAPPING_UNIFORM, "uniform")
        self.assertEqual(MAPPING_LAST, "last")


if __name__ == "__main__":
    unittest.main()
