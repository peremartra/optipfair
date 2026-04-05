"""
Tests for the CLI interface of OptiPFair.
"""

import unittest
import sys
import os
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optipfair.cli.commands import cli, prune, analyze


class TestCLI(unittest.TestCase):
    """Test cases for the CLI interface."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_cli_help(self):
        """Test help message for the main CLI."""
        result = self.runner.invoke(cli, ['--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('OptiPFair:', result.output)
    
    @patch('optipfair.cli.commands.AutoModelForCausalLM')
    @patch('optipfair.cli.commands.AutoTokenizer')
    @patch('optipfair.cli.commands.prune_model')
    def test_prune_command(self, mock_prune_model, mock_tokenizer, mock_model_class):
        """Test the prune command with basic options."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        pruned_model = MagicMock()
        stats = {
            "original_parameters": 1_000_000,
            "pruned_parameters": 800_000,
            "reduction": 200_000,
            "percentage_reduction": 20.0,
            "expansion_rate": 320.0
        }
        mock_prune_model.return_value = (pruned_model, stats)
        
        # Run command
        with patch('os.makedirs'):
            result = self.runner.invoke(prune, [
                '--model-path', 'test-model',
                '--pruning-type', 'MLP_GLU',
                '--method', 'MAW',
                '--pruning-percentage', '20',
                '--output-path', './test-output',
                '--device', 'cpu',
                '--quiet'
            ])
        
        # Check that the command executed successfully
        self.assertEqual(result.exit_code, 0, msg=f"Command failed with: {result.output}")
        
        # Verify mocks were called as expected
        mock_model_class.from_pretrained.assert_called_once()
        mock_tokenizer.from_pretrained.assert_called_once_with('test-model')
        mock_prune_model.assert_called_once()
        
        # Verify model was saved
        pruned_model.save_pretrained.assert_called_once_with('./test-output')
        mock_tokenizer_instance.save_pretrained.assert_called_once_with('./test-output')

    @patch('optipfair.cli.commands.AutoModelForCausalLM')
    @patch('optipfair.cli.commands.AutoTokenizer')
    @patch('optipfair.cli.commands.prune_model')
    def test_prune_command_expansion_divisor_passthrough(self, mock_prune_model, mock_tokenizer, mock_model_class):
        """Test that --expansion-divisor is passed to prune_model for MLP_GLU."""
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model

        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        pruned_model = MagicMock()
        stats = {
            "original_parameters": 1_000_000,
            "pruned_parameters": 900_000,
            "reduction": 100_000,
            "percentage_reduction": 10.0,
            "expansion_rate": 360.0
        }
        mock_prune_model.return_value = (pruned_model, stats)

        with patch('os.makedirs'):
            result = self.runner.invoke(prune, [
                '--model-path', 'test-model',
                '--pruning-type', 'MLP_GLU',
                '--method', 'MAW',
                '--pruning-percentage', '10',
                '--expansion-divisor', '128',
                '--output-path', './test-output',
                '--device', 'cpu',
                '--quiet'
            ])

        self.assertEqual(result.exit_code, 0, msg=f"Command failed with: {result.output}")
        self.assertEqual(mock_prune_model.call_args.kwargs['expansion_divisor'], 128)

    @patch('optipfair.cli.commands.AutoModelForCausalLM')
    @patch('optipfair.cli.commands.AutoTokenizer')
    @patch('optipfair.cli.commands.prune_model')
    def test_prune_command_no_effective_pruning_returns_usage_error(self, mock_prune_model, mock_tokenizer, mock_model_class):
        """Test ValueError from prune_model is shown as UsageError in CLI."""
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model

        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_prune_model.side_effect = ValueError("No effective pruning for layer 0")

        result = self.runner.invoke(prune, [
            '--model-path', 'test-model',
            '--pruning-type', 'MLP_GLU',
            '--method', 'MAW',
            '--pruning-percentage', '0.01',
            '--expansion-divisor', '256',
            '--output-path', './test-output',
            '--device', 'cpu',
            '--quiet'
        ])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("No effective pruning", result.output)
    
    def test_prune_command_mutually_exclusive_params(self):
        """Test error when providing both pruning-percentage and expansion-rate."""
        result = self.runner.invoke(prune, [
            '--model-path', 'test-model',
            '--pruning-percentage', '20',
            '--expansion-rate', '200',
            '--output-path', './test-output'
        ])
        
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("mutually exclusive", result.output)

    def test_prune_command_expansion_divisor_invalid_for_depth(self):
        """Test that expansion divisor cannot be used with DEPTH pruning."""
        result = self.runner.invoke(prune, [
            '--model-path', 'test-model',
            '--pruning-type', 'DEPTH',
            '--num-layers-to-remove', '2',
            '--expansion-divisor', '128',
            '--output-path', './test-output'
        ])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("only valid with --pruning-type MLP_GLU", result.output)
    
    @patch('optipfair.cli.commands.AutoModelForCausalLM')
    def test_analyze_command(self, mock_model_class):
        """Test the analyze command."""
        # Setup mocks
        mock_model = MagicMock()
        layer = MagicMock()
        mlp = MagicMock()
        
        # Mock the layer and MLP structure
        mlp.gate_proj.in_features = 768
        mlp.gate_proj.out_features = 3072
        mlp.down_proj.out_features = 768
        layer.mlp = mlp
        
        # Mock the named_parameters method to return a list of parameter names and shapes
        layer.named_parameters.return_value = [
            ('self_attn.q_proj.weight', MagicMock(numel=lambda: 589824)),
            ('self_attn.k_proj.weight', MagicMock(numel=lambda: 589824)),
            ('self_attn.v_proj.weight', MagicMock(numel=lambda: 589824)),
            ('self_attn.o_proj.weight', MagicMock(numel=lambda: 589824)),
            ('mlp.gate_proj.weight', MagicMock(numel=lambda: 2359296)),
            ('mlp.up_proj.weight', MagicMock(numel=lambda: 2359296)),
            ('mlp.down_proj.weight', MagicMock(numel=lambda: 2359296)),
            ('input_layernorm.weight', MagicMock(numel=lambda: 768)),
            ('post_attention_layernorm.weight', MagicMock(numel=lambda: 768))
        ]
        
        # Set up the model structure
        mock_model.named_parameters = MagicMock(return_value=[])
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Patch get_model_layers to return our mock layers
        with patch('optipfair.pruning.utils.get_model_layers', return_value=[layer]):
            # Patch validate_model_for_glu_pruning to return True
            with patch('optipfair.pruning.utils.validate_model_for_glu_pruning', return_value=True):
                # Patch count_parameters to return a fixed value
                with patch('optipfair.cli.commands.count_parameters', return_value=100_000_000):
                    result = self.runner.invoke(analyze, [
                        '--model-path', 'test-model',
                        '--device', 'cpu'
                    ])
        
        # Check that the command executed successfully
        self.assertEqual(result.exit_code, 0)
        mock_model_class.from_pretrained.assert_called_once_with(
            'test-model',
            device_map='cpu'
        )

if __name__ == '__main__':
    unittest.main()