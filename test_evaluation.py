#!/usr/bin/env python3
"""Test the ARC evaluation system to ensure it works correctly."""

import numpy as np
from evaluators.arc import grid_hash, crop_padding, ARCEvaluator, ARCTestDataset


def test_grid_hash():
    """Test that grid hashing works correctly."""
    print("Testing grid_hash function...")

    # Test identical grids produce same hash
    grid1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    grid2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    hash1 = grid_hash(grid1)
    hash2 = grid_hash(grid2)
    assert hash1 == hash2, "Identical grids should have same hash"
    print(f"  ✓ Identical grids: {hash1[:8]}... == {hash2[:8]}...")

    # Test different grids produce different hashes
    grid3 = np.array([[1, 2], [3, 5]], dtype=np.int32)
    hash3 = grid_hash(grid3)
    assert hash1 != hash3, "Different grids should have different hashes"
    print(f"  ✓ Different grids: {hash1[:8]}... != {hash3[:8]}...")

    # Test with different shapes
    grid4 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    hash4 = grid_hash(grid4)
    assert hash1 != hash4, "Different shapes should have different hashes"
    print(f"  ✓ Different shapes: {hash1[:8]}... != {hash4[:8]}...")

    print("  All hash tests passed!\n")


def test_crop_padding():
    """Test padding removal function."""
    print("Testing crop_padding function...")

    # Test simple case
    # 3x3 grid flattened: [[1,2,10],[3,4,10],[10,10,10]]
    padded = np.array([1, 2, 10, 3, 4, 10, 10, 10, 10])
    expected = np.array([[1, 2], [3, 4]])
    result = crop_padding(padded, padding_value=10)
    assert np.array_equal(result, expected), f"Expected {expected}, got {result}"
    print(f"  ✓ Simple padding removal: {padded.shape} → {result.shape}")

    # Test no padding
    no_pad = np.array([1, 2, 3, 4])  # 2x2 with no padding
    expected = np.array([[1, 2], [3, 4]])
    result = crop_padding(no_pad, padding_value=10)
    assert np.array_equal(result, expected), f"Expected {expected}, got {result}"
    print(f"  ✓ No padding case: {no_pad.shape} → {result.shape}")

    # Test all padding
    all_pad = np.array([10, 10, 10, 10])  # All padding
    result = crop_padding(all_pad, padding_value=10)
    assert result.shape == (0, 0) or result.size == 0, f"All padding should return empty array, got {result}"
    print(f"  ✓ All padding case: {all_pad.shape} → empty array")

    print("  All padding tests passed!\n")


def test_evaluator():
    """Test the ARCEvaluator class."""
    print("Testing ARCEvaluator class...")

    # Create evaluator (will load actual tasks)
    evaluator = ARCEvaluator(
        data_dir="data/ARC-AGI/data",
        pass_ks=(1, 2, 5),
        verbose=False
    )

    # Check tasks were loaded
    assert len(evaluator.eval_tasks) > 0, "Should load evaluation tasks"
    print(f"  ✓ Loaded {len(evaluator.eval_tasks)} evaluation tasks")

    # Test adding predictions
    test_task_id = list(evaluator.eval_tasks.keys())[0]
    test_grid = np.array([[1, 2], [3, 4]], dtype=np.int32)

    evaluator.add_prediction(
        task_id=test_task_id,
        test_idx=0,
        pred_grid=test_grid,
        confidence=0.9
    )

    assert test_task_id in evaluator.predictions
    assert 0 in evaluator.predictions[test_task_id]
    assert len(evaluator.predictions[test_task_id][0]) == 1
    print(f"  ✓ Added prediction for task {test_task_id[:8]}...")

    # Test reset
    evaluator.reset()
    assert len(evaluator.predictions) == 0, "Reset should clear predictions"
    print("  ✓ Reset clears predictions")

    print("  All evaluator tests passed!\n")


def test_test_dataset():
    """Test the ARCTestDataset class."""
    print("Testing ARCTestDataset class...")

    # Load test dataset
    dataset = ARCTestDataset(
        data_dir="data/ARC-AGI/data",
        split="evaluation",
        include_output=True
    )

    # Check examples were loaded
    assert len(dataset) > 0, "Should load test examples"
    print(f"  ✓ Loaded {len(dataset)} test examples")

    # Check first example structure
    example = dataset[0]
    assert 'task_id' in example, "Example should have task_id"
    assert 'test_idx' in example, "Example should have test_idx"
    assert 'input' in example, "Example should have input"
    assert 'output' in example, "Example should have output (when include_output=True)"
    assert 'demonstrations' in example, "Example should have demonstrations"
    print(f"  ✓ Example structure correct: {list(example.keys())}")

    # Check data types
    assert isinstance(example['input'], np.ndarray), "Input should be numpy array"
    assert isinstance(example['output'], np.ndarray), "Output should be numpy array"
    assert example['input'].dtype == np.int32, "Input should be int32"
    assert example['output'].dtype == np.int32, "Output should be int32"
    print(f"  ✓ Data types correct: input {example['input'].shape}, output {example['output'].shape}")

    # Check demonstrations
    assert len(example['demonstrations']) > 0, "Should have at least one demonstration"
    demo = example['demonstrations'][0]
    assert 'input' in demo and 'output' in demo, "Demonstration should have input/output"
    print(f"  ✓ Demonstrations present: {len(example['demonstrations'])} demos")

    print("  All dataset tests passed!\n")


def main():
    """Run all tests."""
    print("="*60)
    print("Testing ARC Evaluation System")
    print("="*60 + "\n")

    try:
        test_grid_hash()
        test_crop_padding()
        test_evaluator()
        test_test_dataset()

        print("="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("\nThe evaluation system is working correctly.")
        print("You can now use:")
        print("  - python evaluate_arc.py <model_path>  # To evaluate a trained model")
        print("  - The evaluator in training via trainer_arc.py")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())