"""
Main entry point for MNIST Dimensionality Reduction and Classification Project.

This script runs experiments comparing PCA and Kernel PCA with SVM and Random Forest
classifiers on the MNIST dataset.
"""

import argparse
import os
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='MNIST Dimensionality Reduction and Classification Experiment')
    parser.add_argument('configs', nargs='*', default=None,
                        help='Config names to run (e.g. A C B_rbf G_90). If omitted, runs all configs.')
    args = parser.parse_args()

    config_filter = args.configs if args.configs else None

    # Create results directory
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)

    print("="*70)
    print("MNIST Dimensionality Reduction and Classification Experiment")
    print("="*70)

    # Import here to avoid circular imports
    from src.experiment import run_experiment, summarize_results

    try:
        # Run the experiment
        all_results = run_experiment(output_dir=results_dir, config_filter=config_filter)

        if not all_results:
            print("No results produced.")
            sys.exit(0)

        # Summarize results
        summarize_results(all_results, output_dir=results_dir)

        print("\n" + "="*70)
        print("Experiment completed successfully!")
        print(f"Results saved to: {results_dir}")
        print("="*70)

    except Exception as e:
        print(f"\nError during experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
