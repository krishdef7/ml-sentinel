"""
Utility functions for ML Sentinel.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import json
from pathlib import Path
from datetime import datetime


def load_results(filepath: str) -> pd.DataFrame:
    """
    Load experiment results from CSV.
    
    Parameters
    ----------
    filepath : str
        Path to results CSV
        
    Returns
    -------
    pd.DataFrame
        Results dataframe
    """
    return pd.read_csv(filepath)


def save_results(
    results: Dict[str, Any],
    filepath: str,
    format: str = 'json'
) -> None:
    """
    Save experiment results.
    
    Parameters
    ----------
    results : dict
        Results dictionary
    filepath : str
        Output path
    format : str
        'json' or 'csv'
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    elif format == 'csv':
        pd.DataFrame(results).to_csv(filepath, index=False)
    else:
        raise ValueError(f"Unknown format: {format}")


def compute_degradation_stats(
    ref_accuracy: float,
    test_accuracy: float,
    n_samples: int,
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    Compute degradation statistics.
    
    Parameters
    ----------
    ref_accuracy : float
        Reference (training/validation) accuracy
    test_accuracy : float
        Test accuracy
    n_samples : int
        Number of test samples
    significance_level : float
        Alpha for significance testing
        
    Returns
    -------
    dict
        Degradation statistics
    """
    from scipy import stats
    
    drop = ref_accuracy - test_accuracy
    
    # Standard error using normal approximation
    se = np.sqrt(test_accuracy * (1 - test_accuracy) / n_samples)
    
    # Z-test for degradation
    z = drop / se if se > 0 else 0
    p_value = 1 - stats.norm.cdf(z)
    
    return {
        'ref_accuracy': ref_accuracy,
        'test_accuracy': test_accuracy,
        'accuracy_drop': drop,
        'standard_error': se,
        'z_statistic': z,
        'p_value': p_value,
        'is_significant': p_value < significance_level,
        'n_samples': n_samples,
    }


def format_ci(
    point: float,
    ci_lower: float,
    ci_upper: float,
    decimals: int = 2
) -> str:
    """
    Format confidence interval as string.
    
    Example: "0.91 [0.85, 0.94]"
    """
    return f"{point:.{decimals}f} [{ci_lower:.{decimals}f}, {ci_upper:.{decimals}f}]"


def print_summary_table(results: Dict[str, Dict]) -> None:
    """
    Print formatted summary table.
    
    Parameters
    ----------
    results : dict
        Results by benchmark name
    """
    print("\n" + "="*70)
    print(f"{'Benchmark':<15} {'Degraded':>10} {'Detected':>10} {'TPR':>8} {'Sentinel r':>12}")
    print("="*70)
    
    total_deg = 0
    total_det = 0
    
    for name, r in results.items():
        n_deg = r.get('n_degraded', 0)
        n_det = int(r.get('drift_tpr', 0) * n_deg)
        tpr = r.get('drift_tpr', 0)
        sent_r = r.get('sentinel_r', 0)
        
        total_deg += n_deg
        total_det += n_det
        
        print(f"{name:<15} {n_deg:>10} {n_det:>10} {tpr:>7.0%} {sent_r:>12.2f}")
    
    print("-"*70)
    overall_tpr = total_det / total_deg if total_deg > 0 else 0
    print(f"{'TOTAL':<15} {total_deg:>10} {total_det:>10} {overall_tpr:>7.0%}")
    print("="*70)


def create_experiment_metadata() -> Dict[str, Any]:
    """
    Create metadata for experiment tracking.
    
    Returns
    -------
    dict
        Metadata including timestamp, versions, etc.
    """
    import platform
    import sys
    
    return {
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'platform': platform.platform(),
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__,
    }


class ExperimentLogger:
    """
    Simple experiment logger.
    
    Example
    -------
    >>> logger = ExperimentLogger('results/experiment.log')
    >>> logger.log('Starting experiment')
    >>> logger.log('Benchmark: Folktables', level='INFO')
    """
    
    def __init__(self, filepath: Optional[str] = None):
        self.filepath = filepath
        self.entries = []
    
    def log(self, message: str, level: str = 'INFO') -> None:
        """Log a message."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
        }
        self.entries.append(entry)
        
        # Print to console
        print(f"[{level}] {message}")
        
        # Write to file if specified
        if self.filepath:
            with open(self.filepath, 'a') as f:
                f.write(f"{entry['timestamp']} [{level}] {message}\n")
    
    def save(self, filepath: str) -> None:
        """Save all log entries to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.entries, f, indent=2)
