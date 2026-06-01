#!/usr/bin/env python3
"""
Download datasets for ML Sentinel experiments.

This script downloads and prepares:
1. Folktables (ACS data) - via folktables package
2. CivilComments - via WILDS package
3. Amazon Reviews - via WILDS package

Usage:
    python download.py
    python download.py --dataset folktables
    python download.py --dataset civilcomments
    python download.py --dataset amazon
"""

import argparse
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent


def download_folktables():
    """Download Folktables ACS data."""
    print("\n📥 Downloading Folktables (ACS Income)...")
    
    try:
        from folktables import ACSDataSource, ACSIncome
        
        # Download for all states
        data_source = ACSDataSource(
            survey_year='2018',
            horizon='1-Year',
            survey='person'
        )
        
        # Just download California as reference
        acs_data = data_source.get_data(states=['CA'], download=True)
        
        print("✓ Folktables data downloaded")
        print(f"  Location: ~/.cache/folktables/")
        print(f"  States available: All 50 + PR")
        
    except ImportError:
        print("❌ folktables not installed. Run: pip install folktables")
    except Exception as e:
        print(f"❌ Error: {e}")


def download_civilcomments():
    """Download CivilComments from WILDS."""
    print("\n📥 Downloading CivilComments...")
    
    try:
        from wilds import get_dataset
        
        dataset = get_dataset(
            dataset='civilcomments',
            root_dir=str(DATA_DIR / 'wilds'),
            download=True
        )
        
        print("✓ CivilComments downloaded")
        print(f"  Location: {DATA_DIR / 'wilds/civilcomments_v1.0'}")
        print(f"  Train size: {len(dataset.get_subset('train'))}")
        print(f"  Test size: {len(dataset.get_subset('test'))}")
        
    except ImportError:
        print("❌ wilds not installed. Run: pip install wilds")
    except Exception as e:
        print(f"❌ Error: {e}")


def download_amazon():
    """Download Amazon Reviews from WILDS."""
    print("\n📥 Downloading Amazon Reviews...")
    
    try:
        from wilds import get_dataset
        
        dataset = get_dataset(
            dataset='amazon',
            root_dir=str(DATA_DIR / 'wilds'),
            download=True
        )
        
        print("✓ Amazon Reviews downloaded")
        print(f"  Location: {DATA_DIR / 'wilds/amazon_v2.1'}")
        print(f"  Train size: {len(dataset.get_subset('train'))}")
        print(f"  Test size: {len(dataset.get_subset('test'))}")
        
    except ImportError:
        print("❌ wilds not installed. Run: pip install wilds")
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Download datasets for ML Sentinel experiments'
    )
    parser.add_argument(
        '--dataset',
        choices=['all', 'folktables', 'civilcomments', 'amazon'],
        default='all',
        help='Which dataset to download (default: all)'
    )
    
    args = parser.parse_args()
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║              ML SENTINEL DATA DOWNLOADER                      ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    if args.dataset in ['all', 'folktables']:
        download_folktables()
    
    if args.dataset in ['all', 'civilcomments']:
        download_civilcomments()
    
    if args.dataset in ['all', 'amazon']:
        download_amazon()
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                    DOWNLOAD COMPLETE                          ║
║                                                               ║
║   Next: python experiments/run_all.py                         ║
╚═══════════════════════════════════════════════════════════════╝
    """)


if __name__ == '__main__':
    main()
