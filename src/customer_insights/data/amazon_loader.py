"""
Data loader and parser for official RecSysDatasets / Amazon Product Reviews (UCSD / Julian McAuley).
Supports streaming JSON.gz, Parquet, and CSV formats.
"""

import gzip
import json
import urllib.request
from pathlib import Path
from typing import Optional, Tuple, Iterator, Dict, Any
import pandas as pd
from ..config import RAW_DATA_DIR


UCSD_BASE_URL = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall"
# Popular lightweight 5-core benchmarks
AVAILABLE_BENCHMARKS = {
    "musical_instruments": "Musical_Instruments_5.json.gz",
    "video_games": "Video_Games_5.json.gz",
    "digital_music": "Digital_Music_5.json.gz",
    "prime_pantry": "Prime_Pantry_5.json.gz"
}


class AmazonDataLoader:
    """
    Ingests and normalizes Amazon Reviews and Metadata from RecSysDatasets repository.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or RAW_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_benchmark(self, category: str = "musical_instruments") -> Path:
        """
        Downloads a 5-core category JSON.gz file from UCSD repository if not already cached.
        """
        if category not in AVAILABLE_BENCHMARKS:
            raise ValueError(f"Category {category} not recognized. Choose from: {list(AVAILABLE_BENCHMARKS.keys())}")

        filename = AVAILABLE_BENCHMARKS[category]
        target_path = self.data_dir / filename
        
        if not target_path.exists():
            download_url = f"{UCSD_BASE_URL}/{filename}"
            print(f"Downloading benchmark from {download_url} to {target_path}...")
            urllib.request.urlretrieve(download_url, target_path)
            print("Download completed successfully.")
        else:
            print(f"File {target_path.name} already exists in cache.")

        return target_path

    @staticmethod
    def parse_gzip_json(file_path: Path, max_records: Optional[int] = None) -> pd.DataFrame:
        """
        Parses a .json.gz file line-by-line into a normalized DataFrame.
        """
        records = []
        count = 0
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    records.append({
                        "reviewerID": item.get("reviewerID"),
                        "asin": item.get("asin"),
                        "overall": float(item.get("overall", 0.0)),
                        "unixReviewTime": int(item.get("unixReviewTime", 0)),
                        "reviewTime": item.get("reviewTime"),
                        "reviewText": item.get("reviewText", ""),
                        "summary": item.get("summary", ""),
                        "verified": bool(item.get("verified", False))
                    })
                    count += 1
                    if max_records and count >= max_records:
                        break

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["unixReviewTime"], unit="s")
        return df

    def load_or_fallback(
        self,
        category: Optional[str] = None,
        max_records: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads cached processed interactions if present, otherwise attempts benchmark download
        or falls back gracefully to synthetic generation.
        """
        parquet_file = self.data_dir / "interactions.parquet"
        products_file = self.data_dir / "products_metadata.csv"

        if parquet_file.exists() and products_file.exists():
            interactions_df = pd.read_parquet(parquet_file)
            products_df = pd.read_csv(products_file)
            return interactions_df, products_df

        if category:
            try:
                gz_path = self.download_benchmark(category)
                interactions_df = self.parse_gzip_json(gz_path, max_records=max_records)
                
                # Derive minimal product catalog from interactions
                products = []
                for asin in interactions_df["asin"].unique():
                    products.append({
                        "asin": asin,
                        "title": f"Product {asin}",
                        "category": category.replace("_", " ").title(),
                        "brand": "Amazon Vendor",
                        "price": 29.99,
                        "sales_rank": 1000
                    })
                products_df = pd.DataFrame(products)
                return interactions_df, products_df
            except (requests.RequestException, IOError, ValueError, KeyError) as e:
                print(f"Warning: Could not stream or parse external benchmark ({type(e).__name__}: {e}). Using synthetic generator fallback.")

        # Fallback to synthetic generator
        from .generator import generate_synthetic_dataset
        return generate_synthetic_dataset(save_to_disk=True)
