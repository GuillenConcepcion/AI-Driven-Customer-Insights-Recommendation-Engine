"""
Synthetic data generator mimicking the UCSD / RecSysDatasets Amazon Review benchmark.
Generates realistic customer purchase/review histories, product catalog, and behavioral personas.
"""

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any
from ..config import data_config, RAW_DATA_DIR


PRODUCT_NAMES = {
    "Electronics": [
        "Ultra HD 4K Action Camera", "Noise Cancelling Wireless Headphones", "Portable Bluetooth Speaker Pro",
        "Smart Home Security Cam", "Fast Wireless Charging Stand", "High-Speed HDMI 2.1 Cable",
        "Multi-Port USB-C Hub Adapter", "True Wireless Earbuds with ANC", "Digital Voice Recorder",
        "Power Bank 20000mAh PD Fast Charge"
    ],
    "Computers & Accessories": [
        "Ergonomic Mechanical Keyboard RGB", "Precision Wireless Gaming Mouse", "USB-C Dual 4K Monitor Dock",
        "High-Performance 2TB NVMe SSD", "Curved Ultrawide Gaming Monitor", "Laptop Cooling Pad with 5 Fans",
        "Webcam 1080p with Dual Stereo Mics", "Adjustable Aluminum Laptop Stand", "Wi-Fi 6E Dual-Band PCIe Card",
        "Durable Braided DisplayPort Cable"
    ],
    "Video Games": [
        "Wireless Controller for PC and Console", "Surround Sound Gaming Headset", "Arcade Fighting Stick Joystick",
        "MicroSDXC Card 512GB High Speed", "RGB Gaming Mouse Pad Extended", "Racing Steering Wheel with Pedals",
        "Console Vertical Cooling Stand", "Custom Silicone Controller Grip", "VR Gaming Headset Carry Case",
        "Analog Stick Precision Rings Pack"
    ],
    "Musical Instruments": [
        "Cardioid Condenser Studio Microphone", "Professional Audio Interface USB", "Digital Electric Keyboard 61 Keys",
        "Acoustic Guitar Starter Pack", "Over-Ear Studio Monitor Headphones", "Adjustable Heavy-Duty Boom Arm",
        "Studio Monitor Isolation Pads", "Guitar Digital Tuner and Capo Kit", "XLR Balanced Audio Cable 10ft",
        "MIDI Controller Keyboard 25-Key"
    ],
    "Home & Kitchen": [
        "Programmable Smart Coffee Maker", "Stainless Steel Electric Gooseneck Kettle", "Air Fryer XL 5.8 Quart",
        "Digital Precision Food Kitchen Scale", "Personal Countertop Blender", "Automatic Touchless Trash Can",
        "Cast Iron Enameled Dutch Oven", "HEPA Air Purifier for Large Rooms", "Smart LED Under Cabinet Lighting",
        "Stainless Steel Chef Knife 8-Inch"
    ],
    "Books": [
        "Designing Machine Learning Systems", "Hands-On Machine Learning with Scikit-Learn", "Designing Data-Intensive Applications",
        "Clean Code and Architecture Handbook", "Python for Data Analysis and Science", "Building Cloud-Native MLOps Pipelines",
        "Deep Learning with PyTorch in Action", "System Design Interview Insider Guide", "Modern Microservices with Docker",
        "Practical Time Series Analysis & Forecasting"
    ],
    "Health & Personal Care": [
        "Sonic Electric Toothbrush with UV Sanitizer", "Deep Tissue Percussion Massage Gun", "Smart Digital Body Fat Bathroom Scale",
        "Aromatherapy Essential Oil Diffuser", "Infrared Non-Contact Forehead Thermometer", "Cordless Water Flosser Dental Cleaner",
        "Ergonomic Memory Foam Lumbar Support Cushion", "Blue Light Blocking Computer Glasses", "Rechargeable Hot Compress Eye Mask",
        "Portable UV Phone & Tool Sanitizer Box"
    ]
}

BRANDS = ["AcuSound", "ApexTech", "Chronos", "CyberPro", "Lumina", "NexusWare", "OmniGear", "PrimeWave", "TitanAudio", "Vanguard"]


def generate_synthetic_dataset(
    num_customers: int = None,
    num_products: int = None,
    num_reviews: int = None,
    seed: int = None,
    save_to_disk: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates realistic synthetic Amazon Reviews and Product Catalog matching RecSysDatasets schema.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (interactions_df, products_df)
    """
    n_cust = num_customers or data_config.num_customers
    n_prod = num_products or data_config.num_products
    n_rev = num_reviews or data_config.num_reviews
    rng_seed = seed or data_config.random_seed

    np.random.seed(rng_seed)
    random.seed(rng_seed)

    # 1. Generate Products Catalog
    products = []
    prod_id = 1000
    for category, items in PRODUCT_NAMES.items():
        for name in items:
            prod_id += 1
            asin = f"B00{prod_id:05d}"
            brand = random.choice(BRANDS)
            base_price = round(random.uniform(15.0, 240.0), 2)
            products.append({
                "asin": asin,
                "title": f"{brand} {name}",
                "category": category,
                "brand": brand,
                "price": base_price,
                "sales_rank": random.randint(100, 50000)
            })

    # Pad or slice to desired num_products
    while len(products) < n_prod:
        prod_id += 1
        asin = f"B00{prod_id:05d}"
        cat = random.choice(list(PRODUCT_NAMES.keys()))
        name = random.choice(PRODUCT_NAMES[cat])
        brand = random.choice(BRANDS)
        products.append({
            "asin": asin,
            "title": f"{brand} {name} Plus",
            "category": cat,
            "brand": brand,
            "price": round(random.uniform(15.0, 240.0), 2),
            "sales_rank": random.randint(100, 50000)
        })
    products_df = pd.DataFrame(products[:n_prod])

    # 2. Generate Customers and Behavioral Personas
    # Personas:
    # 0: Champions / VIPs (15%) - high frequency, high spend, recent reviews
    # 1: Loyalists (25%) - medium-high frequency, steady activity
    # 2: Potential Loyalists (20%) - new users, active recently
    # 3: At-Risk Customers (20%) - active in 2024, dormant in 2025
    # 4: Occasional / Casuals (20%) - low frequency (1-2 reviews)
    customer_ids = [f"A{i:06d}{random.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}" for i in range(1, n_cust + 1)]
    persona_types = np.random.choice([0, 1, 2, 3, 4], size=n_cust, p=[0.15, 0.25, 0.20, 0.20, 0.20])

    start_date = datetime.strptime(data_config.date_start, "%Y-%m-%d")
    end_date = datetime.strptime(data_config.date_end, "%Y-%m-%d")
    mid_point = start_date + (end_date - start_date) * 0.60
    recent_cutoff = end_date - timedelta(days=120)

    # Pre-calculate probabilities of picking each product (power-law popularity)
    product_asins = products_df["asin"].values
    prod_weights = 1.0 / (np.arange(1, len(product_asins) + 1) ** 0.65)
    prod_weights /= prod_weights.sum()

    reviews = []
    
    # Review text bank
    positive_reviews = [
        "Exceptional build quality, exceeds expectations for the price point.",
        "Works straight out of the box. Highly recommended for daily use!",
        "Fantastic performance and great design. Would definitely purchase again.",
        "Very pleased with this purchase. Super fast delivery and great packaging.",
        "Five stars! Reliable, solid construction and crisp performance."
    ]
    neutral_reviews = [
        "Decent product for the cost, but has a few minor design quirks.",
        "Met my basic expectations. Not premium, but good value overall.",
        "Works as advertised. Average battery life and fair ergonomics."
    ]
    negative_reviews = [
        "Not satisfied with the durability. Stopped functioning as expected.",
        "Customer support was slow and the product felt flimsy.",
        "Doesn't live up to the marketing claims. Disappointed with the quality."
    ]

    for cust_id, p_type in zip(customer_ids, persona_types):
        if p_type == 0:  # Champions
            n_purchases = random.randint(12, 28)
            time_dist = lambda: recent_cutoff + timedelta(days=random.randint(0, 119))
            rating_probs = [0.02, 0.03, 0.05, 0.30, 0.60]
        elif p_type == 1:  # Loyalists
            n_purchases = random.randint(7, 16)
            time_dist = lambda: start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
            rating_probs = [0.05, 0.05, 0.10, 0.40, 0.40]
        elif p_type == 2:  # Potential Loyalists
            n_purchases = random.randint(4, 9)
            time_dist = lambda: recent_cutoff - timedelta(days=random.randint(0, 90)) + timedelta(days=random.randint(0, 200))
            rating_probs = [0.05, 0.05, 0.15, 0.45, 0.30]
        elif p_type == 3:  # At-Risk
            n_purchases = random.randint(5, 14)
            time_dist = lambda: start_date + timedelta(days=random.randint(0, int((mid_point - start_date).days)))
            rating_probs = [0.15, 0.15, 0.20, 0.30, 0.20]
        else:  # Occasional
            n_purchases = random.randint(1, 3)
            time_dist = lambda: start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
            rating_probs = [0.10, 0.10, 0.20, 0.30, 0.30]

        # Sample products without replacement if possible
        k = min(n_purchases, len(product_asins))
        chosen_asins = np.random.choice(product_asins, size=k, replace=False, p=prod_weights)

        for asin in chosen_asins:
            dt = min(time_dist(), end_date)
            overall = float(np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], p=rating_probs))
            
            if overall >= 4.0:
                rev_text = random.choice(positive_reviews)
                summary = "Great quality product"
            elif overall == 3.0:
                rev_text = random.choice(neutral_reviews)
                summary = "Average experience"
            else:
                rev_text = random.choice(negative_reviews)
                summary = "Could be better"

            reviews.append({
                "reviewerID": cust_id,
                "asin": asin,
                "overall": overall,
                "reviewTime": dt.strftime("%m %d, %Y"),
                "unixReviewTime": int(dt.timestamp()),
                "timestamp": dt,
                "reviewText": rev_text,
                "summary": summary,
                "verified": random.random() > 0.10
            })

    interactions_df = pd.DataFrame(reviews)

    # Sort chronologically
    interactions_df = interactions_df.sort_values("unixReviewTime").reset_index(drop=True)

    if save_to_disk:
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        products_df.to_csv(RAW_DATA_DIR / "products_metadata.csv", index=False)
        interactions_df.to_parquet(RAW_DATA_DIR / "interactions.parquet", index=False)
        # Also store small sample in CSV for inspection
        interactions_df.head(1000).to_csv(RAW_DATA_DIR / "interactions_sample.csv", index=False)

    return interactions_df, products_df


if __name__ == "__main__":
    df_int, df_prod = generate_synthetic_dataset()
    print(f"Generated {len(df_int)} interactions across {df_int['reviewerID'].nunique()} customers and {len(df_prod)} products.")
