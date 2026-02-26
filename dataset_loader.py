# ai-service/data/dataset_loader.py
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import json
import os
import re
import random
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime, timedelta
import requests
from pathlib import Path
import warnings
import zipfile
import shutil
from typing import Dict, Optional
import subprocess
import sys

warnings.filterwarnings('ignore')

from transaction import TransactionCategory

load_dotenv()


class DatasetLoader:
    """Load and preprocess transaction datasets from multiple sources."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize dataset loader.
        
        Args:
            data_dir: Directory containing datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Store credentials (could also load from environment/config)
        self.kaggle_username = os.getenv('KAGGLE_USERNAME')
        self.kaggle_key = os.getenv('KAGGLE_KEY')
        
        # Common merchant patterns for augmentation
        self.merchant_patterns = {
            TransactionCategory.GROCERIES: [
                "Walmart", "Kroger", "Whole Foods", "Trader Joe's", "Aldi", "Costco",
                "Safeway", "Publix", "Target", "Meijer", "Giant", "Food Lion"
            ],
            TransactionCategory.DINING: [
                "Starbucks", "McDonald's", "Subway", "Chipotle", "Taco Bell",
                "Burger King", "Wendy's", "Domino's", "Pizza Hut", "Panera Bread",
                "Chick-fil-A", "Dunkin'", "Olive Garden", "Applebees"
            ],
            TransactionCategory.TRANSPORTATION: [
                "Uber", "Lyft", "Gas Station", "Shell", "BP", "Exxon", "Mobil",
                "Metro Transit", "Bus Ticket", "Train Ticket", "Parking"
            ],
            TransactionCategory.SHOPPING: [
                "Amazon", "eBay", "Walmart", "Target", "Best Buy", "Home Depot",
                "Lowe's", "Macy's", "Kohl's", "Nordstrom", "Old Navy", "Gap"
            ],
            TransactionCategory.ENTERTAINMENT: [
                "Netflix", "Spotify", "Disney+", "Hulu", "HBO Max", "YouTube Premium",
                "Movie Theater", "Concert", "Sports Game", "Bowling", "Arcade"
            ],
            TransactionCategory.UTILITIES: [
                "Electric Company", "Water Bill", "Gas Company", "Internet Service",
                "Phone Bill", "Cable TV", "Trash Service", "Sewer Bill"
            ],
            TransactionCategory.HOUSING: [
                "Rent Payment", "Mortgage", "Apartment", "Property Management",
                "Home Insurance", "Property Tax", "HOA Fees"
            ],
            TransactionCategory.HEALTHCARE: [
                "Hospital", "Doctor's Office", "Pharmacy", "Dentist", "Optometrist",
                "Health Insurance", "Medical Supplies", "Prescription"
            ]
        }
        
        # Common description patterns for each category
        self.description_patterns = {
            TransactionCategory.GROCERIES: [
                "Groceries at {merchant}",
                "Food shopping",
                "Weekly groceries",
                "Supermarket purchase",
                "{merchant} grocery",
                "Produce purchase"
            ],
            TransactionCategory.DINING: [
                "{merchant}",
                "Lunch at {merchant}",
                "Dinner at {merchant}",
                "Coffee at {merchant}",
                "Fast food",
                "Restaurant meal"
            ],
            TransactionCategory.TRANSPORTATION: [
                "{merchant} ride",
                "Gas at {merchant}",
                "Public transportation",
                "Parking fee",
                "Taxi service",
                "Car maintenance"
            ],
            TransactionCategory.SHOPPING: [
                "Online purchase {merchant}",
                "{merchant} order",
                "Clothing purchase",
                "Electronics {merchant}",
                "Home goods",
                "{merchant} shopping"
            ]
        }
        
        # Amount ranges for each category (min, max, typical)
        self.amount_ranges = {
            TransactionCategory.GROCERIES: (20, 200, 85),
            TransactionCategory.DINING: (5, 80, 25),
            TransactionCategory.TRANSPORTATION: (10, 100, 35),
            TransactionCategory.SHOPPING: (15, 500, 75),
            TransactionCategory.ENTERTAINMENT: (10, 150, 40),
            TransactionCategory.UTILITIES: (30, 400, 120),
            TransactionCategory.HOUSING: (500, 3000, 1500),
            TransactionCategory.HEALTHCARE: (20, 500, 100),
            TransactionCategory.SALARY: (2000, 10000, 4500),
            TransactionCategory.FREELANCE: (100, 5000, 800),
            TransactionCategory.SUBSCRIPTIONS: (5, 50, 15)
        }
    
    def set_kaggle_credentials(self, username: str, api_key: str):
        """Set Kaggle credentials for authenticated downloads."""
        self.kaggle_username = username
        self.kaggle_key = api_key
        
    def load_training_data(self, source: str = "all", 
                          augment: bool = True,
                          sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load training data from multiple sources.
        
        Args:
            source: Data source ('synthetic', 'kaggle', 'openfinancial', 'all')
            augment: Whether to augment the data
            sample_size: Number of samples to return (None for all)
            
        Returns:
            DataFrame with columns: ['description', 'category', 'amount', 'merchant']
        """
        dataframes = []
        
        if source in ['synthetic', 'all']:
            synth_data = self._load_synthetic_data()
            dataframes.append(synth_data)
        
        if source in ['kaggle', 'all']:
            kaggle_data = self._load_kaggle_dataset()
            if kaggle_data is not None:
                dataframes.append(kaggle_data)
        
        if source in ['openfinancial', 'all']:
            open_data = self._load_openfinancial_data()
            if open_data is not None:
                dataframes.append(open_data)
        
        if source in ['augmented', 'all'] and augment:
            augmented_data = self._augment_existing_data()
            dataframes.append(augmented_data)
        
        # Combine all data
        if dataframes:
            combined_data = pd.concat(dataframes, ignore_index=True)
            
            # Remove duplicates
            combined_data = combined_data.drop_duplicates(subset=['description', 'category'])
            
            # Sample if requested
            if sample_size and len(combined_data) > sample_size:
                combined_data = combined_data.sample(n=sample_size, random_state=42)
            
            print(f"Loaded {len(combined_data)} training samples")
            return combined_data
        else:
            print("No data loaded, generating synthetic data only")
            return self._load_synthetic_data(sample_size or 10000)
    
    def _load_synthetic_data(self, num_samples: int = 10000) -> pd.DataFrame:
        """
        Generate synthetic transaction data for training.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            DataFrame with synthetic transaction data
        """
        print(f"Generating {num_samples} synthetic transactions...")
        
        data = []
        categories = list(self.amount_ranges.keys())
        
        # Distribution weights (some categories are more common)
        weights = {
            TransactionCategory.GROCERIES: 0.15,
            TransactionCategory.DINING: 0.12,
            TransactionCategory.SHOPPING: 0.10,
            TransactionCategory.TRANSPORTATION: 0.08,
            TransactionCategory.ENTERTAINMENT: 0.07,
            TransactionCategory.SUBSCRIPTIONS: 0.05,
            TransactionCategory.UTILITIES: 0.06,
            TransactionCategory.HOUSING: 0.05,
            TransactionCategory.HEALTHCARE: 0.04,
            TransactionCategory.SALARY: 0.08,
            TransactionCategory.FREELANCE: 0.05,
            TransactionCategory.OTHER: 0.15
        }
        
        # Normalize weights
        total_weight = sum(weights.get(cat, 0.05) for cat in categories)
        for cat in categories:
            if cat not in weights:
                weights[cat] = 0.05 / len([c for c in categories if c not in weights])
        
        # Generate samples
        for _ in range(num_samples):
            # Select category based on weights
            category = random.choices(
                categories, 
                weights=[weights.get(cat, 0.05) for cat in categories]
            )[0]
            
            # Generate description
            description = self._generate_description(category)
            
            # Generate amount based on category
            if category in self.amount_ranges:
                min_amt, max_amt, typical = self.amount_ranges[category]
                # Use normal distribution around typical value
                amount = max(min_amt, min(max_amt, 
                    np.random.normal(typical, (max_amt - min_amt) / 4)))
                amount = round(amount, 2)
            else:
                amount = round(random.uniform(5, 200), 2)
            
            # Add merchant if applicable
            merchant = None
            if category in self.merchant_patterns:
                merchant = random.choice(self.merchant_patterns[category])
            
            data.append({
                'description': description,
                'category': category.value if hasattr(category, 'value') else category,
                'amount': amount,
                'merchant': merchant,
                'is_synthetic': True
            })
        
        df = pd.DataFrame(data)
        print(f"Generated {len(df)} synthetic transactions")
        return df
    
    def _generate_description(self, category: TransactionCategory) -> str:
        """Generate a realistic transaction description."""
        if category in self.description_patterns:
            pattern = random.choice(self.description_patterns[category])
            if '{merchant}' in pattern and category in self.merchant_patterns:
                merchant = random.choice(self.merchant_patterns[category])
                return pattern.format(merchant=merchant)
            return pattern
        
        # Fallback descriptions by category
        fallback_descriptions = {
            TransactionCategory.SALARY: [
                "Monthly Salary Deposit",
                "Paycheck from {company}",
                "Direct Deposit Salary"
            ],
            TransactionCategory.FREELANCE: [
                "Freelance Payment",
                "Contract Work Invoice #{num}",
                "Consulting Services"
            ],
            TransactionCategory.INVESTMENT: [
                "Dividend Payment",
                "Stock Sale",
                "Investment Return"
            ],
            TransactionCategory.GIFT: [
                "Gift from {name}",
                "Birthday Gift",
                "Holiday Gift"
            ],
            TransactionCategory.EDUCATION: [
                "Tuition Payment",
                "Textbook Purchase",
                "Online Course"
            ],
            TransactionCategory.TRAVEL: [
                "Hotel Booking",
                "Flight Ticket",
                "Vacation Expenses"
            ]
        }
        
        if category in fallback_descriptions:
            pattern = random.choice(fallback_descriptions[category])
            if '{company}' in pattern:
                companies = ["Acme Inc", "Tech Corp", "Global Solutions", "Innovation Co"]
                return pattern.format(company=random.choice(companies))
            elif '{name}' in pattern:
                names = ["John", "Sarah", "Mike", "Emily", "David", "Lisa"]
                return pattern.format(name=random.choice(names))
            elif '{num}' in pattern:
                return pattern.format(num=random.randint(1000, 9999))
            return pattern
        
        # Generic description
        generic = [
            f"{category.value.replace('_', ' ').title()} Transaction",
            "Payment for services",
            "Purchase",
            "Bank Transfer"
        ]
        return random.choice(generic)
    
    def _load_kaggle_dataset(self) -> Optional[pd.DataFrame]:
        """
        Load transaction dataset from Kaggle (if available).
        
        Returns:
            DataFrame with Kaggle data or None
        """
        kaggle_path = self.data_dir / "kaggle" / "transactions.csv"
        
        if kaggle_path.exists():
            try:
                print("Loading Kaggle dataset...")
                df = pd.read_csv(kaggle_path)
                
                # Standardize column names
                column_mapping = {
                    'Description': 'description',
                    'desc': 'description',
                    'Category': 'category',
                    'category_name': 'category',
                    'Amount': 'amount',
                    'Transaction Amount': 'amount',
                    'Merchant': 'merchant',
                    'merchant_name': 'merchant'
                }
                
                df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
                
                # Ensure required columns exist
                required_cols = ['description', 'category']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    print(f"Missing columns in Kaggle data: {missing_cols}")
                    return None
                
                # Clean and filter
                df = df.dropna(subset=['description', 'category'])
                df['description'] = df['description'].astype(str).str.strip()
                df['category'] = df['category'].astype(str).str.strip()
                
                # Filter out invalid categories
                valid_categories = [cat.value for cat in TransactionCategory]
                df = df[df['category'].isin(valid_categories)]
                
                print(f"Loaded {len(df)} samples from Kaggle")
                return df
                
            except Exception as e:
                print(f"Error loading Kaggle dataset: {e}")
        
        print("Kaggle dataset not found")
        return None
    
    def _load_openfinancial_data(self) -> Optional[pd.DataFrame]:
        """
        Load data from Open Financial datasets.
        
        Returns:
            DataFrame with open financial data or None
        """
        open_data_paths = [
            self.data_dir / "open_data" / "bank_transactions.json",
            self.data_dir / "open_data" / "transactions_dataset.csv",
            self.data_dir / "open_data" / "financial_transactions.jsonl"
        ]
        
        for data_path in open_data_paths:
            if data_path.exists():
                try:
                    print(f"Loading open data from {data_path.name}...")
                    
                    if data_path.suffix == '.csv':
                        df = pd.read_csv(data_path)
                    elif data_path.suffix == '.json':
                        df = pd.read_json(data_path)
                    elif data_path.suffix == '.jsonl':
                        df = pd.read_json(data_path, lines=True)
                    else:
                        continue
                    
                    # Map columns to standard format
                    standard_df = self._standardize_dataframe(df)
                    
                    if standard_df is not None and len(standard_df) > 0:
                        print(f"Loaded {len(standard_df)} samples from open data")
                        return standard_df
                        
                except Exception as e:
                    print(f"Error loading {data_path}: {e}")
        
        print("No open financial data found")
        return None
    
    def _standardize_dataframe(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Standardize dataframe columns and format.
        
        Args:
            df: Raw dataframe
            
        Returns:
            Standardized dataframe or None
        """
        if df.empty:
            return None
        
        # Common column name mappings
        column_mappings = {
            'description': ['description', 'desc', 'transaction_description', 'details', 'narration'],
            'category': ['category', 'category_name', 'type', 'transaction_type', 'label'],
            'amount': ['amount', 'transaction_amount', 'amt', 'value'],
            'merchant': ['merchant', 'merchant_name', 'vendor', 'store', 'shop']
        }
        
        result_df = pd.DataFrame()
        
        # Map each required column
        for target_col, possible_cols in column_mappings.items():
            for possible_col in possible_cols:
                if possible_col in df.columns:
                    result_df[target_col] = df[possible_col]
                    break
        
        # If we don't have required columns, return None
        if 'description' not in result_df.columns or 'category' not in result_df.columns:
            return None
        
        # Clean data
        result_df = result_df.dropna(subset=['description', 'category'])
        result_df['description'] = result_df['description'].astype(str).str.strip()
        result_df['category'] = result_df['category'].astype(str).str.strip().str.lower()
        
        # Map common category names to standard categories
        category_mapping = {
            # Groceries
            'grocery': TransactionCategory.GROCERIES.value,
            'groceries': TransactionCategory.GROCERIES.value,
            'supermarket': TransactionCategory.GROCERIES.value,
            'food': TransactionCategory.GROCERIES.value,
            
            # Dining
            'restaurant': TransactionCategory.DINING.value,
            'dining': TransactionCategory.DINING.value,
            'food & drink': TransactionCategory.DINING.value,
            'fast food': TransactionCategory.DINING.value,
            'coffee': TransactionCategory.DINING.value,
            
            # Transportation
            'transport': TransactionCategory.TRANSPORTATION.value,
            'transportation': TransactionCategory.TRANSPORTATION.value,
            'gas': TransactionCategory.TRANSPORTATION.value,
            'fuel': TransactionCategory.TRANSPORTATION.value,
            'uber': TransactionCategory.TRANSPORTATION.value,
            'lyft': TransactionCategory.TRANSPORTATION.value,
            
            # Shopping
            'shopping': TransactionCategory.SHOPPING.value,
            'retail': TransactionCategory.SHOPPING.value,
            'amazon': TransactionCategory.SHOPPING.value,
            
            # Entertainment
            'entertainment': TransactionCategory.ENTERTAINMENT.value,
            'movies': TransactionCategory.ENTERTAINMENT.value,
            'streaming': TransactionCategory.ENTERTAINMENT.value,
            'netflix': TransactionCategory.ENTERTAINMENT.value,
            
            # Utilities
            'utilities': TransactionCategory.UTILITIES.value,
            'electricity': TransactionCategory.UTILITIES.value,
            'water': TransactionCategory.UTILITIES.value,
            'internet': TransactionCategory.UTILITIES.value,
            'phone': TransactionCategory.UTILITIES.value,
            
            # Housing
            'rent': TransactionCategory.HOUSING.value,
            'mortgage': TransactionCategory.HOUSING.value,
            'housing': TransactionCategory.HOUSING.value,
            
            # Salary
            'salary': TransactionCategory.SALARY.value,
            'income': TransactionCategory.SALARY.value,
            'payroll': TransactionCategory.SALARY.value,
            
            # Subscriptions
            'subscription': TransactionCategory.SUBSCRIPTIONS.value,
            'membership': TransactionCategory.SUBSCRIPTIONS.value,
        }
        
        result_df['category'] = result_df['category'].map(
            lambda x: category_mapping.get(x, TransactionCategory.OTHER.value)
        )
        
        # Add amount if missing (generate realistic amounts based on category)
        if 'amount' not in result_df.columns:
            result_df['amount'] = result_df['category'].apply(
                lambda cat: self._generate_amount_for_category(cat)
            )
        
        return result_df
    
    def _generate_amount_for_category(self, category: str) -> float:
        """Generate realistic amount for a category."""
        for enum_category in TransactionCategory:
            if enum_category.value == category and enum_category in self.amount_ranges:
                min_amt, max_amt, typical = self.amount_ranges[enum_category]
                return round(random.uniform(min_amt, max_amt), 2)
        
        # Default amount
        return round(random.uniform(10, 200), 2)
    
    def _augment_existing_data(self) -> pd.DataFrame:
        """
        Augment existing data with variations.
        
        Returns:
            Augmented dataframe
        """
        print("Augmenting existing data...")
        
        # Load existing data
        existing_data = []
        
        # Try to load any existing data first
        for source in ['kaggle', 'openfinancial']:
            if source == 'kaggle':
                data = self._load_kaggle_dataset()
            else:
                data = self._load_openfinancial_data()
            
            if data is not None:
                existing_data.append(data)
        
        if not existing_data:
            # If no existing data, create some base data
            base_data = self._load_synthetic_data(1000)
            existing_data.append(base_data)
        
        # Combine existing data
        combined = pd.concat(existing_data, ignore_index=True) if len(existing_data) > 1 else existing_data[0]
        
        # Augmentation techniques
        augmented_samples = []
        
        for _, row in combined.iterrows():
            original_desc = row['description']
            category = row['category']
            
            # Create variations of the description
            variations = self._create_description_variations(original_desc, category)
            
            for variation in variations:
                augmented_samples.append({
                    'description': variation,
                    'category': category,
                    'amount': row.get('amount', self._generate_amount_for_category(category)),
                    'merchant': row.get('merchant'),
                    'is_augmented': True
                })
        
        augmented_df = pd.DataFrame(augmented_samples)
        
        # Remove exact duplicates
        augmented_df = augmented_df.drop_duplicates(subset=['description', 'category'])
        
        print(f"Generated {len(augmented_df)} augmented samples")
        return augmented_df
    
    def _create_description_variations(self, description: str, category: str) -> List[str]:
        """Create variations of a transaction description."""
        variations = [description]  # Keep original
        
        # Common variations
        common_variations = [
            description.lower(),
            description.upper(),
            description.title(),
        ]
        
        variations.extend(common_variations)
        
        # Add merchant-specific variations
        if any(merchant.lower() in description.lower() for merchant in self.merchant_patterns.get(category, [])):
            # Already has merchant, create variations
            for pattern in ["at {}", "{} purchase", "payment to {}"]:
                variations.append(pattern.format(description))
        
        # Add amount mentions (sometimes descriptions include amounts)
        if random.random() > 0.7:
            amount = random.randint(5, 200)
            variations.append(f"{description} ${amount}")
            variations.append(f"${amount} {description}")
        
        # Add date variations
        if random.random() > 0.8:
            months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            month = random.choice(months)
            variations.append(f"{description} {month}")
            variations.append(f"{month} {description}")
        
        # Add payment method variations
        if random.random() > 0.6:
            payment_methods = ["Visa", "Mastercard", "PayPal", "Apple Pay", "Google Pay"]
            method = random.choice(payment_methods)
            variations.append(f"{description} via {method}")
        
        # Remove duplicates and limit variations
        unique_variations = list(set(variations))
        return unique_variations[:10]  # Return up to 10 variations
    
    def load_user_feedback_data(self, user_id: Optional[str] = None) -> pd.DataFrame:
        """
        Load user feedback data for retraining.
        
        Args:
            user_id: Specific user ID or None for all users
            
        Returns:
            DataFrame with user feedback data
        """
        feedback_path = self.data_dir / "feedback" / "user_feedback.jsonl"
        
        if not feedback_path.exists():
            print("No feedback data found")
            return pd.DataFrame()
        
        try:
            data = []
            with open(feedback_path, 'r') as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        
                        # Filter by user_id if specified
                        if user_id and record.get('user_id') != user_id:
                            continue
                        
                        # Only include corrections (where prediction was wrong)
                        if record.get('was_correct') == False:
                            data.append({
                                'description': record.get('description', ''),
                                'category': record.get('user_category', ''),
                                'user_id': record.get('user_id'),
                                'timestamp': record.get('timestamp')
                            })
            
            df = pd.DataFrame(data)
            print(f"Loaded {len(df)} feedback samples")
            return df
            
        except Exception as e:
            print(f"Error loading feedback data: {e}")
            return pd.DataFrame()
    
    def download_external_datasets(self, force_download: bool = False):
        """
        Download external datasets with proper authentication in headers.
        """
        # Dataset configurations
        datasets = {
            "kaggle_transactions": {
                "url": "https://www.kaggle.com/api/v1/datasets/download/eliasdabbas/personal-transaction-dataset",
                "local_path": self.data_dir / "kaggle" / "personal_transactions.zip",
                "requires_auth": True,
                "extract": True,
                "api_type": "kaggle"  # Add API type for different auth methods
            },
            "bank_transactions": {
                "url": "https://www.kaggle.com/api/v1/datasets/download/ealtman2019/ibm-transactions-for-anti-money-laundering-aml",
                "local_path": self.data_dir / "kaggle" / "bank_transactions.zip",
                "requires_auth": True,
                "extract": True,
                "api_type": "kaggle"
            },
            "financial_transactions": {
                "url": "https://raw.githubusercontent.com/nelgiriyewithana/global-financial-transaction-datasets/main/financial_transactions_dataset.csv",
                "local_path": self.data_dir / "open_data" / "financial_transactions.csv",
                "requires_auth": False,
                "extract": False
            }
        }
    
        print("Starting dataset downloads...")
    
        for name, dataset_info in datasets.items():
            local_path = dataset_info["local_path"]
            local_path.parent.mkdir(parents=True, exist_ok=True)
        
            if local_path.exists() and not force_download:
                print(f"✓ {name} already exists, skipping download")
                continue
        
        print(f"↓ Downloading {name}...")
        
        try:
            headers = {
                'User-Agent': 'FinanceAI-Dataset-Downloader/1.0',
            }
            
            # Handle different authentication methods
            if dataset_info.get("requires_auth"):
                        api_type = dataset_info.get("api_type", "basic")
                
                        if api_type == "kaggle":
                            # Kaggle uses Bearer token, not Basic Auth
                            if not self.kaggle_key:
                                print(f"  ⚠️ Skipping {name}: Kaggle API key required")
                    
                            # Correct Kaggle authentication
                            headers['Authorization'] = f'Bearer {self.kaggle_key}'
                    
                            # Alternative: Kaggle also accepts this header format
                            # headers['X-Kaggle-Api-Key'] = self.kaggle_key
                    
                        elif api_type == "github":
                            # GitHub uses token in Authorization header
                            headers['Authorization'] = f'token {self.github_token}'
                    
            else:
                # Basic auth for other APIs
                auth = (self.kaggle_username, self.kaggle_key)
            
            # Make the request
            if 'auth' in locals() and auth:
                response = requests.get(
                    dataset_info["url"], 
                    headers=headers,
                    auth=auth,
                    timeout=60,
                    stream=True
                )
            else:
                response = requests.get(
                    dataset_info["url"], 
                    headers=headers,
                    timeout=60,
                    stream=True
                )
            
            response.raise_for_status()
            
            # Check if response is HTML (common error page)
            content_type = response.headers.get('content-type', '')
            if 'text/html' in content_type and response.status_code == 200:
                # This might be an error page disguised as 200
                content_sample = response.content[:200]
                if b'<html' in content_sample.lower():
                    print(f"  ⚠️ Received HTML instead of dataset - authentication may have failed")
                    # Try alternative Kaggle auth method
                    if dataset_info.get("api_type") == "kaggle":
                        self._try_alternative_kaggle_auth(name, dataset_info, local_path)
            
            
            # Save the file
            self._save_downloaded_file(response, local_path, dataset_info.get("extract", False))
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                print(f"  ❌ Authentication failed: Check your Kaggle API key")
                print(f"     Make sure you're using an API key, not your password")
            elif e.response.status_code == 403:
                print(f"  ❌ Access forbidden: You may not have access to this dataset")
            elif e.response.status_code == 404:
                print(f"  ❌ Dataset not found: Check the dataset URL")
            else:
                print(f"  ❌ HTTP {e.response.status_code}: {e}")
        except Exception as e:
            print(f"  ❌ Failed to download {name}: {str(e)}")
    

def _try_alternative_kaggle_auth(self, name: str, dataset_info: dict, local_path: Path):
    """Try alternative authentication methods for Kaggle."""
    print(f"  Trying alternative authentication for {name}...")
    
    # Method 2: X-Kaggle-Api-Key header
    headers = {
        'User-Agent': 'FinanceAI-Dataset-Downloader/1.0',
        'X-Kaggle-Api-Key': self.kaggle_key
    }
    
    try:
        response = requests.get(
            dataset_info["url"],
            headers=headers,
            timeout=60,
            stream=True
        )
        response.raise_for_status()
        
        # Check if we got actual data
        content_type = response.headers.get('content-type', '')
        if 'application/zip' in content_type or 'application/octet-stream' in content_type:
            self._save_downloaded_file(response, local_path, dataset_info.get("extract", False))
            print(f"  ✓ Successfully downloaded with X-Kaggle-Api-Key header")
            return True
            
    except Exception as e:
        print(f"  ❌ Alternative authentication also failed")
    
    return False

def _save_downloaded_file(self, response, local_path: Path, extract: bool):
    """Save downloaded file with progress."""
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(local_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    if int(percent) % 10 == 0:
                        print(f"  Progress: {percent:.1f}%", end='\r')
    
    print(f"  ✓ Downloaded ({downloaded / 1024 / 1024:.1f} MB)")
    
    if extract:
        self._extract_zip_file(local_path, local_path.parent)  
    
    def create_train_test_split(self, df: pd.DataFrame, 
                               test_size: float = 0.2,
                               stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create train/test split from dataframe.
        
        Args:
            df: Input dataframe
            test_size: Proportion for test set
            stratify: Whether to stratify by category
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        if stratify and 'category' in df.columns:
            from sklearn.model_selection import train_test_split
            
            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=42,
                stratify=df['category']
            )
        else:
            # Simple random split
            test_mask = np.random.rand(len(df)) < test_size
            train_df = df[~test_mask]
            test_df = df[test_mask]
        
        print(f"Train: {len(train_df)} samples, Test: {len(test_df)} samples")
        return train_df, test_df
    
    def save_dataset(self, df: pd.DataFrame, filename: str):
        """
        Save dataset to file.
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        output_path = self.data_dir / filename
        
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.csv':
            df.to_csv(output_path, index=False)
        elif output_path.suffix == '.json':
            df.to_json(output_path, orient='records', indent=2)
        elif output_path.suffix == '.parquet':
            df.to_parquet(output_path, index=False)
        else:
            # Default to CSV
            df.to_csv(output_path.with_suffix('.csv'), index=False)
        
        print(f"Dataset saved to {output_path}")
    
    def get_dataset_stats(self, df: pd.DataFrame) -> Dict:
        """
        Get statistics about the dataset.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary with dataset statistics
        """
        if df.empty:
            return {"message": "Empty dataset"}
        
        stats = {
            "total_samples": len(df),
            "unique_categories": df['category'].nunique() if 'category' in df.columns else 0,
            "categories_distribution": {},
            "description_length": {
                "mean": df['description'].str.len().mean() if 'description' in df.columns else 0,
                "min": df['description'].str.len().min() if 'description' in df.columns else 0,
                "max": df['description'].str.len().max() if 'description' in df.columns else 0,
            }
        }
        
        if 'category' in df.columns:
            category_counts = df['category'].value_counts().to_dict()
            stats["categories_distribution"] = category_counts
        
        if 'amount' in df.columns:
            stats["amount_stats"] = {
                "mean": df['amount'].mean(),
                "median": df['amount'].median(),
                "min": df['amount'].min(),
                "max": df['amount'].max(),
                "std": df['amount'].std()
            }
        
        return stats