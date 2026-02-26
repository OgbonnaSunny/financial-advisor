# ai-service/data/dataset_generator.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
import random

from transaction import TransactionCategory
from dataset_loader import DatasetLoader

class DatasetGenerator:
    """Generate comprehensive transaction datasets."""
    
    def __init__(self):
        self.loader = DatasetLoader()
        self.transaction_templates = self._load_transaction_templates()
    
    def _load_transaction_templates(self) -> Dict:
        """Load transaction templates for different user personas."""
        return {
            "student": {
                "common_categories": [
                    TransactionCategory.GROCERIES.value,
                    TransactionCategory.DINING.value,
                    TransactionCategory.ENTERTAINMENT.value,
                    TransactionCategory.SUBSCRIPTIONS.value,
                    TransactionCategory.TRANSPORTATION.value,
                    TransactionCategory.EDUCATION.value
                ],
                "income_sources": [
                    TransactionCategory.FREELANCE.value,
                    TransactionCategory.GIFT.value,
                    TransactionCategory.OTHER_INCOME.value
                ],
                "amount_multiplier": 0.5  # Students spend less
            },
            "professional": {
                "common_categories": [
                    TransactionCategory.GROCERIES.value,
                    TransactionCategory.DINING.value,
                    TransactionCategory.SHOPPING.value,
                    TransactionCategory.ENTERTAINMENT.value,
                    TransactionCategory.TRANSPORTATION.value,
                    TransactionCategory.HOUSING.value,
                    TransactionCategory.UTILITIES.value
                ],
                "income_sources": [
                    TransactionCategory.SALARY.value,
                    TransactionCategory.INVESTMENT.value,
                    TransactionCategory.FREELANCE.value
                ],
                "amount_multiplier": 1.2  # Professionals spend more
            },
            "family": {
                "common_categories": [
                    TransactionCategory.GROCERIES.value,
                    TransactionCategory.HOUSING.value,
                    TransactionCategory.UTILITIES.value,
                    TransactionCategory.HEALTHCARE.value,
                    TransactionCategory.KIDS.value,
                    TransactionCategory.EDUCATION.value,
                    TransactionCategory.SHOPPING.value
                ],
                "income_sources": [
                    TransactionCategory.SALARY.value,
                    TransactionCategory.INVESTMENT.value
                ],
                "amount_multiplier": 1.5  # Families have higher expenses
            },
            "retiree": {
                "common_categories": [
                    TransactionCategory.GROCERIES.value,
                    TransactionCategory.HEALTHCARE.value,
                    TransactionCategory.ENTERTAINMENT.value,
                    TransactionCategory.TRAVEL.value,
                    TransactionCategory.UTILITIES.value,
                    TransactionCategory.HOUSING.value
                ],
                "income_sources": [
                    TransactionCategory.INVESTMENT.value,
                    TransactionCategory.SALARY.value,  # Pension
                    TransactionCategory.OTHER_INCOME.value
                ],
                "amount_multiplier": 0.8  # Retirees may spend less
            }
        }
    
    def generate_user_dataset(self, user_persona: str = "professional", 
                            num_transactions: int = 500,
                            time_range_days: int = 365) -> pd.DataFrame:
        """
        Generate realistic transaction dataset for a specific user persona.
        
        Args:
            user_persona: Type of user ('student', 'professional', 'family', 'retiree')
            num_transactions: Number of transactions to generate
            time_range_days: Time range in days
            
        Returns:
            DataFrame with user's transaction history
        """
        if user_persona not in self.transaction_templates:
            user_persona = "professional"
        
        template = self.transaction_templates[user_persona]
        multiplier = template["amount_multiplier"]
        
        transactions = []
        start_date = datetime.now() - timedelta(days=time_range_days)
        
        # Generate income transactions (approximately 10% of total)
        num_income = max(1, num_transactions // 10)
        for i in range(num_income):
            category = random.choice(template["income_sources"])
            amount = self._generate_income_amount(category, multiplier)
            description = self._generate_income_description(category)
            
            transactions.append({
                'description': description,
                'category': category,
                'amount': amount,
                'date': self._random_date(start_date),
                'transaction_type': 'income',
                'user_persona': user_persona
            })
        
        # Generate expense transactions
        num_expenses = num_transactions - num_income
        for i in range(num_expenses):
            # Weighted category selection (some categories are more frequent)
            category = self._weighted_category_selection(template["common_categories"])
            amount = self._generate_expense_amount(category, multiplier)
            description = self._generate_expense_description(category)
            
            transactions.append({
                'description': description,
                'category': category,
                'amount': amount,
                'date': self._random_date(start_date),
                'transaction_type': 'expense',
                'user_persona': user_persona
            })
        
        df = pd.DataFrame(transactions)
        
        # Sort by date
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Add some recurring transactions
        df = self._add_recurring_transactions(df)
        
        print(f"Generated {len(df)} transactions for {user_persona} persona")
        return df
    
    def _weighted_category_selection(self, categories: List[str]) -> str:
        """Select category with weights (some are more common)."""
        weights = {
            TransactionCategory.GROCERIES.value: 0.25,
            TransactionCategory.DINING.value: 0.15,
            TransactionCategory.SHOPPING.value: 0.10,
            TransactionCategory.TRANSPORTATION.value: 0.08,
            TransactionCategory.ENTERTAINMENT.value: 0.07,
            TransactionCategory.HOUSING.value: 0.08,
            TransactionCategory.UTILITIES.value: 0.07,
            TransactionCategory.SUBSCRIPTIONS.value: 0.05,
            TransactionCategory.HEALTHCARE.value: 0.04,
            TransactionCategory.OTHER.value: 0.11
        }
        
        # Use provided weights or default
        weighted_categories = []
        weighted_weights = []
        
        for cat in categories:
            weighted_categories.append(cat)
            weighted_weights.append(weights.get(cat, 0.05))
        
        # Normalize weights
        total_weight = sum(weighted_weights)
        normalized_weights = [w / total_weight for w in weighted_weights]
        
        return random.choices(weighted_categories, weights=normalized_weights)[0]
    
    def _generate_income_amount(self, category: str, multiplier: float) -> float:
        """Generate income amount based on category."""
        base_amounts = {
            TransactionCategory.SALARY.value: (2000, 10000, 4500),
            TransactionCategory.FREELANCE.value: (200, 5000, 1200),
            TransactionCategory.INVESTMENT.value: (100, 5000, 800),
            TransactionCategory.GIFT.value: (20, 1000, 150)
        }
        
        if category in base_amounts:
            min_amt, max_amt, typical = base_amounts[category]
        else:
            min_amt, max_amt, typical = (100, 2000, 500)
        
        # Apply persona multiplier
        amount = np.random.normal(typical * multiplier, (max_amt - min_amt) / 6)
        amount = max(min_amt, min(max_amt, amount))
        
        return round(amount, 2)
    
    def _generate_expense_amount(self, category: str, multiplier: float) -> float:
        """Generate expense amount based on category."""
        # Get base amount ranges from loader
        for enum_category in TransactionCategory:
            if enum_category.value == category:
                if enum_category in self.loader.amount_ranges:
                    min_amt, max_amt, typical = self.loader.amount_ranges[enum_category]
                    break
        else:
            min_amt, max_amt, typical = (5, 500, 50)
        
        # Apply persona multiplier
        amount = np.random.normal(typical * multiplier, (max_amt - min_amt) / 4)
        amount = max(min_amt, min(max_amt, amount))
        
        return round(amount, 2)
    
    def _generate_income_description(self, category: str) -> str:
        """Generate income transaction description."""
        descriptions = {
            TransactionCategory.SALARY.value: [
                "Salary Deposit - {company}",
                "Monthly Paycheck",
                "Direct Deposit Salary",
                "Payroll Payment"
            ],
            TransactionCategory.FREELANCE.value: [
                "Freelance Payment - Project #{num}",
                "Consulting Services Invoice",
                "Contract Work Payment",
                "Client Payment - {client}"
            ],
            TransactionCategory.INVESTMENT.value: [
                "Dividend Payment",
                "Stock Sale Proceeds",
                "Investment Return",
                "Capital Gains Distribution"
            ],
            TransactionCategory.GIFT.value: [
                "Gift from {name}",
                "Birthday Gift",
                "Holiday Present",
                "Cash Gift"
            ]
        }
        
        if category in descriptions:
            template = random.choice(descriptions[category])
            
            if '{company}' in template:
                companies = ["Acme Inc", "Tech Solutions", "Global Corp", "Innovation Ltd"]
                return template.format(company=random.choice(companies))
            elif '{client}' in template:
                clients = ["Client A", "Client B", "XYZ Corp", "ABC Company"]
                return template.format(client=random.choice(clients))
            elif '{name}' in template:
                names = ["John", "Sarah", "Mike", "Emily", "Parents", "Family"]
                return template.format(name=random.choice(names))
            elif '{num}' in template:
                return template.format(num=random.randint(100, 999))
            else:
                return template
        
        return "Income Transaction"
    
    def _generate_expense_description(self, category: str) -> str:
        """Generate expense transaction description."""
        return self.loader._generate_description(
            next(c for c in TransactionCategory if c.value == category)
        )
    
    def _random_date(self, start_date: datetime) -> datetime:
        """Generate random date within range."""
        end_date = datetime.now()
        time_between = end_date - start_date
        random_days = random.randrange(time_between.days)
        return start_date + timedelta(days=random_days)
    
    def _add_recurring_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add recurring transactions to the dataset."""
        recurring_templates = [
            {
                'description': 'Netflix Monthly Subscription',
                'category': TransactionCategory.SUBSCRIPTIONS.value,
                'amount': 15.99,
                'frequency_days': 30
            },
            {
                'description': 'Spotify Premium',
                'category': TransactionCategory.SUBSCRIPTIONS.value,
                'amount': 9.99,
                'frequency_days': 30
            },
            {
                'description': 'Gym Membership',
                'category': TransactionCategory.ENTERTAINMENT.value,
                'amount': 49.99,
                'frequency_days': 30
            },
            {
                'description': 'Rent Payment',
                'category': TransactionCategory.HOUSING.value,
                'amount': 1200.00,
                'frequency_days': 30
            },
            {
                'description': 'Electric Bill',
                'category': TransactionCategory.UTILITIES.value,
                'amount': 85.50,
                'frequency_days': 30
            }
        ]
        
        recurring_transactions = []
        
        for template in recurring_templates:
            # Add 3-6 occurrences of each recurring transaction
            num_occurrences = random.randint(3, 6)
            start_date = pd.to_datetime(df['date']).min()
            
            for i in range(num_occurrences):
                transaction_date = start_date + timedelta(days=template['frequency_days'] * i)
                
                # Add small variations to make it realistic
                amount_variation = template['amount'] * random.uniform(0.95, 1.05)
                description_variation = template['description']
                
                if random.random() > 0.7:
                    description_variation = f"{template['description']} - {transaction_date.strftime('%b %Y')}"
                
                recurring_transactions.append({
                    'description': description_variation,
                    'category': template['category'],
                    'amount': round(amount_variation, 2),
                    'date': transaction_date,
                    'transaction_type': 'expense',
                    'is_recurring': True
                })
        
        # Combine with original transactions
        if recurring_transactions:
            recurring_df = pd.DataFrame(recurring_transactions)
            df = pd.concat([df, recurring_df], ignore_index=True)
            df = df.sort_values('date')
        
        return df
    
    def generate_multi_user_dataset(self, num_users: int = 10, 
                                    transactions_per_user: int = 200) -> pd.DataFrame:
        """
        Generate dataset with multiple user personas.
        
        Args:
            num_users: Number of users to generate
            transactions_per_user: Transactions per user
            
        Returns:
            Combined dataset for all users
        """
        all_data = []
        personas = list(self.transaction_templates.keys())
        
        for user_id in range(1, num_users + 1):
            persona = random.choice(personas)
            user_data = self.generate_user_dataset(
                user_persona=persona,
                num_transactions=transactions_per_user
            )
            
            # Add user_id
            user_data['user_id'] = f"user_{user_id:03d}"
            user_data['user_persona'] = persona
            
            all_data.append(user_data)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Generated dataset with {num_users} users, {len(combined_df)} total transactions")
        
        return combined_df