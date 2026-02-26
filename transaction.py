# ai-service/app/models/transaction.py
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any  # Note: 'Any' not 'any'
from enum import Enum

class TransactionType(str, Enum):
    INCOME = "income"
    EXPENSE = "expense"
    TRANSFER = "transfer"

class TransactionCategory(str, Enum):
    # Income categories
    SALARY = "salary"
    FREELANCE = "freelance"
    INVESTMENT = "investment"
    GIFT = "gift"
    REFUND = "refund"
    OTHER_INCOME = "other_income"
    
    # Expense categories
    HOUSING = "housing"
    UTILITIES = "utilities"
    GROCERIES = "groceries"
    DINING = "dining"
    TRANSPORTATION = "transportation"
    CAR = "car"
    HEALTHCARE = "healthcare"
    INSURANCE = "insurance"
    ENTERTAINMENT = "entertainment"
    SHOPPING = "shopping"
    EDUCATION = "education"
    TRAVEL = "travel"
    SUBSCRIPTIONS = "subscriptions"
    DEBT = "debt"
    SAVINGS = "savings"
    INVESTMENTS = "investments"
    DONATIONS = "donations"
    PERSONAL_CARE = "personal_care"
    PETS = "pets"
    KIDS = "kids"
    BUSINESS = "business"
    TAXES = "taxes"
    OTHER = "other"

class Transaction(BaseModel):
    """Transaction model for AI processing"""
    id: Optional[str] = None
    user_id: str
    description: str = Field(..., min_length=1, max_length=500)
    amount: float = Field(..., gt=0)
    currency: str = "USD"
    transaction_type: TransactionType
    predicted_category: Optional[TransactionCategory] = None
    user_category: Optional[TransactionCategory] = None  # User can override
    confidence_score: Optional[float] = Field(default=None, ge=0, le=1)
    date: datetime = Field(default_factory=datetime.now)  # ✅ Fixed
    merchant: Optional[str] = None
    location: Optional[str] = None
    payment_method: Optional[str] = None
    is_recurring: bool = False
    recurrence_pattern: Optional[str] = None  # "monthly", "weekly", etc.
    tags: List[str] = Field(default_factory=list)  # ✅ Better than []
    notes: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)  # ✅ Fixed (Any, not any)
    is_manual_entry: bool = False
    bank_account_id: Optional[str] = None
    needs_review: bool = False
    created_at: datetime = Field(default_factory=datetime.now)  # ✅ Fixed
    updated_at: datetime = Field(default_factory=datetime.now)  # ✅ Fixed

    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_123",
                "description": "Starbucks Coffee",
                "amount": 5.75,
                "transaction_type": "expense",
                "merchant": "Starbucks",
                "location": "New York, NY",
                "payment_method": "credit_card",
                "tags": ["coffee", "food"],
                "notes": "Morning coffee"
            }
        }