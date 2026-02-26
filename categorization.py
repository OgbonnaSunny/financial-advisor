# ai-service/app/models/categorization.py
from typing import Optional, List, Dict
from transaction import TransactionType, TransactionCategory
from pydantic import BaseModel, Field

class CategorizationRequest(BaseModel):
    """Request for transaction categorization"""
    description: str
    amount: float
    merchant: Optional[str] = None
    user_id: str
    transaction_type: TransactionType
    historical_context: Optional[List[dict]] = None

class CategorizationResponse(BaseModel):
    """Response from categorization model"""
    predicted_category: TransactionCategory
    confidence: float = Field(..., ge=0, le=1)
    alternative_categories: List[Dict[str, float]] = []
    reasoning: Optional[str] = None
    suggested_tags: List[str] = []
    is_duplicate: bool = False
    similar_transactions: List[dict] = []