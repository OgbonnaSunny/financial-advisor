# ai-service/app/models/batch.py
from typing import List, Dict
from pydantic import BaseModel
from transaction import Transaction

class BatchTransactionRequest(BaseModel):
    """Model for batch transaction processing"""
    transactions: List[Transaction]
    user_id: str
    auto_categorize: bool = True
    detect_patterns: bool = True
    generate_insights: bool = False

class BatchTransactionResponse(BaseModel):
    """Response for batch processing"""
    processed_count: int
    categorized_count: int
    insights: List[Dict]
    anomalies: List[Dict]
    summary: Dict