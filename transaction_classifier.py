import os
import joblib
import warnings
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from transaction import TransactionCategory

warnings.filterwarnings("ignore")


class TransactionClassifier:
    
    def __init__(self, model_path: Optional[str] = None):
        # ---------------- Feature extraction ----------------
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            min_df=2,
            max_df=0.95,
            stop_words="english",
            ngram_range=(1, 3),
        )

        # ---------------- Primary classifier ----------------
        base_clf = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-3,
            max_iter=1000,
            random_state=42,
        )

        self.classifier = CalibratedClassifierCV(base_clf, cv=5)

        # ---------------- Fallback classifier ----------------
        self.fallback_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )

        # ---------------- State ----------------
        self.category_to_index: Dict[str, int] = {}
        self.index_to_category: Dict[int, str] = {}
        self.training_history: List[Dict] = []
        self.last_trained: Optional[datetime] = None

        self.confidence_thresholds = {
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4,
        }

        self._trained = False

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            
            
        # Feedback buffer for online learning
        self.feedback_buffer = []
        self.feedback_buffer_max_size = 100    

    # ======================================================
    # Utility
    # ======================================================

    def is_trained(self) -> bool:
        return self._trained

    def _confidence_level(self, prob: float) -> str:
        if prob >= self.confidence_thresholds["high"]:
            return "high"
        if prob >= self.confidence_thresholds["medium"]:
            return "medium"
        if prob >= self.confidence_thresholds["low"]:
            return "low"
        return "very_low"

    # ======================================================
    # Prediction
    # ======================================================

    def predict(self, description: str, return_all: bool = False):
        if not self.is_trained():
            default_category = TransactionCategory.OTHER.value
            if return_all:
                return {
                    "predicted_category": default_category,
                    "confidence": 0.0,
                    "confidence_level": "unknown",
                    "is_fallback": True,
                }
            return default_category

        X = self.vectorizer.transform([description])

        probabilities = self.classifier.predict_proba(X)[0]
        idx = int(np.argmax(probabilities))
        prob = float(probabilities[idx])

        predicted_category = self.index_to_category[idx]
        confidence_level = self._confidence_level(prob)

        is_fallback = False

        if confidence_level in {"low", "very_low"}:
            fallback_idx = self.fallback_classifier.predict(X)[0]
            fallback_category = self.index_to_category.get(
                int(fallback_idx), TransactionCategory.OTHER.value
            )

            if fallback_category != predicted_category:
                predicted_category = fallback_category
                is_fallback = True

        if return_all:
            return {
                "predicted_category": predicted_category,
                "confidence": prob,
                "confidence_level": confidence_level,
                "is_fallback": is_fallback,
            }

        return predicted_category

    def predict_batch(self, descriptions: List[str]) -> List[Dict]:
        if not descriptions:
            return []

        if not self.is_trained():
            return [
                {
                    "description": d,
                    "predicted_category": self._rule_based_prediction(d),
                    "confidence": 0.5,
                    "confidence_level": "low",
                }
                for d in descriptions
            ]

        X = self.vectorizer.transform(descriptions)
        probabilities = self.classifier.predict_proba(X)

        results = []
        for desc, probs in zip(descriptions, probabilities):
            idx = int(np.argmax(probs))
            prob = float(probs[idx])

            results.append(
                {
                    "description": desc,
                    "predicted_category": self.index_to_category[idx],
                    "confidence": prob,
                    "confidence_level": self._confidence_level(prob),
                }
            )

        return results

    # ======================================================
    # Training
    # ======================================================

    def train(
        self,
        descriptions: List[str],
        categories: List[str],
        validation_split: float = 0.2,
    ) -> Dict:
        if not descriptions or not categories:
            raise ValueError("Training data cannot be empty")

        if len(descriptions) != len(categories):
            raise ValueError("Descriptions and categories must match")

        unique_categories = sorted(set(categories))
        self.category_to_index = {
            cat: i for i, cat in enumerate(unique_categories)
        }
        self.index_to_category = {
            i: cat for cat, i in self.category_to_index.items()
        }

        y = np.array([self.category_to_index[c] for c in categories])

        X_train, X_val, y_train, y_val = train_test_split(
            descriptions,
            y,
            test_size=validation_split,
            random_state=42,
            stratify=y,
        )

        X_train_vec = self.vectorizer.fit_transform(X_train)

        self.classifier.fit(X_train_vec, y_train)
        self.fallback_classifier.fit(X_train_vec, y_train)

        metrics = {}

        if len(X_val) > 0:
            X_val_vec = self.vectorizer.transform(X_val)
            y_pred = self.classifier.predict(X_val_vec)

            metrics = {
                "accuracy": accuracy_score(y_val, y_pred),
                "report": classification_report(
                    y_val, y_pred, output_dict=True
                ),
            }

        self.training_history.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "samples": len(descriptions),
                "metrics": metrics,
            }
        )

        self.last_trained = datetime.utcnow()
        self._trained = True

        return metrics
    
    # ======================================================
    # Persistence
    # ======================================================

    def save_model(self, filepath: str):
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        joblib.dump(
            {
                "classifier": self.classifier,
                "fallback_classifier": self.fallback_classifier,
                "vectorizer": self.vectorizer,
                "category_to_index": self.category_to_index,
                "index_to_category": self.index_to_category,
                "training_history": self.training_history,
                "last_trained": self.last_trained,
            },
            filepath,
        )

    def load_model(self, filepath: str):
        data = joblib.load(filepath)

        self.classifier = data["classifier"]
        self.fallback_classifier = data["fallback_classifier"]
        self.vectorizer = data["vectorizer"]
        self.category_to_index = data["category_to_index"]
        self.index_to_category = data["index_to_category"]
        self.training_history = data["training_history"]
        self.last_trained = data["last_trained"]

        self._trained = True

    # ======================================================
    # Rule-based fallback
    # ======================================================

    def _rule_based_prediction(self, description: str) -> str:
        desc = description.lower()

        rules = {
            TransactionCategory.SALARY.value: ["salary", "payroll"],
            TransactionCategory.GROCERIES.value: ["grocery", "market"],
            TransactionCategory.DINING.value: ["restaurant", "coffee"],
            TransactionCategory.TRANSPORTATION.value: ["uber", "taxi"],
            TransactionCategory.SUBSCRIPTIONS.value: ["netflix", "spotify"],
        }

        for category, keywords in rules.items():
            if any(k in desc for k in keywords):
                return category

        return TransactionCategory.OTHER.value
    
    
    def get_model_info(self) -> Dict:
        """Get information about the current model state."""
        return {
            'is_trained': self.is_trained(),
            'last_trained': self.last_trained.isoformat() if self.last_trained else None,
            'num_categories': len(self.category_to_index),
            'categories': list(self.category_to_index.keys()),
            'training_samples': sum(record.get('training_samples', 0) for record in self.training_history),
            'feedback_buffer_size': len(self.feedback_buffer),
            'confidence_thresholds': self.confidence_thresholds
        }
