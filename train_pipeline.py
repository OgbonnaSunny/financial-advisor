# ai-service/training/training_pipeline.py
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# MLflow for experiment tracking
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Local imports
from transaction_classifier import TransactionClassifier
from dataset_loader import DatasetLoader
from transaction import TransactionCategory

class TrainingPipeline:
    """
    Complete training pipeline with full training, incremental retraining,
    and feedback-based model improvement.
    """
    
    def __init__(self, model_dir: str = "models", data_dir: str = "data"):
        """
        Initialize the training pipeline.
        
        Args:
            model_dir: Directory for storing models and logs
            data_dir: Directory for datasets
        """
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        
        # Create directories if they don't exist
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Model paths
        self.model_path = self.model_dir / "transaction_classifier.pkl"
        self.backup_model_path = self.model_dir / "transaction_classifier_backup.pkl"
        
        # Feedback and log paths
        self.feedback_log_path = self.model_dir / "feedback_log.jsonl"
        self.training_log_path = self.model_dir / "training_log.jsonl"
        self.feedback_buffer_path = self.model_dir / "feedback_buffer.jsonl"
        
        # Dataset loader
        self.dataset_loader = DatasetLoader(data_dir=str(self.data_dir))
    
        
        # Initialize MLflow
        mlflow_db_path = self.model_dir / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
        mlflow.set_experiment("transaction-classification")
        
        # Training configuration
        self.config = {
            "validation_split": 0.2,
            "min_feedback_for_retraining": 50,
            "stale_model_days": 30,
            "backup_before_retrain": True,
            "max_training_samples": 50000,
            "batch_size": 1000,
            "feedback_buffer_max_size": 200,
            "auto_save_buffer_interval": 10,  # Auto-save after N additions
        }
        
        # Initialize feedback buffer
        self.feedback_buffer = []
        self.buffer_additions_since_save = 0
        
        # Load or create model
        self.model = self._load_or_create_model()
        
        # Load existing feedback buffer if it exists
        self._load_feedback_buffer()
    
    def _add_to_feedback_buffer(self, descriptions: list, categories: list):
        """
        Add feedback samples to the in-memory buffer for incremental learning.
        
        Args:
            descriptions: List of transaction descriptions
            categories: List of corresponding correct categories
        """
        if not descriptions or not categories:
            return
        
        if len(descriptions) != len(categories):
            print("⚠️ Warning: Descriptions and categories lists must have same length")
            return
        
        # Add each feedback sample to buffer
        for desc, cat in zip(descriptions, categories):
            buffer_entry = {
                "description": desc,
                "category": cat,
                "timestamp": datetime.now().isoformat(),
                "source": "user_feedback"
            }
            
            # Check if similar entry already exists in buffer
            if not self._is_duplicate_in_buffer(desc, cat):
                self.feedback_buffer.append(buffer_entry)
                
                # Keep buffer size manageable
                if len(self.feedback_buffer) > self.config["feedback_buffer_max_size"]:
                    # Remove oldest entries
                    self.feedback_buffer = self.feedback_buffer[-self.config["feedback_buffer_max_size"]:]
        
        self.buffer_additions_since_save += 1
        
        # Auto-save buffer to disk periodically
        if self.buffer_additions_since_save >= self.config["auto_save_buffer_interval"]:
            self._save_feedback_buffer()
            self.buffer_additions_since_save = 0
        
        print(f"📝 Added {len(descriptions)} samples to feedback buffer (total: {len(self.feedback_buffer)})")
    
    def _is_duplicate_in_buffer(self, description: str, category: str) -> bool:
        """
        Check if similar feedback already exists in buffer.
        
        Args:
            description: Transaction description
            category: Category
            
        Returns:
            True if duplicate exists
        """
        # Simple duplicate check
        for entry in self.feedback_buffer[-20:]:  # Check recent entries
            if (entry["description"] == description and 
                entry["category"] == category):
                return True
        
        # Also check for similar descriptions (fuzzy matching)
        # Simple implementation - can be enhanced with NLP
        desc_lower = description.lower()
        for entry in self.feedback_buffer[-50:]:
            entry_desc_lower = entry["description"].lower()
            
            # Check if descriptions are very similar
            if (desc_lower in entry_desc_lower or 
                entry_desc_lower in desc_lower or
                self._jaccard_similarity(desc_lower, entry_desc_lower) > 0.8):
                
                # If categories also match, it's a duplicate
                if entry["category"] == category:
                    return True
        
        return False
    
    def _jaccard_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate Jaccard similarity between two strings.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score between 0 and 1
        """
        set1 = set(str1.split())
        set2 = set(str2.split())
        
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0
    
    def _save_feedback_buffer(self):
        """Save feedback buffer to disk."""
        try:
            with open(self.feedback_buffer_path, 'w', encoding='utf-8') as f:
                for entry in self.feedback_buffer:
                    f.write(json.dumps(entry) + '\n')
            
            # print(f"💾 Feedback buffer saved ({len(self.feedback_buffer)} entries)")
        except Exception as e:
            print(f"❌ Failed to save feedback buffer: {e}")
    
    def _load_feedback_buffer(self):
        """Load feedback buffer from disk."""
        if not self.feedback_buffer_path.exists():
            self.feedback_buffer = []
            return
        
        try:
            self.feedback_buffer = []
            with open(self.feedback_buffer_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        self.feedback_buffer.append(entry)
            
            print(f"📁 Loaded {len(self.feedback_buffer)} entries from feedback buffer")
        except Exception as e:
            print(f"❌ Failed to load feedback buffer: {e}")
            self.feedback_buffer = []
    
    def _load_or_create_model(self) -> TransactionClassifier:
        """Load existing model or create new one."""
        if self.model_path.exists():
            print("Loading existing model...")
            try:
                model = TransactionClassifier(str(self.model_path))
                
                # Check if model is stale
                if model.last_trained:
                    stale_threshold = datetime.now() - timedelta(days=self.config["stale_model_days"])
                    if model.last_trained < stale_threshold:
                        print(f"⚠️ Model is stale (last trained: {model.last_trained}). Consider retraining.")
                
                # Load model info
                model_info = model.get_model_info()
                print(f"✅ Model loaded successfully:")
                print(f"   - Trained: {model_info['is_trained']}")
                print(f"   - Last trained: {model_info['last_trained']}")
                print(f"   - Categories: {model_info['num_categories']}")
                print(f"   - Feedback buffer: {model_info['feedback_buffer_size']}")
                
                return model
                
            except Exception as e:
                print(f"❌ Failed to load model: {e}")
                print("Creating new model...")
                return TransactionClassifier()
        else:
            print("No existing model found. Creating new model...")
            return TransactionClassifier()
    
    def run_full_training(self, dataset_path: str = None, save_model: bool = True):
        """
        Run full training pipeline from scratch.
        
        Args:
            dataset_path: Path to training dataset (optional)
            save_model: Whether to save the trained model
            
        Returns:
            Dictionary with training metrics
        """
        print("=" * 60)
        print("Starting full training pipeline")
        print("=" * 60)
        
        with mlflow.start_run(run_name="full_training"):
            # Load training data
            if dataset_path:
                print(f"Loading dataset from: {dataset_path}")
                if Path(dataset_path).exists():
                    data = pd.read_csv(dataset_path)
                else:
                    print(f"Dataset path not found. Loading default data...")
                    data = self.dataset_loader.load_training_data(
                        source='all', 
                        augment=True,
                        sample_size=self.config["max_training_samples"]
                    )
            else:
                print("Loading default training data...")
                data = self.dataset_loader.load_training_data(
                    source='all', 
                    augment=True,
                    sample_size=self.config["max_training_samples"]
                )
            
            print(f"✅ Loaded {len(data)} training samples")
            
            # Check if we have enough data
            if len(data) < 100:
                print(f"⚠️ Warning: Only {len(data)} samples available. Consider adding more data.")
            
            # Get dataset stats
            stats = self.dataset_loader.get_dataset_stats(data)
            print(f"📊 Dataset statistics:")
            print(f"   - Unique categories: {stats['unique_categories']}")
            print(f"   - Description length: avg={stats['description_length']['mean']:.1f} chars")
            if 'amount_stats' in stats:
                print(f"   - Amount stats: avg=${stats['amount_stats']['mean']:.2f}")
            
            # Preprocess data
            descriptions = data['description'].astype(str).tolist()
            categories = data['category'].astype(str).tolist()
            
            # Log parameters to MLflow
            mlflow.log_params({
                "training_samples": len(descriptions),
                "unique_categories": len(set(categories)),
                "model_type": "SGDClassifier + RandomForest",
                "vectorizer": "TF-IDF",
                "ngram_range": "1-3",
                "max_features": 2000,
                "validation_split": self.config["validation_split"]
            })
            
            # Train model
            print("\n🏋️ Training model...")
            start_time = datetime.now()
            
            try:
                metrics = self.model.train(
                    descriptions, 
                    categories, 
                    validation_split=self.config["validation_split"],
                    retrain=False
                )
                
                training_time = (datetime.now() - start_time).total_seconds()
                print(f"✅ Training completed in {training_time:.2f} seconds")
                
                # Log metrics
                if metrics:
                    mlflow.log_metric("accuracy", metrics.get('accuracy', 0))
                    mlflow.log_metric("training_time", training_time)
                    mlflow.log_metric("training_samples", metrics.get('training_size', 0))
                    mlflow.log_metric("validation_samples", metrics.get('validation_size', 0))
                    
                    # Log classification report
                    if 'report' in metrics:
                        mlflow.log_dict(metrics['report'], "classification_report.json")
                        print(f"📈 Validation accuracy: {metrics.get('accuracy', 0):.3f}")
                
                # Save model
                if save_model:
                    self._backup_current_model()
                    self.model.save_model(str(self.model_path))
                    print(f"💾 Model saved to: {self.model_path}")
                    
                    # Log model to MLflow
                    mlflow.sklearn.log_model(
                        self.model.classifier, 
                        "transaction_classifier",
                        registered_model_name="TransactionClassifier"
                    )
                
                # Log training run
                self._log_training_run("full", len(descriptions), metrics)
                
                print("\n" + "=" * 60)
                print("Full training completed successfully!")
                print("=" * 60)
                
                return metrics
                
            except Exception as e:
                print(f"❌ Training failed: {e}")
                # Restore backup if training failed
                self._restore_backup_model()
                raise
    
    def run_incremental_training(self, feedback_data: pd.DataFrame = None, 
                                trigger_threshold: int = None,
                                use_buffer: bool = True):
        """
        Run incremental training with feedback data.
        
        Args:
            feedback_data: New feedback data for retraining (optional)
            trigger_threshold: Minimum samples to trigger retraining
            use_buffer: Whether to use the feedback buffer
            
        Returns:
            Dictionary with retraining metrics
        """
        print("=" * 60)
        print("Starting incremental training")
        print("=" * 60)
        
        # Use configured threshold if not specified
        if trigger_threshold is None:
            trigger_threshold = self.config["min_feedback_for_retraining"]
        
        # Load feedback data if not provided
        if feedback_data is None or feedback_data.empty:
            if use_buffer and self.feedback_buffer:
                # Convert buffer to DataFrame
                buffer_data = [{"description": entry["description"], 
                                "correct_category": entry["category"]} 
                                for entry in self.feedback_buffer]
                feedback_data = pd.DataFrame(buffer_data)
            else:
                # Load from feedback log
                feedback_data = self._load_feedback_data()
        
        if len(feedback_data) < trigger_threshold:
            print(f"⚠️ Not enough feedback data for retraining: {len(feedback_data)} < {trigger_threshold}")
            print(f"   Feedback data will be buffered for later retraining.")
            return None
        
        print(f"📊 Retraining with {len(feedback_data)} feedback samples")
        
        with mlflow.start_run(run_name="incremental_training"):
            # Extract training data from feedback
            descriptions = feedback_data['description'].astype(str).tolist()
            categories = feedback_data['correct_category'].astype(str).tolist()
            
            # Log parameters
            mlflow.log_params({
                "incremental_samples": len(feedback_data),
                "feedback_categories": len(set(categories)),
                "trigger_threshold": trigger_threshold,
                "timestamp": datetime.now().isoformat()
            })
            
            # Backup current model before retraining
            if self.config["backup_before_retrain"]:
                self._backup_current_model()
            
            # Retrain model
            print("🔄 Retraining model with new feedback...")
            start_time = datetime.now()
            
            try:
                # Use retrain method which handles incremental learning
                self.model.retrain(descriptions, categories, incremental=True)
                
                training_time = (datetime.now() - start_time).total_seconds()
                print(f"✅ Incremental training completed in {training_time:.2f} seconds")
                
                # Save updated model
                self.model.save_model(str(self.model_path))
                
                # Clear feedback buffer after successful retraining
                if use_buffer:
                    self.feedback_buffer.clear()
                    self._save_feedback_buffer()
                    print(f"🧹 Feedback buffer cleared after retraining")
                
                # Log metrics
                mlflow.log_metric("incremental_training_time", training_time)
                mlflow.log_metric("feedback_samples_processed", len(feedback_data))
                
                # Evaluate on a small holdout set if available
                if len(feedback_data) > 100:
                    eval_metrics = self._evaluate_on_feedback(feedback_data)
                    if eval_metrics:
                        mlflow.log_metric("post_retrain_accuracy", eval_metrics.get('accuracy', 0))
                
                # Clear feedback log
                self._clear_feedback_log()
                
                # Log training run
                self._log_training_run("incremental", len(feedback_data), {
                    "training_time": training_time,
                    "samples_processed": len(feedback_data)
                })
                
                print("\n" + "=" * 60)
                print("Incremental training completed successfully!")
                print("=" * 60)
                
                return {
                    "samples_processed": len(feedback_data),
                    "training_time": training_time,
                    "model_updated": True
                }
                
            except Exception as e:
                print(f"❌ Incremental training failed: {e}")
                # Restore backup
                self._restore_backup_model()
                raise
    
    def log_feedback(self, transaction_id: str, description: str, 
                        predicted_category: str, user_category: str, 
                        was_correct: bool, user_id: str = None):
        """
        Log user feedback for retraining and trigger retraining if needed.
        
        Args:
            transaction_id: ID of the transaction
            description: Transaction description
            predicted_category: Category predicted by model
            user_category: Category assigned by user
            was_correct: Whether prediction was correct
            user_id: Optional user ID for personalization
        """
        feedback_entry = {
            "transaction_id": transaction_id,
            "description": description,
            "predicted_category": predicted_category,
            "user_category": user_category,
            "was_correct": was_correct,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # Append to feedback log
        with open(self.feedback_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback_entry) + '\n')
        
        # Add incorrect predictions to feedback buffer for online learning
        if not was_correct:
            self._add_to_feedback_buffer([description], [user_category])
        
        # Check if we should trigger retraining
        if self._should_trigger_retraining():
            print("🚀 Triggering incremental retraining...")
            self.run_incremental_training()
    
    def batch_log_feedback(self, feedback_entries: list):
        """
        Log multiple feedback entries at once.
        
        Args:
            feedback_entries: List of feedback dictionaries
        """
        print(f"📝 Logging {len(feedback_entries)} feedback entries...")
        
        incorrect_descriptions = []
        incorrect_categories = []
        
        with open(self.feedback_log_path, 'a', encoding='utf-8') as f:
            for entry in feedback_entries:
                # Ensure required fields
                if not all(k in entry for k in ['description', 'predicted_category', 'user_category', 'was_correct']):
                    print(f"⚠️ Skipping invalid feedback entry: {entry}")
                    continue
                
                # Add timestamp if not present
                if 'timestamp' not in entry:
                    entry['timestamp'] = datetime.now().isoformat()
                
                f.write(json.dumps(entry) + '\n')
                
                # Collect incorrect predictions for buffer
                if not entry['was_correct']:
                    incorrect_descriptions.append(entry['description'])
                    incorrect_categories.append(entry['user_category'])
        
        # Add to feedback buffer
        if incorrect_descriptions:
            self._add_to_feedback_buffer(incorrect_descriptions, incorrect_categories)
        
        # Check retraining trigger
        if self._should_trigger_retraining():
            print("🚀 Triggering batch incremental retraining...")
            self.run_incremental_training()
    
    def get_feedback_buffer_stats(self) -> dict:
        """
        Get statistics about the feedback buffer.
        
        Returns:
            Dictionary with buffer statistics
        """
        if not self.feedback_buffer:
            return {
                "buffer_size": 0,
                "categories": [],
                "oldest_entry": None,
                "newest_entry": None
            }
        
        # Extract categories
        categories = [entry["category"] for entry in self.feedback_buffer]
        unique_categories = list(set(categories))
        
        # Get timestamps
        timestamps = [datetime.fromisoformat(entry["timestamp"].replace('Z', '+00:00')) 
                        for entry in self.feedback_buffer]
        
        return {
            "buffer_size": len(self.feedback_buffer),
            "unique_categories": len(unique_categories),
            "categories_distribution": {cat: categories.count(cat) for cat in unique_categories},
            "oldest_entry": min(timestamps).isoformat() if timestamps else None,
            "newest_entry": max(timestamps).isoformat() if timestamps else None,
            "buffer_max_size": self.config["feedback_buffer_max_size"]
        }
    
    def flush_feedback_buffer(self, force_retraining: bool = False):
        """
        Force process all feedback in buffer.
        
        Args:
            force_retraining: Whether to force retraining even if below threshold
        """
        if not self.feedback_buffer:
            print("📭 Feedback buffer is empty")
            return
        
        print(f"🚀 Flushing feedback buffer ({len(self.feedback_buffer)} entries)...")
        
        # Convert buffer to DataFrame
        buffer_data = [{"description": entry["description"], 
                        "correct_category": entry["category"]} 
                        for entry in self.feedback_buffer]
        feedback_df = pd.DataFrame(buffer_data)
        
        if force_retraining or len(feedback_df) >= self.config["min_feedback_for_retraining"]:
            result = self.run_incremental_training(feedback_df, use_buffer=True)
            if result:
                print(f"✅ Buffer flushed successfully")
            else:
                print(f"⚠️ Buffer flushing completed but no retraining triggered")
        else:
            print(f"⚠️ Buffer has {len(feedback_df)} entries, but {self.config['min_feedback_for_retraining']} required for retraining")
            print(f"   Use --force flag to override")
    
    def _load_feedback_data(self) -> pd.DataFrame:
        """Load and process feedback data from log file."""
        if not self.feedback_log_path.exists():
            return pd.DataFrame()
        
        try:
            data = []
            with open(self.feedback_log_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            entry = json.loads(line)
                            data.append(entry)
                        except json.JSONDecodeError as e:
                            print(f"⚠️ Error parsing line {line_num}: {e}")
                            continue
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            
            # Filter for corrections (where prediction was incorrect)
            corrections = df[df['was_correct'] == False]
            
            if len(corrections) == 0:
                print("✅ No corrections found in feedback data")
                return pd.DataFrame()
            
            # Prepare training data
            training_data = pd.DataFrame({
                'description': corrections['description'],
                'correct_category': corrections['user_category']
            })
            
            # Remove duplicates
            training_data = training_data.drop_duplicates()
            
            print(f"📊 Loaded {len(training_data)} unique corrections from feedback")
            return training_data
            
        except Exception as e:
            print(f"❌ Error loading feedback data: {e}")
            return pd.DataFrame()
    
    def _should_trigger_retraining(self) -> bool:
        """Check if we have enough feedback to trigger retraining."""
        # Check feedback log
        if self.feedback_log_path.exists():
            try:
                with open(self.feedback_log_path, 'r', encoding='utf-8') as f:
                    line_count = sum(1 for _ in f)
                
                if line_count >= self.config["min_feedback_for_retraining"]:
                    return True
            except Exception as e:
                print(f"⚠️ Error checking feedback log: {e}")
        
        # Check feedback buffer
        if len(self.feedback_buffer) >= self.config["min_feedback_for_retraining"]:
            return True
        
        return False
    
    def _clear_feedback_log(self):
        """Clear the feedback log after processing."""
        if self.feedback_log_path.exists():
            # Instead of deleting, archive it
            archive_path = self.model_dir / f"feedback_archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            self.feedback_log_path.rename(archive_path)
            print(f"📁 Feedback log archived to: {archive_path}")
    
    def _backup_current_model(self):
        """Create a backup of the current model before retraining."""
        if self.model_path.exists():
            import shutil
            shutil.copy2(self.model_path, self.backup_model_path)
            print(f"💾 Model backed up to: {self.backup_model_path}")
    
    def _restore_backup_model(self):
        """Restore model from backup if retraining failed."""
        if self.backup_model_path.exists():
            import shutil
            shutil.copy2(self.backup_model_path, self.model_path)
            print(f"🔄 Model restored from backup: {self.backup_model_path}")
    
    def _log_training_run(self, run_type: str, samples: int, metrics: dict = None):
        """Log training run details to training log."""
        log_entry = {
            "run_type": run_type,
            "timestamp": datetime.now().isoformat(),
            "samples": samples,
            "metrics": metrics or {},
            "model_path": str(self.model_path),
            "feedback_buffer_size": len(self.feedback_buffer)
        }
        
        with open(self.training_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def _evaluate_on_feedback(self, feedback_data: pd.DataFrame) -> dict:
        """Evaluate model on feedback data."""
        if len(feedback_data) < 20:
            return None
        
        # Split feedback data for evaluation
        train_data, eval_data = train_test_split(
            feedback_data, 
            test_size=0.2, 
            random_state=42
        )
        
        # Get predictions
        descriptions = eval_data['description'].tolist()
        true_categories = eval_data['correct_category'].tolist()
        
        predictions = self.model.predict_batch(descriptions)
        predicted_categories = [p['predicted_category'] for p in predictions]
        
        # Calculate metrics
        accuracy = accuracy_score(true_categories, predicted_categories)
        
        # Calculate confidence metrics
        confidences = [p['confidence'] for p in predictions]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        print(f"📈 Post-retrain evaluation on {len(eval_data)} samples:")
        print(f"   - Accuracy: {accuracy:.3f}")
        print(f"   - Avg confidence: {avg_confidence:.3f}")
        
        return {
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "evaluation_samples": len(eval_data)
        }
    
    def evaluate_model(self, test_data: pd.DataFrame = None, 
                        test_size: float = 0.2) -> dict:
        """
        Evaluate model performance on test data.
        
        Args:
            test_data: Test dataset (optional)
            test_size: Proportion for test split if test_data not provided
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.model.is_trained():
            print("❌ Model is not trained")
            return None
        
        print("=" * 60)
        print("Model Evaluation")
        print("=" * 60)
        
        # Load or prepare test data
        if test_data is None:
            print("Loading test data...")
            all_data = self.dataset_loader.load_training_data(
                source='all', 
                augment=False,
                sample_size=10000
            )
            
            if len(all_data) < 100:
                print("⚠️ Not enough data for evaluation")
                return None
            
            # Split data
            train_data, test_data = train_test_split(
                all_data, 
                test_size=test_size, 
                random_state=42,
                stratify=all_data['category'] if 'category' in all_data.columns else None
            )
        
        print(f"📊 Evaluating on {len(test_data)} samples")
        
        # Get predictions
        descriptions = test_data['description'].astype(str).tolist()
        true_categories = test_data['category'].astype(str).tolist()
        
        predictions = self.model.predict_batch(descriptions)
        predicted_categories = [p['predicted_category'] for p in predictions]
        
        # Calculate metrics
        accuracy = accuracy_score(true_categories, predicted_categories)
        
        # Classification report
        report = classification_report(
            true_categories, 
            predicted_categories, 
            output_dict=True
        )
        
        # Confusion matrix
        unique_categories = sorted(set(true_categories + predicted_categories))
        cm = confusion_matrix(true_categories, predicted_categories, labels=unique_categories)
        
        # Confidence metrics
        confidences = [p['confidence'] for p in predictions]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        # Low confidence predictions
        low_confidence_indices = [
            i for i, p in enumerate(predictions) 
            if p.get('confidence_level', 'medium') in ['low', 'very_low']
        ]
        
        # Print results
        print(f"\n📈 Evaluation Results:")
        print(f"   - Accuracy: {accuracy:.3f}")
        print(f"   - Average Confidence: {avg_confidence:.3f}")
        print(f"   - Low Confidence Predictions: {len(low_confidence_indices)} ({len(low_confidence_indices)/len(predictions)*100:.1f}%)")
        
        # Show examples of low confidence predictions
        if low_confidence_indices:
            print(f"\n⚠️ Examples of low confidence predictions:")
            for idx in low_confidence_indices[:3]:  # Show first 3
                desc = descriptions[idx][:50] + "..." if len(descriptions[idx]) > 50 else descriptions[idx]
                print(f"   '{desc}'")
                print(f"     True: {true_categories[idx]}, Predicted: {predicted_categories[idx]}")
                print(f"     Confidence: {predictions[idx]['confidence']:.3f}")
        
        # Category-wise accuracy
        print(f"\n📊 Category-wise Performance (top 10):")
        category_accuracies = {}
        for cat in unique_categories:
            cat_indices = [i for i, c in enumerate(true_categories) if c == cat]
            if cat_indices:
                cat_correct = sum(1 for i in cat_indices if true_categories[i] == predicted_categories[i])
                cat_accuracy = cat_correct / len(cat_indices) if cat_indices else 0
                category_accuracies[cat] = cat_accuracy
        
        # Sort by accuracy and show top 10
        sorted_cats = sorted(category_accuracies.items(), key=lambda x: x[1], reverse=True)[:10]
        for cat, acc in sorted_cats:
            print(f"   - {cat}: {acc:.3f}")
        
        # Log to MLflow
        with mlflow.start_run(run_name="model_evaluation", nested=True):
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("avg_confidence", avg_confidence)
            mlflow.log_metric("low_confidence_predictions", len(low_confidence_indices))
            mlflow.log_dict(report, "classification_report.json")
        
        return {
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "low_confidence_count": len(low_confidence_indices),
            "total_samples": len(test_data),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "category_accuracies": category_accuracies
        }
    
    def get_model_info(self) -> dict:
        """Get information about the current model state."""
        model_info = self.model.get_model_info()
        
        # Add pipeline-specific info
        pipeline_info = {
            "model_path": str(self.model_path),
            "model_exists": self.model_path.exists(),
            "feedback_log_path": str(self.feedback_log_path),
            "feedback_entries": self._count_feedback_entries(),
            "feedback_buffer_stats": self.get_feedback_buffer_stats(),
            "last_training_run": self._get_last_training_run(),
            "config": self.config
        }
        
        return {**model_info, **pipeline_info}
    
    def _count_feedback_entries(self) -> int:
        """Count entries in feedback log."""
        if not self.feedback_log_path.exists():
            return 0
        
        try:
            with open(self.feedback_log_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except:
            return 0
    
    def _get_last_training_run(self) -> dict:
        """Get information about the last training run."""
        if not self.training_log_path.exists():
            return {"status": "no_training_log"}
        
        try:
            with open(self.training_log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1]
                    return json.loads(last_line)
        except:
            pass
        
        return {"status": "error_reading_log"}
    
    def trigger_scheduled_retraining(self):
        """Trigger retraining based on schedule or model staleness."""
        model_info = self.model.get_model_info()
        
        if not model_info['is_trained']:
            print("⚠️ Model not trained. Running full training...")
            return self.run_full_training()
        
        # Check if model is stale
        if model_info['last_trained']:
            last_trained = datetime.fromisoformat(model_info['last_trained'].replace('Z', '+00:00'))
            stale_threshold = datetime.now() - timedelta(days=self.config["stale_model_days"])
            
            if last_trained < stale_threshold:
                print(f"🔄 Model is stale (last trained: {last_trained}). Triggering retraining...")
                
                # Check if we have feedback data
                feedback_data = self._load_feedback_data()
                if len(feedback_data) > 0:
                    print(f"   Using {len(feedback_data)} feedback samples for retraining")
                    return self.run_incremental_training(feedback_data)
                else:
                    print("   No feedback data available. Running full training with latest data...")
                    return self.run_full_training()
            else:
                print(f"✅ Model is up to date (last trained: {last_trained})")
        else:
            print("⚠️ Model training date unknown. Consider retraining.")
    
    def cleanup_old_files(self, days_old: int = 30):
        """
        Clean up old model backups, logs, and archives.
        
        Args:
            days_old: Delete files older than this many days
        """
        print(f"🧹 Cleaning up files older than {days_old} days...")
        
        cutoff_time = datetime.now() - timedelta(days=days_old)
        files_deleted = 0
        
        # Clean up old backup files
        for file_path in self.model_dir.glob("*backup*.pkl"):
            if file_path.stat().st_mtime < cutoff_time.timestamp():
                file_path.unlink()
                files_deleted += 1
                print(f"   Deleted: {file_path.name}")
        
        # Clean up old feedback archives
        for file_path in self.model_dir.glob("feedback_archive_*.jsonl"):
            if file_path.stat().st_mtime < cutoff_time.timestamp():
                file_path.unlink()
                files_deleted += 1
                print(f"   Deleted: {file_path.name}")
        
        # Clean up old MLflow runs (simplified)
        mlflow_dir = self.model_dir / "mlruns"
        if mlflow_dir.exists():
            # In production, use MLflow API for cleanup
            print(f"   Note: MLflow runs can be cleaned via: mlflow gc --older-than {days_old}d")
        
        print(f"✅ Cleanup completed. Deleted {files_deleted} files.")