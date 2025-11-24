"""
Database module for storing uploaded data and retraining history.
Uses SQLite for simplicity and portability.
"""

import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Optional
import json


class DatabaseManager:
    """Manages database operations for uploaded data and retraining history."""
    
    def __init__(self, db_path='data/retraining_database.db'):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()
    
    def get_connection(self):
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn
    
    def init_database(self):
        """Initialize database tables if they don't exist."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Table for uploaded images
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS uploaded_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                class_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER,
                upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                processed BOOLEAN DEFAULT 0,
                used_for_training BOOLEAN DEFAULT 0,
                training_session_id INTEGER,
                metadata TEXT,
                FOREIGN KEY (training_session_id) REFERENCES training_sessions(id)
            )
        ''')
        
        # Table for training sessions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                epochs INTEGER,
                fine_tune_epochs INTEGER,
                initial_accuracy REAL,
                final_accuracy REAL,
                initial_precision REAL,
                final_precision REAL,
                initial_recall REAL,
                final_recall REAL,
                initial_f1_score REAL,
                final_f1_score REAL,
                model_path TEXT,
                status TEXT DEFAULT 'pending',
                notes TEXT,
                images_used INTEGER DEFAULT 0
            )
        ''')
        
        # Table for preprocessing logs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS preprocessing_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                training_session_id INTEGER,
                images_processed INTEGER,
                features_extracted INTEGER,
                processing_time_seconds REAL,
                status TEXT DEFAULT 'completed',
                error_message TEXT,
                FOREIGN KEY (training_session_id) REFERENCES training_sessions(id)
            )
        ''')
        
        conn.commit()
        conn.close()
        print(f"Database initialized at {self.db_path}")
    
    def save_uploaded_image(self, filename: str, class_name: str, file_path: str, 
                           file_size: int, metadata: Optional[Dict] = None) -> int:
        """
        Save uploaded image information to database.
        
        Args:
            filename: Name of the uploaded file
            class_name: Class label for the image
            file_path: Full path where file is stored
            file_size: Size of file in bytes
            metadata: Optional metadata dictionary
            
        Returns:
            ID of the inserted record
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor.execute('''
            INSERT INTO uploaded_images 
            (filename, class_name, file_path, file_size, metadata)
            VALUES (?, ?, ?, ?, ?)
        ''', (filename, class_name, file_path, file_size, metadata_json))
        
        image_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return image_id
    
    def get_uploaded_images(self, processed: Optional[bool] = None, 
                           class_name: Optional[str] = None) -> List[Dict]:
        """
        Get uploaded images from database.
        
        Args:
            processed: Filter by processed status (None for all)
            class_name: Filter by class name (None for all)
            
        Returns:
            List of image records as dictionaries
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        query = "SELECT * FROM uploaded_images WHERE 1=1"
        params = []
        
        if processed is not None:
            query += " AND processed = ?"
            params.append(1 if processed else 0)
        
        if class_name:
            query += " AND class_name = ?"
            params.append(class_name)
        
        query += " ORDER BY upload_timestamp DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def mark_images_processed(self, image_ids: List[int], training_session_id: int):
        """
        Mark images as processed and used for training.
        
        Args:
            image_ids: List of image IDs to mark
            training_session_id: ID of training session
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        placeholders = ','.join(['?'] * len(image_ids))
        cursor.execute(f'''
            UPDATE uploaded_images 
            SET processed = 1, used_for_training = 1, training_session_id = ?
            WHERE id IN ({placeholders})
        ''', [training_session_id] + image_ids)
        
        conn.commit()
        conn.close()
    
    def create_training_session(self, epochs: int, fine_tune_epochs: int, 
                               model_path: str, notes: Optional[str] = None) -> int:
        """
        Create a new training session record.
        
        Args:
            epochs: Number of training epochs
            fine_tune_epochs: Number of fine-tuning epochs
            model_path: Path where model will be saved
            notes: Optional notes about the training session
            
        Returns:
            ID of the created training session
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO training_sessions 
            (epochs, fine_tune_epochs, model_path, status, notes)
            VALUES (?, ?, ?, 'pending', ?)
        ''', (epochs, fine_tune_epochs, model_path, notes))
        
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return session_id
    
    def update_training_session(self, session_id: int, status: str,
                               initial_metrics: Optional[Dict] = None,
                               final_metrics: Optional[Dict] = None,
                               images_used: Optional[int] = None):
        """
        Update training session with results.
        
        Args:
            session_id: ID of training session
            status: Status ('pending', 'in_progress', 'completed', 'failed')
            initial_metrics: Dictionary with initial model metrics
            final_metrics: Dictionary with final model metrics
            images_used: Number of images used in training
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        updates = ["status = ?"]
        params = [status]
        
        if initial_metrics:
            updates.append("initial_accuracy = ?")
            updates.append("initial_precision = ?")
            updates.append("initial_recall = ?")
            updates.append("initial_f1_score = ?")
            params.extend([
                initial_metrics.get('accuracy'),
                initial_metrics.get('precision'),
                initial_metrics.get('recall'),
                initial_metrics.get('f1_score')
            ])
        
        if final_metrics:
            updates.append("final_accuracy = ?")
            updates.append("final_precision = ?")
            updates.append("final_recall = ?")
            updates.append("final_f1_score = ?")
            params.extend([
                final_metrics.get('accuracy'),
                final_metrics.get('precision'),
                final_metrics.get('recall'),
                final_metrics.get('f1_score')
            ])
        
        if images_used is not None:
            updates.append("images_used = ?")
            params.append(images_used)
        
        params.append(session_id)
        
        query = f"UPDATE training_sessions SET {', '.join(updates)} WHERE id = ?"
        cursor.execute(query, params)
        
        conn.commit()
        conn.close()
    
    def log_preprocessing(self, training_session_id: int, images_processed: int,
                        features_extracted: int, processing_time: float,
                        status: str = 'completed', error_message: Optional[str] = None):
        """
        Log preprocessing activity.
        
        Args:
            training_session_id: ID of associated training session
            images_processed: Number of images processed
            features_extracted: Number of features extracted
            processing_time: Time taken in seconds
            status: Status of preprocessing
            error_message: Error message if failed
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO preprocessing_logs
            (training_session_id, images_processed, features_extracted, 
             processing_time_seconds, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (training_session_id, images_processed, features_extracted,
              processing_time, status, error_message))
        
        conn.commit()
        conn.close()
    
    def get_training_sessions(self, limit: int = 10) -> List[Dict]:
        """
        Get recent training sessions.
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of training session records
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM training_sessions 
            ORDER BY session_timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_training_statistics(self) -> Dict:
        """
        Get overall training statistics.
        
        Returns:
            Dictionary with statistics
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Total uploaded images
        cursor.execute("SELECT COUNT(*) FROM uploaded_images")
        total_images = cursor.fetchone()[0]
        
        # Processed images
        cursor.execute("SELECT COUNT(*) FROM uploaded_images WHERE processed = 1")
        processed_images = cursor.fetchone()[0]
        
        # Total training sessions
        cursor.execute("SELECT COUNT(*) FROM training_sessions")
        total_sessions = cursor.fetchone()[0]
        
        # Completed sessions
        cursor.execute("SELECT COUNT(*) FROM training_sessions WHERE status = 'completed'")
        completed_sessions = cursor.fetchone()[0]
        
        # Images by class
        cursor.execute('''
            SELECT class_name, COUNT(*) as count 
            FROM uploaded_images 
            GROUP BY class_name
        ''')
        images_by_class = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        return {
            'total_uploaded_images': total_images,
            'processed_images': processed_images,
            'total_training_sessions': total_sessions,
            'completed_sessions': completed_sessions,
            'images_by_class': images_by_class
        }


# Global database instance
_db_instance = None

def get_database() -> DatabaseManager:
    """Get or create global database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager()
    return _db_instance

