"""
Locust load testing script for brain tumor classification API.
Simulates flood of requests to test model performance.
"""

from locust import HttpUser, task, between
import random
import os
import io
from PIL import Image
import numpy as np


class BrainTumorAPIUser(HttpUser):
    """
    Locust user class for load testing the brain tumor classification API.
    """
    wait_time = between(1, 3)  # Wait between 1 and 3 seconds between requests
    
    def on_start(self):
        """Called when a simulated user starts."""
        # Create a dummy image for testing
        self.test_image = self.create_dummy_image()
    
    def create_dummy_image(self):
        """Create a dummy MRI-like image for testing."""
        # Create a random image that looks like an MRI scan
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        return img_bytes.getvalue()
    
    @task(3)
    def predict_single_image(self):
        """
        Test single image prediction endpoint.
        Weight: 3 (most common operation)
        """
        files = {'file': ('test_image.jpg', self.test_image, 'image/jpeg')}
        
        with self.client.post(
            "/predict",
            files=files,
            catch_response=True,
            name="Predict Single Image"
        ) as response:
            if response.status_code == 200:
                response.success()
                result = response.json()
                # Log prediction time if available
                if 'prediction_time_seconds' in result:
                    response.request_meta['response_time'] = result['prediction_time_seconds'] * 1000
            elif response.status_code == 503:
                response.failure("Model not loaded")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def predict_batch(self):
        """
        Test batch prediction endpoint.
        Weight: 1 (less common)
        """
        files = [
            ('files', ('test_image1.jpg', self.test_image, 'image/jpeg')),
            ('files', ('test_image2.jpg', self.test_image, 'image/jpeg')),
        ]
        
        with self.client.post(
            "/predict/batch",
            files=files,
            catch_response=True,
            name="Predict Batch"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def health_check(self):
        """
        Test health check endpoint.
        Weight: 1
        """
        with self.client.get("/health", catch_response=True, name="Health Check") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def uptime_check(self):
        """
        Test uptime endpoint.
        Weight: 1
        """
        with self.client.get("/uptime", catch_response=True, name="Uptime Check") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")


# Configuration for different test scenarios
class QuickTestUser(BrainTumorAPIUser):
    """Quick test with shorter wait times."""
    wait_time = between(0.5, 1.5)


class HeavyLoadUser(BrainTumorAPIUser):
    """Heavy load test with minimal wait times."""
    wait_time = between(0.1, 0.5)


# Example usage:
# locust -f locustfile.py --host=http://localhost:8000
# 
# For different user types:
# locust -f locustfile.py --host=http://localhost:8000 -u 100 -r 10
# 
# Where:
# -u: Number of users
# -r: Spawn rate (users per second)
# --host: API base URL

