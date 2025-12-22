# disease_detection/detector.py
import numpy as np
import tensorflow as tf
import cv2
from typing import Dict, Tuple, Any
import json
import os
from django.conf import settings

class DiseaseDetector:
    def __init__(self, model_path: str = None):
        """
        Initialize the Disease Detector with a trained model
        
        Args:
            model_path: Path to the trained Keras model
        """
        # Use default path if not provided
        if model_path is None:
            model_path = os.path.join(settings.BASE_DIR, 'digitalkissan/ai_detection/trained_model.keras')
        
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            
        # Define class names (same as before)
        self.class_names = [
            'Apple___Apple_scab',
            'Apple___Black_rot',
            'Apple___Cedar_apple_rust',
            'Apple___healthy',
            'Blueberry___healthy',
            'Cherry_(including_sour)___Powdery_mildew',
            'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight',
            'Corn_(maize)___healthy',
            'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)',
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)',
            'Peach___Bacterial_spot',
            'Peach___healthy',
            'Pepper,_bell___Bacterial_spot',
            'Pepper,_bell___healthy',
            'Potato___Early_blight',
            'Potato___Late_blight',
            'Potato___healthy',
            'Raspberry___healthy',
            'Soybean___healthy',
            'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch',
            'Strawberry___healthy',
            'Tomato___Bacterial_spot',
            'Tomato___Early_blight',
            'Tomato___Late_blight',
            'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]
        
    
        self.plant_type_mapping = {
            'Apple': ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy'],
            'Blueberry': ['Blueberry___healthy'],
            'Cherry': ['Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy'],
            'Corn': ['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
                    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy'],
            'Grape': ['Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy'],
            'Orange': ['Orange___Haunglongbing_(Citrus_greening)'],
            'Peach': ['Peach___Bacterial_spot', 'Peach___healthy'],
            'Pepper': ['Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy'],
            'Potato': ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy'],
            'Raspberry': ['Raspberry___healthy'],
            'Soybean': ['Soybean___healthy'],
            'Squash': ['Squash___Powdery_mildew'],
            'Strawberry': ['Strawberry___Leaf_scorch', 'Strawberry___healthy'],
            'Tomato': ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
                      'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                      'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        }
        
        # Disease information database (same as before)
        self.disease_info = self._load_disease_info()
        
    def _load_disease_info(self) -> Dict:
        """Load disease information including treatments and prevention tips"""
        return {
            'Tomato___Late_blight': {
                'plant': 'Tomato',
                'disease': 'Late Blight',
                'scientific_name': 'Phytophthora infestans',
                'severity': 'High',
                'treatment': [
                    'Remove and destroy infected plants immediately',
                    'Apply copper-based fungicides weekly',
                    'Use chlorothalonil or mancozeb as preventive sprays',
                    'Ensure proper spacing for air circulation'
                ],
                'prevention': [
                    'Plant resistant varieties when available',
                    'Water at soil level to keep foliage dry',
                    'Avoid overhead irrigation',
                    'Rotate crops regularly',
                    'Remove plant debris after harvest'
                ],
                'risk_factors': ['Cool, wet weather', 'High humidity', 'Poor air circulation'],
                'confidence_threshold': 0.7
            },
            # ... Add other diseases as before
        }
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for model prediction
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array
        """
        try:
            # Load and resize image
            image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
            
            # Convert to array and normalize
            input_arr = tf.keras.preprocessing.image.img_to_array(image)
            input_arr = input_arr / 255.0  # Normalize to [0, 1]
            input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension
            
            return input_arr
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            raise
    
    def predict_from_path(self, image_path: str) -> Dict[str, Any]:
        """
        Predict disease from image path
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing prediction results
        """
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # Preprocess image
            input_arr = self.preprocess_image(image_path)
            
            # Make prediction
            predictions = self.model.predict(input_arr)
            
            # Get top 3 predictions
            top_indices = np.argsort(predictions[0])[-3:][::-1]
            top_predictions = [
                {
                    "class_name": self.class_names[i],
                    "confidence": float(predictions[0][i]),
                    "percentage": float(predictions[0][i] * 100)
                }
                for i in top_indices
            ]
            
            # Get top prediction
            top_prediction = top_predictions[0]
            class_name = top_prediction["class_name"]
            
            # Extract plant and disease from class name
            plant, disease = self._extract_plant_disease(class_name)
            
            # Check if healthy
            is_healthy = "healthy" in class_name.lower()
            
            # Get additional info if disease exists in database
            additional_info = {}
            if class_name in self.disease_info:
                additional_info = self.disease_info[class_name]
            
            # Prepare response
            response = {
                "success": True,
                "predictions": top_predictions,
                "top_prediction": top_prediction,
                "plant": plant,
                "disease": disease if not is_healthy else "Healthy",
                "is_healthy": is_healthy,
                "additional_info": additional_info,
                "timestamp": tf.timestamp().numpy()
            }
            
            return response
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}", "success": False}
    
    def predict_from_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Predict disease from image bytes
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Dictionary containing prediction results
        """
        # Save bytes to temporary file and predict
        import tempfile
        import os
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_file.write(image_bytes)
                temp_path = temp_file.name
            
            result = self.predict_from_path(temp_path)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            return result
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}", "success": False}
    
    def _extract_plant_disease(self, class_name: str) -> Tuple[str, str]:
        """
        Extract plant name and disease from class name
        
        Args:
            class_name: Model class name
            
        Returns:
            Tuple of (plant_name, disease_name)
        """
        try:
            # Split by '___'
            parts = class_name.split('___')
            if len(parts) >= 2:
                plant = parts[0]
                disease = parts[1].replace('_', ' ')
                
                # Clean up plant name
                plant = plant.replace('_', ' ')
                if '(including_sour)' in plant:
                    plant = plant.replace('(including_sour)', '').strip()
                if '(maize)' in plant:
                    plant = plant.replace('(maize)', '').strip()
                if ', bell' in plant:
                    plant = plant.replace(', bell', '').strip()
                
                return plant.strip(), disease.strip()
            return "Unknown", "Unknown"
        except:
            return "Unknown", "Unknown"
    
    def get_plant_types(self) -> list:
        """Get list of supported plant types"""
        return list(self.plant_type_mapping.keys())
    
    def get_diseases_for_plant(self, plant_type: str) -> list:
        """Get diseases for a specific plant type"""
        return self.plant_type_mapping.get(plant_type, [])
    
    def analyze_severity(self, confidence: float, disease_class: str) -> str:
        """
        Analyze disease severity based on confidence
        
        Args:
            confidence: Prediction confidence (0-1)
            disease_class: Disease class name
            
        Returns:
            Severity level (Low, Medium, High)
        """
        if "healthy" in disease_class.lower():
            return "Healthy"
        
        if confidence > 0.85:
            return "High"
        elif confidence > 0.7:
            return "Medium"
        else:
            return "Low"
    
    def generate_report(self, prediction_result: Dict) -> Dict:
        """
        Generate a comprehensive disease report
        
        Args:
            prediction_result: Prediction result dictionary
            
        Returns:
            Comprehensive report dictionary
        """
        if not prediction_result.get("success", False):
            return {"error": "Prediction failed"}
        
        top_pred = prediction_result["top_prediction"]
        plant = prediction_result["plant"]
        disease = prediction_result["disease"]
        is_healthy = prediction_result["is_healthy"]
        
        # Calculate severity
        severity = self.analyze_severity(top_pred["confidence"], top_pred["class_name"])
        
        report = {
            "plant": plant,
            "status": "Healthy" if is_healthy else "Diseased",
            "disease": disease,
            "severity": severity,
            "confidence": top_pred["confidence"],
            "confidence_percentage": top_pred["percentage"],
            "timestamp": prediction_result.get("timestamp", tf.timestamp().numpy()),
            "recommendations": self._get_recommendations(plant, disease, severity),
            "additional_info": prediction_result.get("additional_info", {})
        }
        
        # Add alternative predictions if confidence is low
        if top_pred["confidence"] < 0.7:
            report["alternative_predictions"] = prediction_result["predictions"][1:3]
            report["note"] = "Low confidence - consider alternative diagnoses"
        
        return report
    
    def _get_recommendations(self, plant: str, disease: str, severity: str) -> Dict:
        """Get treatment recommendations based on plant, disease, and severity"""
        recommendations = {
            "immediate_actions": [],
            "short_term_treatments": [],
            "long_term_prevention": []
        }
        
        if disease.lower() == "healthy":
            recommendations["immediate_actions"] = ["Continue current care routine"]
            recommendations["short_term_treatments"] = ["Monitor plant health regularly"]
            recommendations["long_term_prevention"] = ["Maintain proper watering and fertilization"]
        else:
            # Generic recommendations based on severity
            if severity == "High":
                recommendations["immediate_actions"] = [
                    f"Immediately isolate {plant} plant",
                    "Remove and destroy severely infected leaves/plants",
                    "Apply emergency treatment"
                ]
            elif severity == "Medium":
                recommendations["immediate_actions"] = [
                    f"Remove infected parts of {plant}",
                    "Apply recommended fungicide",
                    "Improve growing conditions"
                ]
            else:  # Low severity
                recommendations["immediate_actions"] = [
                    f"Monitor {plant} closely",
                    "Apply preventive measures",
                    "Improve plant immunity"
                ]
            
            # Disease-specific recommendations
            disease_key = f"{plant.replace(' ', '_')}___{disease.replace(' ', '_')}"
            if disease_key in self.disease_info:
                info = self.disease_info[disease_key]
                recommendations["short_term_treatments"] = info.get("treatment", [])
                recommendations["long_term_prevention"] = info.get("prevention", [])
        
        return recommendations