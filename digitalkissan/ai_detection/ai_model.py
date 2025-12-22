# disease_detection/detector.py
import numpy as np
import tensorflow as tf
import cv2
from typing import Dict, Tuple, Any
import json
import os
import logging
import sys
from datetime import datetime
from django.conf import settings

# Configure logging with both console and file output
def setup_logging():
    """Setup comprehensive logging configuration"""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(settings.BASE_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a unique log file for each session
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'disease_detector_{timestamp}.log')
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure root logger
    root_logger.setLevel(logging.DEBUG)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    
    # File handler (detailed logs)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler (simpler output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Log startup information
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Log level: DEBUG (file), INFO (console)")
    
    return log_file, logger

# Initialize logging
log_file, _ = setup_logging()
logger = logging.getLogger(__name__)

class DiseaseDetector:
    def __init__(self, model_path: str = None):
        """
        Initialize the Disease Detector with a trained model
        
        Args:
            model_path: Path to the trained Keras model
        """
        logger.info("=" * 80)
        logger.info("INITIALIZING DISEASE DETECTOR")
        logger.info("=" * 80)
        
        # Log environment info
        logger.info(f"Python version: {sys.version}")
        logger.info(f"TensorFlow version: {tf.__version__}")
        logger.info(f"NumPy version: {np.__version__}")
        logger.info(f"Django settings DEBUG: {settings.DEBUG}")
        
        # Use default path if not provided
        if model_path is None:
            model_path = os.path.join(settings.BASE_DIR, 'ai_detection', 'trained_model.keras')
        
        logger.info(f"Model path: {model_path}")
        logger.info(f"Model file exists: {os.path.exists(model_path)}")
        
        if not os.path.exists(model_path):
            logger.error(f"CRITICAL: Model file not found at {model_path}")
            logger.error(f"Current working directory: {os.getcwd()}")
            logger.error(f"BASE_DIR: {settings.BASE_DIR}")
            
            # Try to find the model
            logger.info(" Searching for model file...")
            for root, dirs, files in os.walk(settings.BASE_DIR):
                if 'trained_model.keras' in files:
                    found_path = os.path.join(root, 'trained_model.keras')
                    logger.info(f" Found model at: {found_path}")
                    model_path = found_path
                    break
        
        try:
            logger.info("Loading TensorFlow model...")
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
            
            # Log model summary
            logger.info(" MODEL ARCHITECTURE:")
            logger.info(f"  Input shape: {self.model.input_shape}")
            logger.info(f"  Output shape: {self.model.output_shape}")
            logger.info(f"  Number of layers: {len(self.model.layers)}")
            logger.info(f"  Number of parameters: {self.model.count_params():,}")
            
            # Log the output layer details
            output_layer = self.model.layers[-1]
            logger.info(f"  Output layer: {output_layer.name}")
            logger.info(f"  Output activation: {output_layer.activation.__name__ if hasattr(output_layer.activation, '__name__') else str(output_layer.activation)}")
            
        except Exception as e:
            logger.error(f" Error loading model: {e}")
            logger.exception("Model loading failed with traceback:")
            self.model = None
            
        # Define class names
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
        
        logger.info(f"Total class names loaded: {len(self.class_names)}")
        logger.debug(f"Class names (first 10): {self.class_names[:10]}")
        
        # Validate class count matches model output
        if self.model and self.model.output_shape[-1] != len(self.class_names):
            logger.error(f" CLASS COUNT MISMATCH!")
            logger.error(f"  Model expects {self.model.output_shape[-1]} classes")
            logger.error(f"  Class names list has {len(self.class_names)} classes")
        
        # Plant type mapping
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
        
        logger.info(f"🌱 Plant types loaded: {len(self.plant_type_mapping.keys())}")
        logger.debug(f"Plant types: {list(self.plant_type_mapping.keys())}")
        
        # Disease information database
        self.disease_info = self._load_disease_info()
        logger.info(f" Disease info entries loaded: {len(self.disease_info)}")
        
        logger.info("=" * 80)
        logger.info(" DISEASE DETECTOR INITIALIZATION COMPLETE")
        logger.info("=" * 80)
        
        
        self._save_initialization_summary()
    
    

    def generate_report(self, result: Dict) -> Dict:
        """Generate comprehensive report from prediction results"""
        logger.info(" Generating comprehensive report")
    
        severity = "Low"
        if not result.get('is_healthy', True):
            if result.get('additional_info', {}).get('severity'):
                severity = result['additional_info']['severity']
            else:
                # Determine severity based on confidence
                confidence = result.get('top_prediction', {}).get('percentage', 0)
                if confidence > 80:
                    severity = 'High'
                elif confidence > 60:
                    severity = 'Medium'
                else:
                    severity = 'Low'
    
        recommendations = {
            'short_term_treatments': result.get('additional_info', {}).get('treatment', []),
            'long_term_prevention': result.get('additional_info', {}).get('prevention', []),
            'immediate_actions': [
                'Isolate affected plant if possible',
                'Remove severely infected leaves',
                'Improve air circulation around plant'
            ] if not result.get('is_healthy', True) else [
                'Continue regular watering schedule',
                'Monitor for any changes',
                'Maintain optimal growing conditions'
            ]
        }
    
        return {
            'severity': severity,
            'recommendations': recommendations,
            'confidence_level': 'High' if result.get('top_prediction', {}).get('percentage', 0) > 80 else 'Medium',
            'risk_factors': result.get('additional_info', {}).get('risk_factors', ['Unknown'])
        }

    def get_diseases_for_plant(self, plant_type: str) -> list[str]:
        """
        Get diseases for a specific plant type
    
        Args:
            plant_type: Name of the plant
        
        Returns:
            List of diseases for the plant
        """
        diseases = []
        for class_name in self.class_names:
            if plant_type in class_name:
                disease_part = class_name.split('___')[1] if '___' in class_name else class_name
                disease = disease_part.replace('_', ' ').replace('(including sour)', '').strip()
                if disease not in diseases and disease.lower() != 'healthy':
                    diseases.append(disease)
        return diseases

    def get_supported_formats(self) -> list[str]:
        """Get supported image formats"""
        return ['jpg', 'jpeg', 'png', 'webp']
        
    def _save_initialization_summary(self):
        """Save initialization summary to a separate file"""
        summary_file = os.path.join(settings.BASE_DIR, 'logs', 'detector_summary.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Disease Detector Initialization Summary\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Model loaded: {self.model is not None}\n")
            f.write(f"Total classes: {len(self.class_names)}\n")
            f.write(f"Plant types: {len(self.plant_type_mapping)}\n")
            f.write(f"Disease info entries: {len(self.disease_info)}\n")
            
            if self.model:
                f.write(f"\nModel Architecture:\n")
                f.write(f"  Input shape: {self.model.input_shape}\n")
                f.write(f"  Output shape: {self.model.output_shape}\n")
                f.write(f"  Parameters: {self.model.count_params():,}\n")
        
        logger.info(f" Initialization summary saved to: {summary_file}")
        
    def _load_disease_info(self) -> Dict:
       """Load comprehensive disease information for all classes"""
       logger.debug("Loading comprehensive disease information database")
       return {
        # Apple diseases
        'Apple___Apple_scab': {
            'plant': 'Apple',
            'disease': 'Apple Scab',
            'scientific_name': 'Venturia inaequalis',
            'severity': 'Medium',
            'treatment': [
                'Apply fungicides containing myclobutanil or captan',
                'Prune to improve air circulation',
                'Remove infected leaves and fruit'
            ],
            'prevention': [
                'Plant resistant varieties',
                'Rake and destroy fallen leaves',
                'Ensure good air circulation',
                'Avoid overhead watering'
            ],
            'risk_factors': ['Cool, wet spring weather', 'High humidity'],
            'confidence_threshold': 0.7
        },
        'Apple___Black_rot': {
            'plant': 'Apple',
            'disease': 'Black Rot',
            'scientific_name': 'Botryosphaeria obtusa',
            'severity': 'Medium',
            'treatment': [
                'Prune infected branches 6-8 inches below cankers',
                'Apply copper-based fungicides',
                'Remove mummified fruit from trees'
            ],
            'prevention': [
                'Practice good sanitation',
                'Avoid wounding trees',
                'Maintain tree vigor',
                'Remove dead wood promptly'
            ],
            'risk_factors': ['Warm, humid weather', 'Tree wounds'],
            'confidence_threshold': 0.7
        },
        'Apple___Cedar_apple_rust': {
            'plant': 'Apple',
            'disease': 'Cedar Apple Rust',
            'scientific_name': 'Gymnosporangium juniperi-virginianae',
            'severity': 'Medium',
            'treatment': [
                'Apply fungicides containing myclobutanil or triadimefon',
                'Remove galls from nearby junipers',
                'Protect new growth in spring'
            ],
            'prevention': [
                'Plant resistant apple varieties',
                'Remove nearby juniper hosts',
                'Space trees for good air flow'
            ],
            'risk_factors': ['Wet spring weather', 'Proximity to junipers'],
            'confidence_threshold': 0.7
        },
        'Apple___healthy': {
            'plant': 'Apple',
            'disease': 'Healthy',
            'scientific_name': 'N/A',
            'severity': 'None',
            'treatment': [
                'Continue regular monitoring',
                'Maintain proper watering schedule',
                'Apply balanced fertilizer'
            ],
            'prevention': [
                'Regular inspection for pests',
                'Proper pruning techniques',
                'Soil testing and amendment'
            ],
            'risk_factors': [],
            'confidence_threshold': 0.7
        },
        
        # Blueberry
        'Blueberry___healthy': {
            'plant': 'Blueberry',
            'disease': 'Healthy',
            'scientific_name': 'N/A',
            'severity': 'None',
            'treatment': [
                'Maintain soil acidity (pH 4.5-5.5)',
                'Apply appropriate fertilizers',
                'Monitor for pest issues'
            ],
            'prevention': [
                'Regular pruning for air circulation',
                'Proper mulching with pine needles',
                'Adequate water management'
            ],
            'risk_factors': [],
            'confidence_threshold': 0.7
        },
        
        # Cherry diseases
        'Cherry_(including_sour)___Powdery_mildew': {
            'plant': 'Cherry',
            'disease': 'Powdery Mildew',
            'scientific_name': 'Podosphaera clandestina',
            'severity': 'Medium',
            'treatment': [
                'Apply sulfur or potassium bicarbonate fungicides',
                'Prune infected shoots',
                'Improve air circulation'
            ],
            'prevention': [
                'Plant resistant varieties',
                'Avoid overhead irrigation',
                'Maintain proper spacing'
            ],
            'risk_factors': ['Warm, dry days with cool nights', 'High humidity'],
            'confidence_threshold': 0.7
        },
        'Cherry_(including_sour)___healthy': {
            'plant': 'Cherry',
            'disease': 'Healthy',
            'scientific_name': 'N/A',
            'severity': 'None',
            'treatment': [
                'Regular watering during dry periods',
                'Annual pruning for shape and health',
                'Monitor for common cherry pests'
            ],
            'prevention': [
                'Proper site selection with good drainage',
                'Regular soil testing',
                'Winter protection in cold climates'
            ],
            'risk_factors': [],
            'confidence_threshold': 0.7
        },
        
        # Corn (Maize) diseases
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
            'plant': 'Corn',
            'disease': 'Gray Leaf Spot',
            'scientific_name': 'Cercospora zeae-maydis',
            'severity': 'High',
            'treatment': [
                'Apply fungicides containing azoxystrobin or pyraclostrobin',
                'Remove crop debris after harvest'
            ],
            'prevention': [
                'Plant resistant hybrids',
                'Practice crop rotation',
                'Avoid continuous corn planting'
            ],
            'risk_factors': ['Warm, humid conditions', 'Frequent rainfall'],
            'confidence_threshold': 0.7
        },
        'Corn_(maize)___Common_rust_': {
            'plant': 'Corn',
            'disease': 'Common Rust',
            'scientific_name': 'Puccinia sorghi',
            'severity': 'Medium',
            'treatment': [
                'Apply fungicides containing azoxystrobin or pyraclostrobin',
                'Remove severely infected plants'
            ],
            'prevention': [
                'Plant resistant hybrids',
                'Ensure proper spacing',
                'Avoid late planting'
            ],
            'risk_factors': ['Cool temperatures', 'High humidity', 'Dew formation'],
            'confidence_threshold': 0.7
        },
        'Corn_(maize)___Northern_Leaf_Blight': {
            'plant': 'Corn',
            'disease': 'Northern Leaf Blight',
            'scientific_name': 'Exserohilum turcicum',
            'severity': 'Medium',
            'treatment': [
                'Apply fungicides at first sign of disease',
                'Remove and destroy infected plant debris'
            ],
            'prevention': [
                'Use resistant hybrids',
                'Practice crop rotation',
                'Plow under crop residue'
            ],
            'risk_factors': ['Moderate temperatures', 'High humidity'],
            'confidence_threshold': 0.7
        },
        'Corn_(maize)___healthy': {
            'plant': 'Corn',
            'disease': 'Healthy',
            'scientific_name': 'N/A',
            'severity': 'None',
            'treatment': [
                'Maintain proper nitrogen levels',
                'Ensure adequate irrigation',
                'Monitor for pest pressure'
            ],
            'prevention': [
                'Soil testing and proper fertilization',
                'Weed control',
                'Proper planting depth and spacing'
            ],
            'risk_factors': [],
            'confidence_threshold': 0.7
        },
        
        # Grape diseases
        'Grape___Black_rot': {
            'plant': 'Grape',
            'disease': 'Black Rot',
            'scientific_name': 'Guignardia bidwellii',
            'severity': 'High',
            'treatment': [
                'Apply fungicides containing mancozeb or myclobutanil',
                'Remove and destroy infected fruit and leaves'
            ],
            'prevention': [
                'Prune for good air circulation',
                'Remove mummified berries',
                'Site selection with good air flow'
            ],
            'risk_factors': ['Warm, wet weather during flowering'],
            'confidence_threshold': 0.7
        },
        'Grape___Esca_(Black_Measles)': {
            'plant': 'Grape',
            'disease': 'Esca (Black Measles)',
            'scientific_name': 'Phaeomoniella chlamydospora',
            'severity': 'High',
            'treatment': [
                'Prune infected wood well below symptoms',
                'Protect pruning wounds',
                'Maintain vine vigor'
            ],
            'prevention': [
                'Use certified disease-free plants',
                'Avoid wounding vines',
                'Proper pruning techniques'
            ],
            'risk_factors': ['Vine stress', 'Pruning wounds'],
            'confidence_threshold': 0.7
        },
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
            'plant': 'Grape',
            'disease': 'Leaf Blight',
            'scientific_name': 'Pseudocercospora vitis',
            'severity': 'Medium',
            'treatment': [
                'Apply copper-based fungicides',
                'Remove severely infected leaves'
            ],
            'prevention': [
                'Good canopy management',
                'Remove fallen leaves',
                'Avoid overhead irrigation'
            ],
            'risk_factors': ['Warm, humid conditions'],
            'confidence_threshold': 0.7
        },
        'Grape___healthy': {
            'plant': 'Grape',
            'disease': 'Healthy',
            'scientific_name': 'N/A',
            'severity': 'None',
            'treatment': [
                'Regular monitoring for pests',
                'Appropriate irrigation',
                'Balanced fertilization'
            ],
            'prevention': [
                'Proper trellising system',
                'Regular pruning',
                'Good soil drainage'
            ],
            'risk_factors': [],
            'confidence_threshold': 0.7
        },
        
        # Orange
        'Orange___Haunglongbing_(Citrus_greening)': {
            'plant': 'Orange',
            'disease': 'Citrus Greening (HLB)',
            'scientific_name': 'Candidatus Liberibacter asiaticus',
            'severity': 'High',
            'treatment': [
                'Remove infected trees',
                'Control psyllid vectors with insecticides',
                'Use nutritional supplements to support tree health'
            ],
            'prevention': [
                'Plant disease-free nursery stock',
                'Regular monitoring for psyllids',
                'Remove alternative host plants'
            ],
            'risk_factors': ['Asian citrus psyllid presence', 'Warm climates'],
            'confidence_threshold': 0.7
        },
        
        # Peach diseases
        'Peach___Bacterial_spot': {
            'plant': 'Peach',
            'disease': 'Bacterial Spot',
            'scientific_name': 'Xanthomonas arboricola pv. pruni',
            'severity': 'Medium',
            'treatment': [
                'Apply copper bactericides during dormancy',
                'Prune infected branches',
                'Improve air circulation'
            ],
            'prevention': [
                'Plant resistant varieties',
                'Avoid overhead irrigation',
                'Proper orchard sanitation'
            ],
            'risk_factors': ['Warm, rainy weather', 'Wind-driven rain'],
            'confidence_threshold': 0.7
        },
        'Peach___healthy': {
            'plant': 'Peach',
            'disease': 'Healthy',
            'scientific_name': 'N/A',
            'severity': 'None',
            'treatment': [
                'Regular thinning of fruit',
                'Proper watering schedule',
                'Monitor for common pests'
            ],
            'prevention': [
                'Annual pruning for shape',
                'Winter chill requirement fulfillment',
                'Soil pH management'
            ],
            'risk_factors': [],
            'confidence_threshold': 0.7
        },
        
        # Pepper diseases
        'Pepper,_bell___Bacterial_spot': {
            'plant': 'Pepper',
            'disease': 'Bacterial Spot',
            'scientific_name': 'Xanthomonas campestris pv. vesicatoria',
            'severity': 'Medium',
            'treatment': [
                'Apply copper-based bactericides',
                'Remove infected plants',
                'Avoid working with wet plants'
            ],
            'prevention': [
                'Use disease-free seeds',
                'Practice crop rotation',
                'Avoid overhead watering'
            ],
            'risk_factors': ['Warm, wet conditions', 'Plant injury'],
            'confidence_threshold': 0.7
        },
        'Pepper,_bell___healthy': {
            'plant': 'Pepper',
            'disease': 'Healthy',
            'scientific_name': 'N/A',
            'severity': 'None',
            'treatment': [
                'Regular watering',
                'Support plants with stakes',
                'Monitor for pest issues'
            ],
            'prevention': [
                'Proper spacing for air flow',
                'Mulching to retain moisture',
                'Regular fertilization'
            ],
            'risk_factors': [],
            'confidence_threshold': 0.7
        },
        
        # Potato diseases
        'Potato___Early_blight': {
            'plant': 'Potato',
            'disease': 'Early Blight',
            'scientific_name': 'Alternaria solani',
            'severity': 'Medium',
            'treatment': [
                'Apply chlorothalonil or mancozeb fungicides',
                'Remove infected leaves',
                'Ensure proper plant nutrition'
            ],
            'prevention': [
                'Use certified seed potatoes',
                'Practice crop rotation',
                'Maintain soil fertility'
            ],
            'risk_factors': ['Warm, humid conditions', 'Plant stress'],
            'confidence_threshold': 0.7
        },
        'Potato___Late_blight': {
            'plant': 'Potato',
            'disease': 'Late Blight',
            'scientific_name': 'Phytophthora infestans',
            'severity': 'High',
            'treatment': [
                'Apply fungicides containing chlorothalonil or mancozeb',
                'Remove infected plants immediately',
                'Destroy all potato culls'
            ],
            'prevention': [
                'Plant resistant varieties',
                'Use certified disease-free seed',
                'Destroy volunteer potato plants'
            ],
            'risk_factors': ['Cool, wet weather', 'High humidity'],
            'confidence_threshold': 0.7
        },
        'Potato___healthy': {
            'plant': 'Potato',
            'disease': 'Healthy',
            'scientific_name': 'N/A',
            'severity': 'None',
            'treatment': [
                'Regular hilling of soil',
                'Consistent moisture management',
                'Monitor for Colorado potato beetle'
            ],
            'prevention': [
                'Proper soil preparation',
                'Crop rotation every 3-4 years',
                'Good weed control'
            ],
            'risk_factors': [],
            'confidence_threshold': 0.7
        },
        
        # Raspberry
        'Raspberry___healthy': {
            'plant': 'Raspberry',
            'disease': 'Healthy',
            'scientific_name': 'N/A',
            'severity': 'None',
            'treatment': [
                'Regular watering during fruiting',
                'Annual pruning of old canes',
                'Support with trellising'
            ],
            'prevention': [
                'Plant in well-drained soil',
                'Proper spacing for air flow',
                'Mulch to control weeds'
            ],
            'risk_factors': [],
            'confidence_threshold': 0.7
        },
        
        # Soybean
        'Soybean___healthy': {
            'plant': 'Soybean',
            'disease': 'Healthy',
            'scientific_name': 'N/A',
            'severity': 'None',
            'treatment': [
                'Proper irrigation management',
                'Monitor for pest pressure',
                'Soil testing for nutrient needs'
            ],
            'prevention': [
                'Crop rotation with non-legumes',
                'Proper planting density',
                'Weed control programs'
            ],
            'risk_factors': [],
            'confidence_threshold': 0.7
        },
        
        # Squash
        'Squash___Powdery_mildew': {
            'plant': 'Squash',
            'disease': 'Powdery Mildew',
            'scientific_name': 'Podosphaera xanthii',
            'severity': 'Medium',
            'treatment': [
                'Apply fungicides containing potassium bicarbonate or neem oil',
                'Remove severely infected leaves',
                'Improve air circulation'
            ],
            'prevention': [
                'Plant resistant varieties',
                'Avoid overhead watering',
                'Provide adequate spacing'
            ],
            'risk_factors': ['Warm days with cool nights', 'High humidity'],
            'confidence_threshold': 0.7
        },
        
        # Strawberry diseases
        'Strawberry___Leaf_scorch': {
            'plant': 'Strawberry',
            'disease': 'Leaf Scorch',
            'scientific_name': 'Diplocarpon earlianum',
            'severity': 'Medium',
            'treatment': [
                'Apply fungicides containing captan or thiram',
                'Remove infected leaves',
                'Improve air circulation'
            ],
            'prevention': [
                'Plant resistant varieties',
                'Remove old leaves after harvest',
                'Avoid overhead irrigation'
            ],
            'risk_factors': ['Wet foliage', 'High humidity'],
            'confidence_threshold': 0.7
        },
        'Strawberry___healthy': {
            'plant': 'Strawberry',
            'disease': 'Healthy',
            'scientific_name': 'N/A',
            'severity': 'None',
            'treatment': [
                'Regular watering during dry spells',
                'Renovate beds after harvest',
                'Monitor for spider mites'
            ],
            'prevention': [
                'Proper bed preparation',
                'Mulching with straw',
                'Crop rotation every 3-4 years'
            ],
            'risk_factors': [],
            'confidence_threshold': 0.7
        },
        
        # Tomato diseases (comprehensive)
        'Tomato___Bacterial_spot': {
            'plant': 'Tomato',
            'disease': 'Bacterial Spot',
            'scientific_name': 'Xanthomonas campestris pv. vesicatoria',
            'severity': 'Medium',
            'treatment': [
                'Apply copper-based bactericides',
                'Remove infected plants',
                'Avoid overhead watering'
            ],
            'prevention': [
                'Use disease-free seeds',
                'Practice crop rotation',
                'Sanitize gardening tools'
            ],
            'risk_factors': ['Warm, wet weather', 'Plant injury'],
            'confidence_threshold': 0.7
        },
        'Tomato___Early_blight': {
            'plant': 'Tomato',
            'disease': 'Early Blight',
            'scientific_name': 'Alternaria solani',
            'severity': 'Medium',
            'treatment': [
                'Apply chlorothalonil or copper fungicides',
                'Remove infected leaves',
                'Use fungicidal sprays every 7-10 days'
            ],
            'prevention': [
                'Practice crop rotation',
                'Use drip irrigation',
                'Apply mulch to prevent soil splash'
            ],
            'risk_factors': ['Warm, humid conditions', 'Wet foliage', 'Poor nutrition'],
            'confidence_threshold': 0.7
        },
        'Tomato___Late_blight': {
            'plant': 'Tomato',
            'disease': 'Late Blight',
            'scientific_name': 'Phytophthora infestans',
            'severity': 'High',
            'treatment': [
                'Remove and destroy infected plants immediately',
                'Apply copper-based fungicides weekly',
                'Use chlorothalonil or mancozeb as preventive sprays'
            ],
            'prevention': [
                'Plant resistant varieties when available',
                'Water at soil level to keep foliage dry',
                'Rotate crops regularly'
            ],
            'risk_factors': ['Cool, wet weather', 'High humidity', 'Poor air circulation'],
            'confidence_threshold': 0.7
        },
        'Tomato___Leaf_Mold': {
            'plant': 'Tomato',
            'disease': 'Leaf Mold',
            'scientific_name': 'Fulvia fulva',
            'severity': 'Medium',
            'treatment': [
                'Apply fungicides containing chlorothalonil',
                'Remove infected leaves',
                'Improve ventilation'
            ],
            'prevention': [
                'Plant resistant varieties',
                'Space plants for good air flow',
                'Avoid overhead watering'
            ],
            'risk_factors': ['High humidity', 'Poor air circulation'],
            'confidence_threshold': 0.7
        },
        'Tomato___Septoria_leaf_spot': {
            'plant': 'Tomato',
            'disease': 'Septoria Leaf Spot',
            'scientific_name': 'Septoria lycopersici',
            'severity': 'Medium',
            'treatment': [
                'Apply copper or chlorothalonil fungicides',
                'Remove infected leaves',
                'Keep foliage dry'
            ],
            'prevention': [
                'Practice crop rotation',
                'Remove plant debris',
                'Use drip irrigation'
            ],
            'risk_factors': ['Wet foliage', 'High humidity'],
            'confidence_threshold': 0.7
        },
        'Tomato___Spider_mites Two-spotted_spider_mite': {
            'plant': 'Tomato',
            'disease': 'Spider Mites',
            'scientific_name': 'Tetranychus urticae',
            'severity': 'Medium',
            'treatment': [
                'Apply insecticidal soap or neem oil',
                'Increase humidity around plants',
                'Use miticides if severe'
            ],
            'prevention': [
                'Regular monitoring of undersides of leaves',
                'Maintain plant vigor',
                'Avoid over-fertilizing with nitrogen'
            ],
            'risk_factors': ['Hot, dry conditions', 'Dusty environments'],
            'confidence_threshold': 0.7
        },
        'Tomato___Target_Spot': {
            'plant': 'Tomato',
            'disease': 'Target Spot',
            'scientific_name': 'Corynespora cassiicola',
            'severity': 'Medium',
            'treatment': [
                'Apply fungicides containing chlorothalonil',
                'Remove infected leaves',
                'Improve air circulation'
            ],
            'prevention': [
                'Plant resistant varieties',
                'Practice crop rotation',
                'Avoid overhead irrigation'
            ],
            'risk_factors': ['Warm, humid conditions'],
            'confidence_threshold': 0.7
        },
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
            'plant': 'Tomato',
            'disease': 'Yellow Leaf Curl Virus',
            'scientific_name': 'Tomato yellow leaf curl virus',
            'severity': 'High',
            'treatment': [
                'Remove and destroy infected plants',
                'Control whitefly vectors with insecticides',
                'Use reflective mulches'
            ],
            'prevention': [
                'Plant resistant varieties',
                'Use insect netting',
                'Remove weed hosts'
            ],
            'risk_factors': ['Whitefly presence', 'Warm temperatures'],
            'confidence_threshold': 0.7
        },
        'Tomato___Tomato_mosaic_virus': {
            'plant': 'Tomato',
            'disease': 'Mosaic Virus',
            'scientific_name': 'Tobacco mosaic virus',
            'severity': 'Medium',
            'treatment': [
                'Remove infected plants',
                'Disinfect tools after handling plants',
                'Avoid smoking near tomatoes'
            ],
            'prevention': [
                'Use virus-free seeds',
                'Practice good sanitation',
                'Control aphid vectors'
            ],
            'risk_factors': ['Mechanical transmission', 'Aphid presence'],
            'confidence_threshold': 0.7
        },
        'Tomato___healthy': {
            'plant': 'Tomato',
            'disease': 'Healthy',
            'scientific_name': 'N/A',
            'severity': 'None',
            'treatment': [
                'Regular watering at soil level',
                'Proper staking or caging',
                'Monitor for common pests'
            ],
            'prevention': [
                'Crop rotation every 2-3 years',
                'Proper spacing for air circulation',
                'Soil testing and amendment'
            ],
            'risk_factors': [],
            'confidence_threshold': 0.7
        }
    }
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for model prediction - MATCHES JUPYTER NOTEBOOK EXACTLY
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array
        """
        logger.info("=" * 60)
        logger.info(f" STARTING IMAGE PREPROCESSING")
        logger.info(f" Image: {image_path}")
        logger.info("=" * 60)
        
        logger.debug(f"Image file exists: {os.path.exists(image_path)}")
        if os.path.exists(image_path):
            logger.debug(f"Image file size: {os.path.getsize(image_path):,} bytes")
        
        try:
            logger.info("Step 1/3: Loading image with tf.keras.preprocessing.image.load_img()")
            logger.debug(f"Target size: (128, 128)")
            image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
            logger.info(f"Image loaded successfully")
            logger.debug(f"Image size: {image.size}")
            logger.debug(f"Image mode: {image.mode}")
            
            logger.info("Step 2/3: Converting to array with img_to_array()")
            logger.debug("⚠️  IMPORTANT: No normalization applied (matches Jupyter notebook)")
            input_arr = tf.keras.preprocessing.image.img_to_array(image)
            logger.info(f"Array conversion complete")
            
            # Log detailed array information
            logger.debug(f"Array shape: {input_arr.shape}")
            logger.debug(f"Array dtype: {input_arr.dtype}")
            logger.debug(f"Array min value: {input_arr.min():.2f}")
            logger.debug(f"Array max value: {input_arr.max():.2f}")
            logger.debug(f"Array mean: {input_arr.mean():.2f}")
            logger.debug(f"Array std: {input_arr.std():.2f}")
            
            # Log sample pixel values
            logger.debug("Sample pixel values (top-left corner, RGB):")
            for i in range(min(3, input_arr.shape[0])):
                for j in range(min(3, input_arr.shape[1])):
                    rgb = input_arr[i, j, :]
                    logger.debug(f"  Pixel [{i},{j}]: R={rgb[0]:.1f}, G={rgb[1]:.1f}, B={rgb[2]:.1f}")
            
            logger.info("Step 3/3: Creating batch dimension with np.array([input_arr])")
            logger.debug("IMPORTANT: Using np.array([input_arr]) exactly like Jupyter")
            input_arr = np.array([input_arr])
            
            logger.info("=" * 60)
            logger.info(" PREPROCESSING COMPLETE - SUMMARY:")
            logger.info("=" * 60)
            logger.info(f"  Final array shape: {input_arr.shape}")
            logger.info(f"  Final array dtype: {input_arr.dtype}")
            logger.info(f"  Value range: {input_arr.min():.1f} to {input_arr.max():.1f}")
            logger.info(f"  Expected by model: {self.model.input_shape if self.model else 'N/A'}")
            
            # Save preprocessing details to file
            self._save_preprocessing_details(image_path, input_arr)
            
            return input_arr
            
        except Exception as e:
            logger.error(f" PREPROCESSING FAILED: {e}")
            logger.exception("Preprocessing error traceback:")
            raise
    
    def _save_preprocessing_details(self, image_path: str, input_arr: np.ndarray):
        """Save preprocessing details to a file for analysis"""
        try:
            preprocess_log = os.path.join(settings.BASE_DIR, 'logs', 'preprocessing_details.txt')
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            with open(preprocess_log, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Preprocessing Details - {timestamp}\n")
                f.write(f"{'='*60}\n")
                f.write(f"Image: {image_path}\n")
                f.write(f"Final shape: {input_arr.shape}\n")
                f.write(f"Data type: {input_arr.dtype}\n")
                f.write(f"Min value: {input_arr.min():.2f}\n")
                f.write(f"Max value: {input_arr.max():.2f}\n")
                f.write(f"Mean: {input_arr.mean():.2f}\n")
                f.write(f"Std: {input_arr.std():.2f}\n")
                f.write(f"Sample pixels (first 3x3):\n")
                
                for i in range(min(3, input_arr.shape[1])):
                    for j in range(min(3, input_arr.shape[2])):
                        rgb = input_arr[0, i, j, :]
                        f.write(f"  [{i},{j}]: R={rgb[0]:6.1f}, G={rgb[1]:6.1f}, B={rgb[2]:6.1f}\n")
            
            logger.debug(f"Preprocessing details saved to: {preprocess_log}")
        except Exception as e:
            logger.warning(f"Could not save preprocessing details: {e}")
    
    def predict_from_path(self, image_path: str) -> Dict[str, Any]:
        """
        Predict disease from image path
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing prediction results
        """
        logger.info("=" * 80)
        logger.info(" STARTING PREDICTION PIPELINE")
        logger.info(f" Input: {image_path}")
        logger.info(f" Start time: {datetime.now().strftime('%H:%M:%S')}")
        logger.info("=" * 80)
        
        if self.model is None:
            logger.error(" CRITICAL: Model not loaded, cannot make prediction")
            return {"error": "Model not loaded", "success": False}
        
        try:
            # Step 1: Preprocessing
            logger.info(" STEP 1: IMAGE PREPROCESSING")
            input_arr = self.preprocess_image(image_path)
            
            # Step 2: Model Prediction
            logger.info(" STEP 2: MODEL PREDICTION")
            logger.debug(f"Input array shape for prediction: {input_arr.shape}")
            logger.debug(f"Calling model.predict()...")
            
            start_time = datetime.now()
            predictions = self.model.predict(input_arr, verbose=0)
            prediction_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f" Model prediction completed in {prediction_time:.3f} seconds")
            logger.debug(f"Raw predictions shape: {predictions.shape}")
            logger.debug(f"Predictions sum (should be ~1.0): {predictions.sum():.6f}")
            
            # Step 3: Process Results
            logger.info(" STEP 3: PROCESSING RESULTS")
            
            # Get all predictions sorted
            all_indices = np.argsort(predictions[0])[::-1]
            total_predictions = len(all_indices)
            
            # Log complete prediction table
            logger.info("=" * 80)
            logger.info(" COMPLETE PREDICTION DISTRIBUTION")
            logger.info("=" * 80)
            
            prediction_results = []
            for i, idx in enumerate(all_indices):
                confidence = float(predictions[0][idx])
                class_name = self.class_names[idx]
                percentage = confidence * 100
                
                prediction_results.append({
                    'rank': i + 1,
                    'class': class_name,
                    'confidence': confidence,
                    'percentage': percentage,
                    'is_top': i == 0
                })
                
                # Log detailed info for top 15 predictions
                if i < 15:
                    marker = "🏆" if i == 0 else "  "
                    logger.info(f"{marker} {i+1:2d}. {class_name:55s}: {confidence:.6f} ({percentage:6.2f}%)")
            
            # Log distribution summary
            logger.info("-" * 80)
            top_conf = prediction_results[0]['confidence']
            second_conf = prediction_results[1]['confidence'] if len(prediction_results) > 1 else 0
            confidence_gap = top_conf - second_conf
            
            logger.info(f" DISTRIBUTION SUMMARY:")
            logger.info(f"  Top confidence: {top_conf:.6f} ({top_conf*100:.2f}%)")
            logger.info(f"  2nd confidence: {second_conf:.6f} ({second_conf*100:.2f}%)")
            logger.info(f"  Confidence gap: {confidence_gap:.6f}")
            logger.info(f"  Total classes: {total_predictions}")
            
            # Get top 3 predictions for response
            top_indices = all_indices[:3]
            top_predictions = [
                {
                    "class_name": self.class_names[i],
                    "confidence": float(predictions[0][i]),
                    "percentage": float(predictions[0][i] * 100)
                }
                for i in top_indices
            ]
            
            # Extract information from top prediction
            top_prediction = top_predictions[0]
            class_name = top_prediction["class_name"]
            
            logger.info(" STEP 4: EXTRACTING INFORMATION")
            plant, disease = self._extract_plant_disease(class_name)
            is_healthy = "healthy" in class_name.lower()
            
            logger.info(f"  Top class: {class_name}")
            logger.info(f"  Extracted plant: '{plant}'")
            logger.info(f"  Extracted disease: '{disease}'")
            logger.info(f"  Is healthy: {is_healthy}")
            
            # Get additional info
            additional_info = {}
            if class_name in self.disease_info:
                additional_info = self.disease_info[class_name]
                logger.info(f"  Found disease info in database")
            else:
                logger.info(f"  No specific disease info in database")
            
            # Prepare response
            response = {
                "success": True,
                "predictions": top_predictions,
                "top_prediction": top_prediction,
                "plant": plant,
                "disease": disease if not is_healthy else "Healthy",
                "is_healthy": is_healthy,
                "additional_info": additional_info,
                "timestamp": tf.timestamp().numpy(),
                "prediction_time": prediction_time,
                "all_predictions": prediction_results[:10]  # Include top 10
            }
            
            # Final summary
            logger.info("=" * 80)
            logger.info("🎉 PREDICTION COMPLETE - FINAL RESULT")
            logger.info("=" * 80)
            logger.info(f"   FINAL DIAGNOSIS: {plant} - {disease}")
            logger.info(f"   CONFIDENCE: {top_prediction['percentage']:.2f}%")
            logger.info(f"   PREDICTION TIME: {prediction_time:.3f}s")
            logger.info(f"   STATUS: {'Healthy' if is_healthy else 'Diseased'}")
            logger.info(f"   TOP CLASS: {class_name}")
            
            # Save prediction results to file
            self._save_prediction_results(image_path, response, prediction_results)
            
            logger.info(f" Results saved to log file")
            logger.info("=" * 80)
            
            return response
            
        except Exception as e:
            logger.error(f" PREDICTION FAILED: {e}")
            logger.exception("Prediction error traceback:")
            return {"error": f"Prediction failed: {str(e)}", "success": False}
    
    def _save_prediction_results(self, image_path: str, response: Dict, all_predictions: list):
        """Save detailed prediction results to a file"""
        try:
            pred_log = os.path.join(settings.BASE_DIR, 'logs', 'prediction_results.csv')
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Check if file exists to write header
            write_header = not os.path.exists(pred_log)
            
            with open(pred_log, 'a', encoding='utf-8') as f:
                if write_header:
                    f.write("timestamp,image,plant,disease,confidence,is_healthy,top_class,top_5_classes\n")
                
                top_class = response['top_prediction']['class_name']
                confidence = response['top_prediction']['percentage']
                
                # Get top 5 classes as string
                top_5 = ";".join([p['class'] for p in all_predictions[:5]])
                
                f.write(f"{timestamp},{image_path},{response['plant']},{response['disease']},{confidence:.2f},{response['is_healthy']},{top_class},\"{top_5}\"\n")
            
            # Also save detailed results
            detail_log = os.path.join(settings.BASE_DIR, 'logs', f"prediction_detail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            with open(detail_log, 'w', encoding='utf-8') as f:
                f.write(f"Detailed Prediction Results\n")
                f.write(f"Generated: {timestamp}\n")
                f.write(f"Image: {image_path}\n")
                f.write(f"Plant: {response['plant']}\n")
                f.write(f"Disease: {response['disease']}\n")
                f.write(f"Confidence: {confidence:.2f}%\n")
                f.write(f"Healthy: {response['is_healthy']}\n")
                f.write("\nTop 20 Predictions:\n")
                f.write("-" * 80 + "\n")
                
                for pred in all_predictions[:20]:
                    f.write(f"{pred['rank']:2d}. {pred['class']:55s}: {pred['confidence']:.6f} ({pred['percentage']:6.2f}%)\n")
            
            logger.debug(f"Detailed results saved to: {detail_log}")
        except Exception as e:
            logger.warning(f"Could not save prediction results: {e}")
    
    def _extract_plant_disease(self, class_name: str) -> Tuple[str, str]:
        """
        Extract plant and disease names from class name
    
        Args:
            class_name: The full class name (e.g., 'Tomato___Late_blight')
        
        Returns:
        Tuple of (plant, disease)
        """
        logger.info(f"🔍 Extracting plant/disease from: {class_name}")
    
    
        for plant, classes in self.plant_type_mapping.items():
            if class_name in classes:
            # Extract disease name from class_name
              parts = class_name.split('___')
              if len(parts) == 2:
                  disease = parts[1].replace('_', ' ').strip()
                  logger.debug(f"Found in plant mapping: {plant} -> {disease}")
                  return plant, disease
    
    # Fallback: parse the class name
        if '___' in class_name:
            parts = class_name.split('___')
            plant = parts[0].replace('_', ' ').replace(',', '').strip()
        
            if len(parts) > 1:
                # Clean up disease name
                disease = parts[1].replace('_', ' ').strip()
            # Remove extra disease scientific names in parentheses
                if '(' in disease:
                    disease = disease.split('(')[0].strip()
            else:
                disease = "Unknown"
        else:
            plant = "Unknown"
            disease = "Unknown"
    
    # Clean up plant name
        if 'healthy' in disease.lower():
            disease = 'Healthy'
    
    # Special cases for specific plants
        if plant.lower() == 'pepper, bell':
            plant = 'Pepper'
        elif plant.lower() == 'cherry (including sour)':
            plant = 'Cherry'
        elif plant.lower() == 'corn (maize)':
            plant = 'Corn'
    
        logger.info(f"Extracted: Plant='{plant}', Disease='{disease}'")
        return plant, disease
    
    def predict_from_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Predict disease from image bytes
    
        Args:
            image_bytes: Image bytes
        
        Returns:
            Dictionary containing prediction results
        """
        logger.info(" STARTING PREDICTION FROM BYTES")
    
        try:
            # Save bytes to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                tmp_file.write(image_bytes)
                tmp_path = tmp_file.name
        
            logger.info(f"Saved bytes to temp file: {tmp_path}")
        
        # Use existing path prediction
            result = self.predict_from_path(tmp_path)
        
        # Clean up temp file
            os.unlink(tmp_path)
            logger.debug(f"Cleaned up temp file: {tmp_path}")
        
            return result
        
        except Exception as e:
            logger.error(f" Prediction from bytes failed: {e}")
            logger.exception("Prediction error traceback:")
            return {"error": f"Prediction failed: {str(e)}", "success": False}
        
    
    
    def test_with_jupyter_image(self, test_image_path: str = None):
        """
        Test function to verify the detector works exactly like Jupyter
        
        Args:
            test_image_path: Path to test image (use your corn rust image)
        """
        test_log_file = os.path.join(settings.BASE_DIR, 'logs', 'jupyter_comparison.log')
        
        # Create a separate logger for test
        test_logger = logging.getLogger('jupyter_test')
        test_handler = logging.FileHandler(test_log_file, encoding='utf-8')
        test_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S'))
        test_logger.addHandler(test_handler)
        test_logger.setLevel(logging.INFO)
        
        test_logger.info("=" * 80)
        test_logger.info(" JUPYTER NOTEBOOK COMPATIBILITY TEST")
        test_logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        test_logger.info("=" * 80)
        
        if test_image_path is None:
            test_image_path = os.path.join(settings.BASE_DIR, "model_train", "new_plant_dataset", "new_plant_dataset", "test", "test", "CornCommonRust2.JPG")
        
        test_logger.info(f"Test image: {test_image_path}")
        test_logger.info(f"Image exists: {os.path.exists(test_image_path)}")
        
        if not os.path.exists(test_image_path):
            test_logger.error(" Test image not found!")
            test_logger.info("Searching for image...")
            
            # Search for the image
            search_paths = [
                "CornCommonRust2.JPG",
                os.path.join("test", "CornCommonRust2.JPG"),
                os.path.join("test", "test", "CornCommonRust2.JPG"),
                os.path.join("model_train", "test", "CornCommonRust2.JPG"),
                os.path.join("model_train", "new_plant_dataset", "test", "CornCommonRust2.JPG"),
            ]
            
            for path in search_paths:
                full_path = os.path.join(settings.BASE_DIR, path)
                if os.path.exists(full_path):
                    test_image_path = full_path
                    test_logger.info(f"✅ Found image at: {test_image_path}")
                    break
        
        if not os.path.exists(test_image_path):
            test_logger.error(" Could not find test image")
            test_logger.info("=" * 80)
            test_logger.info(" TEST FAILED - Image not found")
            test_logger.info("=" * 80)
            return
        
        test_logger.info(f" Starting prediction test...")
        
        # Run prediction
        result = self.predict_from_path(test_image_path)
        
        if result.get("success"):
            test_logger.info("=" * 80)
            test_logger.info(" TEST RESULTS")
            test_logger.info("=" * 80)
            test_logger.info(f"Plant: {result['plant']}")
            test_logger.info(f"Disease: {result['disease']}")
            test_logger.info(f"Confidence: {result['top_prediction']['percentage']:.2f}%")
            test_logger.info(f"Healthy: {result['is_healthy']}")
            test_logger.info(f"Top class: {result['top_prediction']['class_name']}")
            
            # Jupyter expected result
            test_logger.info("\n JUPYTER NOTEBOOK EXPECTED:")
            test_logger.info("  Plant: Corn")
            test_logger.info("  Disease: Common rust")
            test_logger.info("  Confidence: High (>80%)")
            
            # Comparison
            expected_plant = "Corn"
            expected_disease = "Common rust"
            
            plant_match = result['plant'] == expected_plant
            disease_match = result['disease'] == expected_disease
            confidence_high = result['top_prediction']['percentage'] > 80
            
            test_logger.info("\n COMPARISON RESULTS:")
            test_logger.info(f"  Plant match: {'✅' if plant_match else '❌'} ({result['plant']} vs {expected_plant})")
            test_logger.info(f"  Disease match: {'✅' if disease_match else '❌'} ({result['disease']} vs {expected_disease})")
            test_logger.info(f"  Confidence high: {'✅' if confidence_high else '❌'} ({result['top_prediction']['percentage']:.2f}%)")
            
            if plant_match and disease_match and confidence_high:
                test_logger.info("\n🎉 PERFECT MATCH WITH JUPYTER NOTEBOOK!")
                test_logger.info("✅ Django implementation is working correctly")
            else:
                test_logger.info("\n⚠️  PARTIAL OR NO MATCH WITH JUPYTER")
                test_logger.info("❌ Check preprocessing differences")
                
                # Log differences
                test_logger.info("\n🔧 TROUBLESHOOTING:")
                if not plant_match:
                    test_logger.info(f"  - Plant extraction issue: Got '{result['plant']}', expected '{expected_plant}'")
                if not disease_match:
                    test_logger.info(f"  - Disease extraction issue: Got '{result['disease']}', expected '{expected_disease}'")
                if not confidence_high:
                    test_logger.info(f"  - Low confidence: {result['top_prediction']['percentage']:.2f}%")
                    test_logger.info(f"  - Check preprocessing normalization")
                
        else:
            test_logger.info("=" * 80)
            test_logger.info(" TEST FAILED")
            test_logger.info("=" * 80)
            test_logger.info(f"Error: {result.get('error', 'Unknown error')}")
        
        test_logger.info("=" * 80)
        test_logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        test_logger.info("Test log saved to: " + test_log_file)
        test_logger.info("=" * 80)
        
        # Also log to main logger
        logger.info(f"Jupyter compatibility test completed. See: {test_log_file}")
    
    def get_disease_info(self, class_name: str) -> Dict:
       """
       Get disease information for a specific class
    
       Args:
           class_name: The class name
        
       Returns:
           Disease information dictionary
       """
       return self.disease_info.get(class_name, {})
    
    def get_class_names(self) -> list[str]:
      """
      Get list of all class names

      Returns:
          List of class names
      """
      return self.class_names.copy()

    def get_plant_types(self) -> list[str]:
       """
          Get list of supported plant types
    
          Returns:
              List of plant type names
          """
       return list(self.plant_type_mapping.keys())

    def get_log_summary(self) -> Dict:
        """Get summary of all log files"""
        log_dir = os.path.join(settings.BASE_DIR, 'logs')
        
        if not os.path.exists(log_dir):
            return {"error": "Log directory not found"}
        
        log_files = []
        for file in os.listdir(log_dir):
            if file.endswith('.log') or file.endswith('.txt'):
                file_path = os.path.join(log_dir, file)
                stats = os.stat(file_path)
                log_files.append({
                    'name': file,
                    'size': stats.st_size,
                    'modified': datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                    'path': file_path
                })
        
        
        log_files.sort(key=lambda x: x['modified'], reverse=True)
        
        summary = {
            'log_directory': log_dir,
            'total_files': len(log_files),
            'files': log_files[:10], 
            'current_log': log_file
        }
        
        logger.info(f"Log summary: {len(log_files)} files in {log_dir}")
        return summary

