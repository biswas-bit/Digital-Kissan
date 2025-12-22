# views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os
import json
from .ai_model import DiseaseDetector 

# Initialize the detector once when the module loads
detector = DiseaseDetector()

def disease_detection(request):
    """Render the disease detection page"""
    return render(request, "AI/disease_detection.html")  # Fixed template path

@csrf_exempt
def analyze_plant_image(request):
    """
    API endpoint to analyze uploaded plant image
    Accepts POST request with image file
    Returns JSON with disease detection results
    """
    if request.method != 'POST':
        return JsonResponse({
            'success': False,
            'error': 'Only POST method is allowed'
        }, status=405)
    
    if 'image' not in request.FILES:
        return JsonResponse({
            'success': False,
            'error': 'No image file provided'
        }, status=400)
    
    try:
        # Get the uploaded file
        image_file = request.FILES['image']
        
        # Validate file type
        allowed_extensions = ['jpg', 'jpeg', 'png', 'webp']
        file_extension = image_file.name.split('.')[-1].lower()
        
        if file_extension not in allowed_extensions:
            return JsonResponse({
                'success': False,
                'error': f'Invalid file type. Allowed types: {", ".join(allowed_extensions)}'
            }, status=400)
        
        # Validate file size (max 10MB)
        max_size = 10 * 1024 * 1024  # 10MB in bytes
        if image_file.size > max_size:
            return JsonResponse({
                'success': False,
                'error': 'File size exceeds 10MB limit'
            }, status=400)
        
        # Save the uploaded file temporarily
        file_name = default_storage.save(
            f'temp/{image_file.name}',
            ContentFile(image_file.read())
        )
        file_path = default_storage.path(file_name)
        
        try:
            # Perform prediction
            result = detector.predict_from_path(file_path)
            
            if not result.get('success', False):
                return JsonResponse({
                    'success': False,
                    'error': result.get('error', 'Prediction failed')
                }, status=500)
            
            # Generate comprehensive report
            report = detector.generate_report(result)
            
            # Prepare response data matching frontend expectations
            response_data = {
                'success': True,
                'plant': result.get('plant', 'Unknown'),
                'disease': result.get('disease', 'Unknown'),
                'is_healthy': result.get('is_healthy', True),
                'confidence': result.get('top_prediction', {}).get('confidence', 0),
                'confidence_percentage': result.get('top_prediction', {}).get('percentage', 0),
                'severity': report.get('severity', 'Low'),
                'predictions': result.get('predictions', []),
                'recommendations': report.get('recommendations', {}),
                'additional_info': result.get('additional_info', {})
            }
            
            print(f"Response data: {json.dumps(response_data, indent=2)}")  # For debugging
            return JsonResponse(response_data)
            
        finally:
            # Clean up: delete temporary file
            if os.path.exists(file_path):
                default_storage.delete(file_name)
    
    except Exception as e:
        import traceback
        print(f"Error in analyze_plant_image: {str(e)}")
        print(traceback.format_exc())
        return JsonResponse({
            'success': False,
            'error': f'Server error: {str(e)}'
        }, status=500)

def get_plant_types(request):
    """
    API endpoint to get list of supported plant types
    """
    try:
        plant_types = detector.get_plant_types()
        return JsonResponse({
            'success': True,
            'plant_types': plant_types
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

def get_diseases_for_plant(request, plant_type):
    """
    API endpoint to get diseases for a specific plant type
    """
    try:
        diseases = detector.get_diseases_for_plant(plant_type)
        return JsonResponse({
            'success': True,
            'plant_type': plant_type,
            'diseases': diseases
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)