# urls.py
from django.urls import path
from . import views

app_name = 'ai_detection'

urlpatterns = [
    # Main disease detection page
    path('disease-detection/', views.disease_detection, name='disease_detection'),
    
    # API endpoints
    path('api/analyze/', views.analyze_plant_image, name='analyze_plant_image'),
    path('api/plant-types/', views.get_plant_types, name='get_plant_types'),
    path('api/diseases/<str:plant_type>/', views.get_diseases_for_plant, name='get_diseases_for_plant'),
]