
from django.urls import path
from . import views

app_name = 'ai_detection'

urlpatterns = [
    path('disease_detection',views.disease_detection,name='disease_detection'),
]
