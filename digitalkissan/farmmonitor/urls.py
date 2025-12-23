# ai_detection/urls.py
from django.urls import path
from . import views
app_name = 'farmmonitor'
urlpatterns = [
    # Dashboard page
    path('dashboard/', views.dashboard_view, name='dashboard'),
    
    # API endpoints
    path('api/dashboard-data/', views.DashboardDataAPI.as_view(), name='dashboard_data'),
    path('api/trigger-action/', views.TriggerActionAPI.as_view(), name='trigger_action'),
    path('api/historical-data/', views.HistoricalDataAPI.as_view(), name='historical_data'),
    path('api/field-data/<str:field_name>/', views.FieldDataAPI.as_view(), name='field_data'),
]