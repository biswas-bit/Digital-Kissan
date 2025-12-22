from . import views
from django.urls import path

app_name = 'farmmonitor'
urlpatterns = [
    path('',views.dashboard, name='dashboard'),
]
