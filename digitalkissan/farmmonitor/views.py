from django.shortcuts import render
from django.utils import timezone

def dashboard(request):
    context = {
        'page_title': 'Farm Monitoring Dashboard',
        'active_page': 'monitoring',
        'farm_data': {
            'total_fields': 4,
            'total_area': '4.2 ha',
            'crops': ['Wheat', 'Corn', 'Tomatoes', 'Potatoes'],
            'last_updated': timezone.now()
        }
    }
    return render(request, "farmiot/dashboard.html",context)
