# ai_detection/views.py
import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
from datetime import datetime, timedelta
import random
import numpy as np

# Try to import simulators
try:
    from .services import (
        weather_simulator, 
        soil_simulator, 
        crop_simulator, 
        rainfall_simulator
    )
    SIMULATORS_AVAILABLE = True
    print("✅ Simulators imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("⚠️ Using fallback simulators")
    SIMULATORS_AVAILABLE = False
    
    # Create fallback simulators
    class FallbackSimulator:
        def get_current_weather(self):
            return {
                'temperature': 24 + random.uniform(-2, 2),
                'humidity': 65 + random.uniform(-5, 5),
                'condition': random.choice(['Clear', 'Partly Cloudy', 'Cloudy', 'Light Rain']),
                'wind_speed': 12 + random.uniform(-3, 3),
                'forecast': [
                    {'day': 'Mon', 'temp': 26, 'condition': 'Clear'},
                    {'day': 'Tue', 'temp': 28, 'condition': 'Partly Cloudy'},
                    {'day': 'Wed', 'temp': 25, 'condition': 'Cloudy'}
                ],
                'timestamp': datetime.now()
            }
        
        def get_soil_data(self, field_name=None):
            if field_name and field_name in ['North Field', 'South Field', 'East Field', 'West Field']:
                base_moisture = {'North Field': 72, 'South Field': 42, 'East Field': 65, 'West Field': 78}[field_name]
                return {
                    'moisture': base_moisture + random.uniform(-3, 3),
                    'temperature': 22 + random.uniform(-2, 2),
                    'ph': 6.8 + random.uniform(-0.3, 0.3),
                    'nitrogen': 150 + random.uniform(-10, 10),
                    'phosphorus': 45 + random.uniform(-5, 5),
                    'potassium': 180 + random.uniform(-15, 15),
                    'timestamp': datetime.now()
                }
            
            # Return data for all fields
            return {
                'overall': {
                    'moisture': 68 + random.uniform(-3, 3),
                    'temperature': 22.5 + random.uniform(-1, 1),
                    'ph_level': 6.8 + random.uniform(-0.2, 0.2),
                    'nitrogen': 152 + random.uniform(-10, 10),
                    'phosphorus': 42 + random.uniform(-5, 5),
                    'potassium': 185 + random.uniform(-15, 15)
                },
                'by_field': {
                    'North Field': {
                        'moisture': 72 + random.uniform(-3, 3),
                        'ph_level': 6.5 + random.uniform(-0.2, 0.2),
                        'temperature': 21.8 + random.uniform(-1, 1)
                    },
                    'South Field': {
                        'moisture': 42 + random.uniform(-3, 3),
                        'ph_level': 6.2 + random.uniform(-0.2, 0.2),
                        'temperature': 23.1 + random.uniform(-1, 1)
                    },
                    'East Field': {
                        'moisture': 65 + random.uniform(-3, 3),
                        'ph_level': 7.1 + random.uniform(-0.2, 0.2),
                        'temperature': 22.3 + random.uniform(-1, 1)
                    },
                    'West Field': {
                        'moisture': 78 + random.uniform(-3, 3),
                        'ph_level': 6.8 + random.uniform(-0.2, 0.2),
                        'temperature': 21.5 + random.uniform(-1, 1)
                    }
                }
            }
        
        def simulate_irrigation(self, field_name):
            return {'status': 'irrigation_simulated', 'field': field_name, 'timestamp': datetime.now()}
        
        def get_crop_health(self, crop_name, soil_data=None, weather_data=None):
            base_scores = {
                'Tomatoes': 92,
                'Corn': 85,
                'Wheat': 88,
                'Potatoes': 82
            }
            base_score = base_scores.get(crop_name, 85)
            
            # Add some variation
            variation = random.uniform(-3, 3)
            
            return {
                'health_score': max(0, min(100, base_score + variation)),
                'growth_stage': 'Flowering' if crop_name == 'Tomatoes' else 
                               'Tasseling' if crop_name == 'Corn' else 
                               'Heading' if crop_name == 'Wheat' else 'Tuber Bulking',
                'stage_progress': 0.65 if crop_name == 'Tomatoes' else 
                                 0.75 if crop_name == 'Corn' else 
                                 0.8 if crop_name == 'Wheat' else 0.6,
                'days_since_planting': 45 if crop_name == 'Tomatoes' else 
                                      60 if crop_name == 'Corn' else 
                                      75 if crop_name == 'Wheat' else 50,
                'pest_pressure': round(random.uniform(1, 5), 1),
                'disease_risk': round(random.uniform(1, 4), 1),
                'last_updated': datetime.now()
            }
        
        def get_growth_progress(self, crop_name):
            base_data = [
                {'week': 'Week 1', 'actual': 15, 'expected': 10},
                {'week': 'Week 2', 'actual': 30, 'expected': 25},
                {'week': 'Week 3', 'actual': 50, 'expected': 45},
                {'week': 'Week 4', 'actual': 65, 'expected': 60},
                {'week': 'Week 5', 'actual': 78, 'expected': 75},
                {'week': 'Week 6', 'actual': 85, 'expected': 85},
                {'week': 'Week 7', 'actual': 92, 'expected': 92},
                {'week': 'Week 8', 'actual': 96, 'expected': 96}
            ]
            
            # Add some variation
            for item in base_data:
                if item['actual']:
                    item['actual'] += random.uniform(-2, 2)
                    item['actual'] = max(0, min(100, item['actual']))
            
            return base_data
        
        def get_rainfall_predictions(self, days=7):
            days_list = ['Today', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
            predictions = []
            
            for i in range(days):
                rainfall = random.uniform(0, 20)
                if rainfall < 2:
                    status = 'Light'
                    status_class = 'good'
                elif rainfall < 10:
                    status = 'Moderate'
                    status_class = 'warning'
                else:
                    status = 'Heavy'
                    status_class = 'critical'
                
                predictions.append({
                    'day': days_list[i],
                    'date': (datetime.now() + timedelta(days=i)).strftime('%d %b'),
                    'rainfall': round(rainfall, 1),
                    'probability': random.randint(10, 95),
                    'status': status,
                    'status_class': status_class
                })
            
            return predictions
    
    # Create global instances
    weather_simulator = FallbackSimulator()
    soil_simulator = FallbackSimulator()
    crop_simulator = FallbackSimulator()
    rainfall_simulator = FallbackSimulator()

# =============================================================================
# VIEW FUNCTIONS
# =============================================================================

def dashboard_view(request):
    """Render the main dashboard page"""
    context = {
        'simulators_available': SIMULATORS_AVAILABLE,
        'current_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'page_title': 'Farm Monitoring Dashboard',
        'api_endpoints': {
            'dashboard_data': '/farmmonitor/api/dashboard-data/',
            'trigger_action': '/farmmonitor/api/trigger-action/',
        }
    }
    return render(request, 'farmiot/dashboard.html', context)

# =============================================================================
# API VIEWS
# =============================================================================

class DashboardDataAPI(View):
    """API endpoint to get dashboard data"""
    
    def get(self, request):
        try:
            period = request.GET.get('period', 'today')
            print(f"📊 Generating dashboard data for period: {period}")
            
            data = self._generate_dashboard_data(period)
            
            return JsonResponse({
                'success': True,
                'data': data,
                'simulators_available': SIMULATORS_AVAILABLE,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            print(f"❌ Error in DashboardDataAPI: {e}")
            return JsonResponse({
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }, status=500)
    
    def _generate_dashboard_data(self, period):
        """Generate comprehensive dashboard data"""
        try:
            # Get current weather data
            weather_data = weather_simulator.get_current_weather()
            
            # Get soil data for all fields
            soil_data = soil_simulator.get_soil_data()
            
            # Get crop health data for all crops
            crop_health_data = {}
            for crop_name in ['Tomatoes', 'Corn', 'Wheat', 'Potatoes']:
                crop_health = crop_simulator.get_crop_health(
                    crop_name, 
                    soil_data.get('overall', {}),
                    weather_data
                )
                crop_health_data[crop_name] = crop_health
            
            # Get rainfall predictions
            rainfall_predictions = rainfall_simulator.get_rainfall_predictions(7)
            
            # Generate KPIs
            kpis = self._calculate_kpis(soil_data, crop_health_data, weather_data)
            
            # Generate alerts
            alerts = self._generate_alerts(soil_data, crop_health_data, weather_data, rainfall_predictions)
            
            # Generate crop health distribution
            crop_distribution = self._generate_crop_distribution(crop_health_data)
            
            # Generate growth progress data
            growth_progress = crop_simulator.get_growth_progress('Tomatoes')
            
            # Prepare final data
            result = {
                'weather': {
                    'current_temp': weather_data.get('temperature', 24),
                    'humidity': weather_data.get('humidity', 65),
                    'condition': weather_data.get('condition', 'Partly Cloudy'),
                    'wind_speed': weather_data.get('wind_speed', 12),
                    'forecast': weather_data.get('forecast', []),
                    'location': 'Digital Kisan Farm',
                    'timestamp': weather_data.get('timestamp', datetime.now()).isoformat()
                },
                'soil': {
                    'overall': soil_data.get('overall', {}),
                    'by_field': soil_data.get('by_field', {})
                },
                'crops': {
                    'health_data': crop_health_data,
                    'health_distribution': crop_distribution,
                    'growth_progress': growth_progress
                },
                'rainfall': {
                    'predictions': rainfall_predictions
                },
                'kpis': kpis,
                'alerts': alerts,
                'period': period,
                'timestamp': datetime.now().isoformat(),
                'simulator_status': 'active' if SIMULATORS_AVAILABLE else 'fallback'
            }
            
            print(f"✅ Generated dashboard data successfully")
            return result
            
        except Exception as e:
            print(f"❌ Error generating dashboard data: {e}")
            return self._get_fallback_full_data()
    
    def _calculate_kpis(self, soil_data, crop_health_data, weather_data):
        """Calculate key performance indicators"""
        try:
            soil_overall = soil_data.get('overall', {})
            
            # Average crop health
            crop_health_values = [data['health_score'] for data in crop_health_data.values()]
            avg_crop_health = sum(crop_health_values) / len(crop_health_values) if crop_health_values else 0
            
            # Yield prediction based on multiple factors
            yield_base = 8.2
            weather_factor = 1.0
            condition = weather_data.get('condition', 'Clear')
            if condition in ['Clear', 'Partly Cloudy']:
                weather_factor = 1.1
            elif condition in ['Heavy Rain', 'Thunderstorm']:
                weather_factor = 0.9
            
            soil_factor = 1.0
            moisture = soil_overall.get('moisture', 65)
            if moisture > 70:
                soil_factor = 1.05
            elif moisture < 50:
                soil_factor = 0.95
            
            yield_prediction = round(yield_base * weather_factor * soil_factor, 1)
            
            # Water usage (simulated)
            base_water = 4250
            moisture_factor = moisture / 65
            water_usage = int(base_water * (1 + (1 - moisture_factor) * 0.3))
            
            # Irrigation efficiency
            irrigation_efficiency = 85 + random.uniform(-5, 5)
            
            return {
                'soil_moisture': round(moisture, 1),
                'crop_health': round(avg_crop_health, 1),
                'yield_prediction': yield_prediction,
                'water_usage': water_usage,
                'irrigation_efficiency': round(irrigation_efficiency, 1)
            }
        except Exception as e:
            print(f"❌ Error calculating KPIs: {e}")
            return {
                'soil_moisture': 68,
                'crop_health': 85,
                'yield_prediction': 8.2,
                'water_usage': 4250,
                'irrigation_efficiency': 85
            }
    
    def _generate_alerts(self, soil_data, crop_health_data, weather_data, rainfall_predictions):
        """Generate dynamic alerts based on current conditions"""
        alerts = []
        now = datetime.now()
        
        try:
            # Check soil moisture alerts
            for field_name, field_data in soil_data.get('by_field', {}).items():
                moisture = field_data.get('moisture', 0)
                if moisture < 45:
                    alerts.append({
                        'type': 'warning',
                        'message': f'Irrigation required in {field_name} - Soil moisture at {moisture:.1f}%',
                        'icon': 'droplet',
                        'priority': 'high',
                        'timestamp': now.isoformat()
                    })
                elif moisture > 80:
                    alerts.append({
                        'type': 'warning',
                        'message': f'Excessive moisture in {field_name} - Consider drainage',
                        'icon': 'droplet',
                        'priority': 'medium',
                        'timestamp': now.isoformat()
                    })
            
            # Check crop health alerts
            for crop_name, crop_data in crop_health_data.items():
                health_score = crop_data.get('health_score', 0)
                if health_score < 60:
                    alerts.append({
                        'type': 'critical',
                        'message': f'{crop_name} health critical ({health_score:.1f}%) - Immediate attention needed',
                        'icon': 'alert-triangle',
                        'priority': 'critical',
                        'timestamp': now.isoformat()
                    })
                elif health_score < 75:
                    alerts.append({
                        'type': 'warning',
                        'message': f'{crop_name} health low ({health_score:.1f}%) - Monitor closely',
                        'icon': 'alert-circle',
                        'priority': 'medium',
                        'timestamp': now.isoformat()
                    })
            
            # Check weather alerts
            condition = weather_data.get('condition', '')
            if condition in ['Heavy Rain', 'Thunderstorm']:
                alerts.append({
                    'type': 'critical',
                    'message': f'Weather alert: {condition} detected. Take necessary precautions.',
                    'icon': 'cloud-rain',
                    'priority': 'critical',
                    'timestamp': now.isoformat()
                })
            
            # Check rainfall predictions for heavy rain
            for prediction in rainfall_predictions[:3]:  # Check next 3 days
                if prediction.get('rainfall', 0) > 10 and prediction.get('probability', 0) > 70:
                    alerts.append({
                        'type': 'critical',
                        'message': f'Heavy rainfall predicted ({prediction["rainfall"]:.1f}mm, {prediction["probability"]}% probability) in 3 days - Prepare drainage',
                        'icon': 'cloud-rain',
                        'priority': 'high',
                        'timestamp': now.isoformat()
                    })
                    break
            
            # Check for optimal fertilizer application window
            if condition in ['Clear', 'Partly Cloudy'] and 18 <= now.hour <= 22:
                alerts.append({
                    'type': 'info',
                    'message': 'Optimal time for fertilizer application - Weather window: Next 48 hours',
                    'icon': 'sunrise',
                    'priority': 'low',
                    'timestamp': now.isoformat()
                })
            
            # If no alerts, add an informational one
            if not alerts:
                alerts.append({
                    'type': 'info',
                    'message': 'All systems operating normally. No alerts at this time.',
                    'icon': 'check-circle',
                    'priority': 'low',
                    'timestamp': now.isoformat()
                })
            
            # Sort alerts by priority
            priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
            alerts.sort(key=lambda x: priority_order.get(x.get('priority', 'low'), 4))
            
            return alerts[:5]  # Return top 5 alerts
            
        except Exception as e:
            print(f"❌ Error generating alerts: {e}")
            return [{
                'type': 'info',
                'message': 'Alert system initializing...',
                'icon': 'info',
                'priority': 'low',
                'timestamp': now.isoformat()
            }]
    
    def _generate_crop_distribution(self, crop_health_data):
        """Generate crop health distribution data"""
        try:
            distribution = []
            
            for crop_name, health_data in crop_health_data.items():
                distribution.append({
                    'field': crop_name,
                    'health': health_data.get('health_score', 0),
                    'stage': health_data.get('growth_stage', 'Unknown'),
                    'progress': health_data.get('stage_progress', 0)
                })
            
            return distribution
        except Exception as e:
            print(f"❌ Error generating crop distribution: {e}")
            return []
    
    def _get_fallback_full_data(self):
        """Return complete fallback data when everything fails"""
        now = datetime.now()
        
        return {
            'weather': {
                'current_temp': 24,
                'humidity': 65,
                'condition': 'Partly Cloudy',
                'wind_speed': 12,
                'forecast': [
                    {'day': 'Mon', 'temp': 26, 'condition': 'Clear'},
                    {'day': 'Tue', 'temp': 28, 'condition': 'Partly Cloudy'},
                    {'day': 'Wed', 'temp': 25, 'condition': 'Cloudy'}
                ],
                'location': 'Digital Kisan Farm',
                'timestamp': now.isoformat()
            },
            'soil': {
                'overall': {
                    'moisture': 68,
                    'temperature': 22.5,
                    'ph_level': 6.8,
                    'nitrogen': 152,
                    'phosphorus': 42,
                    'potassium': 185
                },
                'by_field': {
                    'North Field': {'moisture': 72, 'ph_level': 6.5, 'temperature': 21.8},
                    'South Field': {'moisture': 42, 'ph_level': 6.2, 'temperature': 23.1},
                    'East Field': {'moisture': 65, 'ph_level': 7.1, 'temperature': 22.3},
                    'West Field': {'moisture': 78, 'ph_level': 6.8, 'temperature': 21.5}
                }
            },
            'crops': {
                'health_data': {
                    'Tomatoes': {'health_score': 92, 'growth_stage': 'Flowering', 'stage_progress': 0.65},
                    'Corn': {'health_score': 85, 'growth_stage': 'Tasseling', 'stage_progress': 0.75},
                    'Wheat': {'health_score': 88, 'growth_stage': 'Heading', 'stage_progress': 0.8},
                    'Potatoes': {'health_score': 82, 'growth_stage': 'Tuber Bulking', 'stage_progress': 0.6}
                },
                'health_distribution': [
                    {'field': 'Tomatoes', 'health': 92, 'stage': 'Flowering', 'progress': 0.65},
                    {'field': 'Corn', 'health': 85, 'stage': 'Tasseling', 'progress': 0.75},
                    {'field': 'Wheat', 'health': 88, 'stage': 'Heading', 'progress': 0.8},
                    {'field': 'Potatoes', 'health': 82, 'stage': 'Tuber Bulking', 'progress': 0.6}
                ],
                'growth_progress': [
                    {'week': 'Week 1', 'actual': 15, 'expected': 10},
                    {'week': 'Week 2', 'actual': 30, 'expected': 25},
                    {'week': 'Week 3', 'actual': 50, 'expected': 45},
                    {'week': 'Week 4', 'actual': 65, 'expected': 60},
                    {'week': 'Week 5', 'actual': 78, 'expected': 75},
                    {'week': 'Week 6', 'actual': 85, 'expected': 85},
                    {'week': 'Week 7', 'actual': 92, 'expected': 92},
                    {'week': 'Week 8', 'actual': 96, 'expected': 96}
                ]
            },
            'rainfall': {
                'predictions': [
                    {'day': 'Today', 'rainfall': 2.4, 'probability': 40, 'status': 'Light', 'status_class': 'good'},
                    {'day': 'Tue', 'rainfall': 8.7, 'probability': 75, 'status': 'Moderate', 'status_class': 'warning'},
                    {'day': 'Wed', 'rainfall': 15.2, 'probability': 90, 'status': 'Heavy', 'status_class': 'critical'}
                ]
            },
            'kpis': {
                'soil_moisture': 68,
                'crop_health': 85,
                'yield_prediction': 8.2,
                'water_usage': 4250,
                'irrigation_efficiency': 85
            },
            'alerts': [
                {
                    'type': 'info',
                    'message': 'Using fallback data - System initializing',
                    'icon': 'info',
                    'priority': 'low',
                    'timestamp': now.isoformat()
                }
            ],
            'period': 'today',
            'timestamp': now.isoformat(),
            'simulator_status': 'fallback'
        }

# =============================================================================
# ACTION API
# =============================================================================

@method_decorator(csrf_exempt, name='dispatch')
class TriggerActionAPI(View):
    """API endpoint to trigger farm actions"""
    
    def post(self, request):
        try:
            data = json.loads(request.body)
            action = data.get('action')
            
            if not action:
                return JsonResponse({
                    'success': False,
                    'error': 'No action specified'
                }, status=400)
            
            print(f"⚡ Triggering action: {action}")
            
            # Simulate different actions
            result = self._simulate_action(action)
            
            return JsonResponse({
                'success': True,
                'message': f'Action "{action}" executed successfully',
                'result': result,
                'timestamp': datetime.now().isoformat()
            })
            
        except json.JSONDecodeError:
            return JsonResponse({
                'success': False,
                'error': 'Invalid JSON data'
            }, status=400)
        except Exception as e:
            print(f"❌ Error in TriggerActionAPI: {e}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
    
    def _simulate_action(self, action):
        """Simulate farm actions"""
        actions = {
            'irrigation': {
                'simulate': lambda: soil_simulator.simulate_irrigation('North Field'),
                'message': 'Irrigation scheduled for North Field'
            },
            'soil_test': {
                'simulate': lambda: soil_simulator.get_soil_data('North Field'),
                'message': 'Soil test scheduled. Results will be available in 24 hours.'
            },
            'generate_report': {
                'simulate': lambda: self._generate_sample_report(),
                'message': 'Farm report generated successfully'
            },
            'export_data': {
                'simulate': lambda: {'status': 'data_exported', 'timestamp': datetime.now().isoformat()},
                'message': 'Data exported successfully'
            }
        }
        
        if action in actions:
            action_data = actions[action]
            result = action_data['simulate']()
            return {
                'action': action,
                'timestamp': datetime.now().isoformat(),
                'result': result,
                'message': action_data['message']
            }
        
        return {
            'error': 'Unknown action',
            'available_actions': list(actions.keys())
        }

    def _generate_sample_report(self):
        """Generate a sample farm report"""
        now = datetime.now()
        return {
            'report_id': f'FR-{now.strftime("%Y%m%d-%H%M%S")}',
            'generated_at': now.isoformat(),
            'title': 'Farm Monitoring Report',
            'sections': ['Weather Analysis', 'Soil Health', 'Crop Status', 'Recommendations'],
            'summary': 'Comprehensive farm analysis report',
            'download_url': f'/reports/FR-{now.strftime("%Y%m%d-%H%M%S")}.pdf',
            'status': 'generated'
        }

# =============================================================================
# HISTORICAL DATA API
# =============================================================================

class HistoricalDataAPI(View):
    """API endpoint for historical data"""
    
    def get(self, request):
        try:
            metric = request.GET.get('metric', 'temperature')
            days = int(request.GET.get('days', 7))
            
            print(f"📈 Generating historical data for {metric} ({days} days)")
            
            data = self._generate_historical_data(metric, days)
            
            return JsonResponse({
                'success': True,
                'metric': metric,
                'days': days,
                'data': data,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            print(f"❌ Error in HistoricalDataAPI: {e}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
    
    def _generate_historical_data(self, metric, days):
        """Generate historical data for charts"""
        data = []
        current_date = datetime.now()
        
        for i in range(days):
            date = current_date - timedelta(days=i)
            
            # Generate realistic historical data based on metric
            if metric == 'temperature':
                base_temp = 24
                daily_variation = np.sin(i * np.pi / 7) * 5
                random_variation = random.uniform(-2, 2)
                value = base_temp + daily_variation + random_variation
                
            elif metric == 'rainfall':
                value = random.uniform(0, 15) if random.random() > 0.6 else 0
                
            elif metric == 'soil_moisture':
                base_moisture = 65
                variation = random.uniform(-10, 5)
                value = max(30, min(90, base_moisture + variation))
                
            elif metric == 'crop_health':
                base_health = 85
                trend = i * 0.5  # Slight upward trend
                variation = random.uniform(-3, 3)
                value = base_health + trend + variation
                
            else:
                value = random.uniform(0, 100)
            
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'value': round(value, 2),
                'day': date.strftime('%a'),
                'timestamp': date.isoformat()
            })
        
        return list(reversed(data))  # Oldest to newest

# =============================================================================
# FIELD DATA API
# =============================================================================

class FieldDataAPI(View):
    """API endpoint for field-specific data"""
    
    def get(self, request, field_name):
        try:
            print(f"🌾 Getting data for field: {field_name}")
            
            # Get soil data for specific field
            soil_data = soil_simulator.get_soil_data(field_name)
            
            # Get weather data
            weather_data = weather_simulator.get_current_weather()
            
            # Generate field-specific insights
            insights = self._generate_field_insights(field_name, soil_data, weather_data)
            
            return JsonResponse({
                'success': True,
                'field': field_name,
                'soil_data': soil_data,
                'weather_data': weather_data,
                'insights': insights,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            print(f"❌ Error in FieldDataAPI: {e}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
    
    def _generate_field_insights(self, field_name, soil_data, weather_data):
        """Generate insights for a specific field"""
        insights = []
        
        moisture = soil_data.get('moisture', 0)
        ph = soil_data.get('ph', 0)
        temperature = soil_data.get('temperature', 0)
        
        try:
            # Moisture insights
            if moisture < 45:
                insights.append({
                    'type': 'warning',
                    'title': 'Low Soil Moisture',
                    'description': f'Soil moisture ({moisture:.1f}%) is below optimal range (45-70%). Consider irrigation.',
                    'action': 'schedule_irrigation',
                    'priority': 'high'
                })
            elif moisture > 75:
                insights.append({
                    'type': 'warning',
                    'title': 'High Soil Moisture',
                    'description': f'Soil moisture ({moisture:.1f}%) is above optimal range. Monitor for waterlogging.',
                    'action': 'check_drainage',
                    'priority': 'medium'
                })
            elif 60 <= moisture <= 70:
                insights.append({
                    'type': 'info',
                    'title': 'Optimal Moisture',
                    'description': f'Soil moisture ({moisture:.1f}%) is within optimal range.',
                    'action': 'monitor',
                    'priority': 'low'
                })
            
            # pH insights
            if ph < 6.0:
                insights.append({
                    'type': 'info',
                    'title': 'Acidic Soil',
                    'description': f'pH level ({ph:.1f}) is slightly acidic. Consider adding lime for optimal crop growth.',
                    'action': 'adjust_ph',
                    'priority': 'medium'
                })
            elif ph > 7.5:
                insights.append({
                    'type': 'info',
                    'title': 'Alkaline Soil',
                    'description': f'pH level ({ph:.1f}) is alkaline. Consider adding sulfur or acidifying fertilizers.',
                    'action': 'adjust_ph',
                    'priority': 'medium'
                })
            elif 6.0 <= ph <= 7.0:
                insights.append({
                    'type': 'info',
                    'title': 'Optimal pH',
                    'description': f'pH level ({ph:.1f}) is within optimal range (6.0-7.0).',
                    'action': 'maintain',
                    'priority': 'low'
                })
            
            # Temperature insights
            if temperature < 18:
                insights.append({
                    'type': 'info',
                    'title': 'Low Soil Temperature',
                    'description': f'Soil temperature ({temperature:.1f}°C) may slow seed germination and root growth.',
                    'action': 'monitor_growth',
                    'priority': 'low'
                })
            elif temperature > 30:
                insights.append({
                    'type': 'warning',
                    'title': 'High Soil Temperature',
                    'description': f'Soil temperature ({temperature:.1f}°C) may stress plants. Ensure adequate watering.',
                    'action': 'increase_irrigation',
                    'priority': 'medium'
                })
            elif 20 <= temperature <= 28:
                insights.append({
                    'type': 'info',
                    'title': 'Optimal Temperature',
                    'description': f'Soil temperature ({temperature:.1f}°C) is within optimal range for most crops.',
                    'action': 'monitor',
                    'priority': 'low'
                })
            
            # Weather-based insights
            condition = weather_data.get('condition', '')
            if condition == 'Heavy Rain' and moisture > 60:
                insights.append({
                    'type': 'critical',
                    'title': 'Rainfall Warning',
                    'description': 'Heavy rain predicted with already moist soil. Prepare drainage systems.',
                    'action': 'prepare_drainage',
                    'priority': 'high'
                })
            
            # If no insights, add a general one
            if not insights:
                insights.append({
                    'type': 'info',
                    'title': 'Field Status',
                    'description': f'{field_name} is in good condition. No immediate actions required.',
                    'action': 'monitor',
                    'priority': 'low'
                })
            
            return insights
            
        except Exception as e:
            print(f"❌ Error generating field insights: {e}")
            return [{
                'type': 'info',
                'title': 'Field Analysis',
                'description': f'Analyzing {field_name} conditions...',
                'action': 'monitor',
                'priority': 'low'
            }]

# =============================================================================
# HEALTH CHECK API
# =============================================================================

class HealthCheckAPI(View):
    """API endpoint for system health check"""
    
    def get(self, request):
        try:
            return JsonResponse({
                'success': True,
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'simulators_available': SIMULATORS_AVAILABLE,
                'apis': {
                    'dashboard_data': '/ai_detection/api/dashboard-data/',
                    'trigger_action': '/ai_detection/api/trigger-action/',
                    'historical_data': '/ai_detection/api/historical-data/',
                    'field_data': '/ai_detection/api/field-data/{field_name}/',
                    'health_check': '/ai_detection/api/health-check/'
                },
                'system_info': {
                    'python_version': '3.x',
                    'django_version': '4.x',
                    'server_time': datetime.now().isoformat()
                }
            })
        except Exception as e:
            return JsonResponse({
                'success': False,
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }, status=500)