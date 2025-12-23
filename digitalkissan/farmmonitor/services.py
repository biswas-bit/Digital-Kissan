# ai_detection/services.py
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import random
from django.core.cache import cache

@dataclass
class SimulatedSensor:
    """Simulates a physical sensor with realistic behavior"""
    
    name: str
    min_value: float
    max_value: float
    base_value: float
    daily_pattern: List[float] = field(default_factory=list)
    noise_level: float = 0.1
    last_value: float = None
    trend: float = 0.0
    update_count: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.daily_pattern:
            # Generate realistic 24-hour pattern
            self.daily_pattern = self._generate_daily_pattern()
        
        if self.last_value is None:
            self.last_value = self.base_value
    
    def _generate_daily_pattern(self):
        """Generate realistic daily pattern using sine waves"""
        hours = np.arange(0, 24, 1)
        
        # Main daily cycle (higher during day, lower at night)
        daily_cycle = np.sin((hours - 6) * np.pi / 12)
        
        # Add seasonal effect based on month
        current_month = datetime.now().month
        seasonal_effect = np.sin((current_month - 1) * np.pi / 6) * 0.3
        
        # Combine patterns with different weights
        pattern = daily_cycle * 0.7 + seasonal_effect * 0.3
        
        # Normalize to 0-1 range
        pattern_min = pattern.min()
        pattern_max = pattern.max()
        if pattern_max != pattern_min:
            pattern = (pattern - pattern_min) / (pattern_max - pattern_min)
        else:
            pattern = np.zeros_like(pattern)
        
        # Scale to sensor range
        pattern = self.min_value + pattern * (self.max_value - self.min_value)
        
        return list(pattern)
    
    def get_current_value(self):
        """Get current simulated sensor reading"""
        current_time = datetime.now()
        current_hour = current_time.hour
        current_minute = current_time.minute
        current_second = current_time.second
        
        # Calculate time elapsed since sensor start (for realistic long-term variations)
        time_elapsed = (current_time - self.start_time).total_seconds() / 3600  # in hours
        
        # Get base value from daily pattern
        hour_value = self.daily_pattern[current_hour]
        
        # Interpolate between hours for smooth minute-by-minute changes
        next_hour = (current_hour + 1) % 24
        next_hour_value = self.daily_pattern[next_hour]
        
        # Linear interpolation based on minutes
        minute_fraction = (current_minute * 60 + current_second) / 3600.0
        base_value = hour_value + (next_hour_value - hour_value) * minute_fraction
        
        # Add gradual long-term trend (slow changes over days)
        self.update_count += 1
        if self.update_count % 1440 == 0:  # Change trend every 24 hours (1440 minutes)
            self.trend += random.uniform(-0.02, 0.02)
            # Limit trend to prevent runaway values
            self.trend = max(-0.1, min(0.1, self.trend))
        
        # Apply trend
        base_value += self.trend * base_value
        
        # Add random noise (more noise during certain conditions)
        noise_multiplier = 1.0
        # More noise at midday (increased activity)
        if 11 <= current_hour <= 14:
            noise_multiplier = 1.8
        # More noise during temperature extremes
        elif 'temperature' in self.name.lower() and (current_hour < 6 or current_hour > 20):
            noise_multiplier = 1.3
        
        noise = random.uniform(-self.noise_level, self.noise_level) * noise_multiplier
        
        # Add weather effect
        weather_effect = self._get_weather_effect()
        
        # Calculate final value
        value = base_value + noise + weather_effect
        
        # Ensure within bounds
        value = max(self.min_value, min(self.max_value, value))
        
        # Smooth transition from last value (if available)
        if self.last_value is not None:
            # Use adaptive smoothing based on how much time has passed
            smoothing_factor = 0.85  # Higher = smoother
            value = self.last_value * smoothing_factor + value * (1 - smoothing_factor)
        
        self.last_value = value
        return round(value, 2)
    
    def _get_weather_effect(self):
        """Simulate weather effects on sensor readings"""
        current_hour = datetime.now().hour
        month = datetime.now().month
        
        # Check cache for weather condition to maintain consistency
        cache_key = f"weather_condition_{datetime.now().strftime('%Y%m%d%H')}"
        current_condition = cache.get(cache_key)
        
        # If no cached condition, generate one
        if current_condition is None:
            # Simulate occasional rain events
            if random.random() < 0.15:  # 15% chance of rain effect
                current_condition = 'rain'
            else:
                current_condition = 'normal'
            # Cache for 1 hour
            cache.set(cache_key, current_condition, 3600)
        
        # Rain effect
        if current_condition == 'rain':
            if 'moisture' in self.name.lower():
                return random.uniform(5, 20)  # Increase moisture during rain
            elif 'temperature' in self.name.lower():
                return random.uniform(-4, -1)  # Decrease temperature during rain
            elif 'ph' in self.name.lower():
                return random.uniform(-0.2, 0)  # Slightly acidify soil during rain
        
        # Seasonal effects
        if month in [12, 1, 2]:  # Winter
            if 'temperature' in self.name.lower():
                return -random.uniform(3, 10)
        elif month in [6, 7, 8]:  # Summer
            if 'temperature' in self.name.lower():
                return random.uniform(3, 8)
            elif 'moisture' in self.name.lower():
                # More evaporation in summer
                return -random.uniform(2, 5)
        
        # Diurnal effects (day/night cycle)
        if 'temperature' in self.name.lower():
            if 22 <= current_hour <= 6:  # Night hours
                return -random.uniform(1, 3)
            elif 10 <= current_hour <= 16:  # Peak day hours
                return random.uniform(1, 4)
        
        return 0

class RealisticWeatherSimulator:
    """Simulates realistic weather patterns"""
    
    def __init__(self):
        self.temperature = SimulatedSensor(
            name="temperature",
            min_value=-5,
            max_value=45,
            base_value=25,
            noise_level=0.8
        )
        
        self.humidity = SimulatedSensor(
            name="humidity",
            min_value=20,
            max_value=100,
            base_value=65,
            noise_level=1.5
        )
        
        self.wind_speed = SimulatedSensor(
            name="wind_speed",
            min_value=0,
            max_value=50,
            base_value=12,
            noise_level=0.4
        )
        
        self._weather_conditions = [
            'Clear', 'Partly Cloudy', 'Cloudy', 
            'Light Rain', 'Heavy Rain', 'Thunderstorm',
            'Foggy', 'Windy'
        ]
        self._current_condition = 'Partly Cloudy'
        self._condition_changes = []
        self._last_condition_update = datetime.now()
        
    def get_current_weather(self):
        """Get current simulated weather"""
        current_time = datetime.now()
        
        # Get sensor values
        temp = self.temperature.get_current_value()
        humidity = self.humidity.get_current_value()
        wind = self.wind_speed.get_current_value()
        
        # Update weather condition based on values and time
        self._update_weather_condition(temp, humidity, wind)
        
        # Generate realistic forecast
        forecast = self._generate_forecast()
        
        return {
            'temperature': temp,
            'humidity': humidity,
            'wind_speed': wind,
            'condition': self._current_condition,
            'forecast': forecast,
            'timestamp': current_time
        }
    
    def _update_weather_condition(self, temp, humidity, wind):
        """Update weather condition based on sensor values and time"""
        current_time = datetime.now()
        current_hour = current_time.hour
        
        # Record condition changes
        self._condition_changes.append((current_time, self._current_condition))
        
        # Keep only last 12 hours of changes
        cutoff_time = current_time - timedelta(hours=12)
        self._condition_changes = [c for c in self._condition_changes 
                                  if c[0] > cutoff_time]
        
        # Don't update too frequently (minimum 15 minutes between changes)
        if (current_time - self._last_condition_update).total_seconds() < 900:
            return
        
        # Determine new condition based on values and time
        new_condition = self._current_condition  # Default to current
        
        # High humidity and temperature suggests possible rain
        if humidity > 85 and temp > 28:
            if random.random() < 0.4:  # 40% chance
                new_condition = 'Heavy Rain' if random.random() < 0.3 else 'Light Rain'
        elif humidity > 75:
            if random.random() < 0.3:  # 30% chance
                new_condition = random.choice(['Cloudy', 'Light Rain'])
        elif wind > 20:
            new_condition = 'Windy'
        elif temp > 35:
            new_condition = 'Clear'
        elif current_hour < 6 or current_hour >= 20:  # Night hours
            if random.random() < 0.2:
                new_condition = random.choice(['Clear', 'Partly Cloudy', 'Cloudy'])
        else:
            # Daytime - more likely to be clear/partly cloudy
            if random.random() < 0.25:  # 25% chance to change
                # Weight conditions based on time of day
                if current_hour < 12:
                    # Morning - more likely to be clear
                    new_condition = random.choices(
                        ['Clear', 'Partly Cloudy', 'Cloudy', 'Light Rain'],
                        weights=[0.4, 0.3, 0.2, 0.1]
                    )[0]
                else:
                    # Afternoon - more variation
                    new_condition = random.choice(self._weather_conditions)
        
        # Smooth condition transitions - only change if consistent for last 2 checks
        condition_history = [c[1] for c in self._condition_changes[-2:]]
        if condition_history.count(new_condition) >= 1 or len(condition_history) < 2:
            self._current_condition = new_condition
            self._last_condition_update = current_time
    
    def _generate_forecast(self):
        """Generate realistic 3-day forecast"""
        forecast = []
        current_temp = self.temperature.get_current_value()
        current_condition = self._current_condition
        
        for i in range(3):
            day_time = datetime.now() + timedelta(days=i)
            day_name = day_time.strftime('%a')
            
            # Base temperature with gradual changes
            day_temp = current_temp + random.uniform(-4, 4)
            
            # Add seasonal adjustment
            month = day_time.month
            if month in [12, 1, 2]:  # Winter
                day_temp -= random.uniform(2, 8)
            elif month in [6, 7, 8]:  # Summer
                day_temp += random.uniform(2, 8)
            
            # Today's forecast is based on current conditions
            if i == 0:
                condition = current_condition
                # Adjust temp for time of day
                current_hour = datetime.now().hour
                if current_hour < 12:
                    day_temp += random.uniform(1, 4)  # Warming up
                elif current_hour > 18:
                    day_temp -= random.uniform(1, 3)  # Cooling down
            else:
                # Future days have more variation
                day_temp += random.uniform(-6, 6)
                # Future conditions depend on temperature
                if day_temp > 30:
                    condition = random.choices(
                        ['Clear', 'Partly Cloudy', 'Cloudy'],
                        weights=[0.5, 0.3, 0.2]
                    )[0]
                elif day_temp < 15:
                    condition = random.choices(
                        ['Cloudy', 'Light Rain', 'Clear'],
                        weights=[0.4, 0.4, 0.2]
                    )[0]
                else:
                    condition = random.choice(self._weather_conditions)
            
            forecast.append({
                'day': day_name,
                'temp': round(day_temp),
                'condition': condition,
                'date': day_time.strftime('%d %b')
            })
        
        return forecast

class DynamicSoilSimulator:
    """Simulates dynamic soil conditions"""
    
    def __init__(self):
        # Initialize fields with different base conditions
        self.fields = {
            'North Field': {
                'moisture': SimulatedSensor(
                    name="soil_moisture_north",
                    min_value=25,
                    max_value=90,
                    base_value=70,
                    noise_level=0.7
                ),
                'temperature': SimulatedSensor(
                    name="soil_temp_north",
                    min_value=12,
                    max_value=38,
                    base_value=21,
                    noise_level=0.4
                ),
                'ph': SimulatedSensor(
                    name="soil_ph_north",
                    min_value=5.0,
                    max_value=8.5,
                    base_value=6.7,
                    noise_level=0.03
                ),
                'description': 'North-facing, good drainage'
            },
            'South Field': {
                'moisture': SimulatedSensor(
                    name="soil_moisture_south",
                    min_value=20,
                    max_value=85,
                    base_value=48,
                    noise_level=0.9
                ),
                'temperature': SimulatedSensor(
                    name="soil_temp_south",
                    min_value=14,
                    max_value=40,
                    base_value=25,
                    noise_level=0.3
                ),
                'ph': SimulatedSensor(
                    name="soil_ph_south",
                    min_value=5.5,
                    max_value=8.0,
                    base_value=6.4,
                    noise_level=0.04
                ),
                'description': 'South-facing, sun-exposed'
            },
            'East Field': {
                'moisture': SimulatedSensor(
                    name="soil_moisture_east",
                    min_value=30,
                    max_value=95,
                    base_value=75,
                    noise_level=0.6
                ),
                'temperature': SimulatedSensor(
                    name="soil_temp_east",
                    min_value=10,
                    max_value=36,
                    base_value=20,
                    noise_level=0.35
                ),
                'ph': SimulatedSensor(
                    name="soil_ph_east",
                    min_value=5.8,
                    max_value=8.3,
                    base_value=7.0,
                    noise_level=0.025
                ),
                'description': 'East-facing, morning sun'
            },
            'West Field': {
                'moisture': SimulatedSensor(
                    name="soil_moisture_west",
                    min_value=35,
                    max_value=98,
                    base_value=80,
                    noise_level=0.8
                ),
                'temperature': SimulatedSensor(
                    name="soil_temp_west",
                    min_value=13,
                    max_value=37,
                    base_value=22,
                    noise_level=0.4
                ),
                'ph': SimulatedSensor(
                    name="soil_ph_west",
                    min_value=5.9,
                    max_value=8.1,
                    base_value=6.9,
                    noise_level=0.035
                ),
                'description': 'West-facing, afternoon sun'
            }
        }
        
        # Shared nutrient sensors (affect all fields)
        self.nutrients = {
            'nitrogen': SimulatedSensor(
                name="soil_nitrogen",
                min_value=80,
                max_value=250,
                base_value=155,
                noise_level=2.5
            ),
            'phosphorus': SimulatedSensor(
                name="soil_phosphorus",
                min_value=15,
                max_value=85,
                base_value=48,
                noise_level=1.2
            ),
            'potassium': SimulatedSensor(
                name="soil_potassium",
                min_value=100,
                max_value=280,
                base_value=185,
                noise_level=3.5
            )
        }
        
        self._last_irrigation = {}
        self._last_fertilization = datetime.now() - timedelta(days=30)
        self._crop_absorption = {
            'nitrogen': 0.95,  # 5% absorption per day
            'phosphorus': 0.97,  # 3% absorption per day
            'potassium': 0.96   # 4% absorption per day
        }
        self._weather_influence = 1.0
        
    def get_soil_data(self, field_name: str = None):
        """Get soil data for specific field or all fields"""
        current_time = datetime.now()
        
        # Apply nutrient absorption over time
        self._apply_nutrient_absorption(current_time)
        
        if field_name and field_name in self.fields:
            field_data = self.fields[field_name]
            return {
                'moisture': field_data['moisture'].get_current_value(),
                'temperature': field_data['temperature'].get_current_value(),
                'ph': field_data['ph'].get_current_value(),
                'nitrogen': self.nutrients['nitrogen'].get_current_value(),
                'phosphorus': self.nutrients['phosphorus'].get_current_value(),
                'potassium': self.nutrients['potassium'].get_current_value(),
                'field_description': field_data.get('description', ''),
                'last_irrigation': self._last_irrigation.get(field_name),
                'timestamp': current_time
            }
        else:
            # Get data for all fields
            all_data = {}
            for field_name, sensors in self.fields.items():
                all_data[field_name] = {
                    'moisture': sensors['moisture'].get_current_value(),
                    'temperature': sensors['temperature'].get_current_value(),
                    'ph': sensors['ph'].get_current_value(),
                    'field_description': sensors.get('description', ''),
                    'timestamp': current_time
                }
            
            # Add nutrient data
            nutrient_data = {
                'nitrogen': self.nutrients['nitrogen'].get_current_value(),
                'phosphorus': self.nutrients['phosphorus'].get_current_value(),
                'potassium': self.nutrients['potassium'].get_current_value()
            }
            
            # Calculate overall metrics
            overall_metrics = self._calculate_overall_metrics(all_data, nutrient_data)
            
            return {
                'by_field': all_data,
                'nutrients': nutrient_data,
                'overall': overall_metrics,
                'system_status': {
                    'last_fertilization': self._last_fertilization,
                    'weather_influence': self._weather_influence,
                    'total_fields': len(self.fields)
                }
            }
    
    def _apply_nutrient_absorption(self, current_time):
        """Apply nutrient absorption by crops over time"""
        # Calculate days since last update (simulate daily absorption)
        if not hasattr(self, '_last_nutrient_update'):
            self._last_nutrient_update = current_time
        
        hours_since_update = (current_time - self._last_nutrient_update).total_seconds() / 3600
        
        if hours_since_update >= 1:  # Update at least once per hour
            # Calculate absorption factor based on time
            absorption_factor = hours_since_update / 24  # Daily rate
            
            for nutrient_name, absorption_rate in self._crop_absorption.items():
                if nutrient_name in self.nutrients:
                    nutrient = self.nutrients[nutrient_name]
                    # Apply absorption (nutrients get used up by crops)
                    absorption = nutrient.base_value * (1 - absorption_rate) * absorption_factor
                    nutrient.base_value = max(nutrient.min_value, nutrient.base_value - absorption)
            
            self._last_nutrient_update = current_time
    
    def _calculate_overall_metrics(self, field_data, nutrient_data):
        """Calculate overall soil metrics"""
        if not field_data:
            return {}
        
        moistures = [data['moisture'] for data in field_data.values()]
        temperatures = [data['temperature'] for data in field_data.values()]
        ph_values = [data['ph'] for data in field_data.values()]
        
        avg_moisture = sum(moistures) / len(moistures)
        avg_temp = sum(temperatures) / len(temperatures)
        avg_ph = sum(ph_values) / len(ph_values)
        
        # Calculate health scores
        moisture_score = self._calculate_moisture_score(avg_moisture)
        temp_score = self._calculate_temperature_score(avg_temp)
        ph_score = self._calculate_ph_score(avg_ph)
        
        overall_health = (moisture_score + temp_score + ph_score) / 3
        
        return {
            'moisture': round(avg_moisture, 1),
            'temperature': round(avg_temp, 1),
            'ph_level': round(avg_ph, 1),
            'nitrogen': round(nutrient_data['nitrogen']),
            'phosphorus': round(nutrient_data['phosphorus']),
            'potassium': round(nutrient_data['potassium']),
            'health_score': round(overall_health, 1),
            'moisture_status': self._get_moisture_status(avg_moisture),
            'temperature_status': self._get_temperature_status(avg_temp),
            'ph_status': self._get_ph_status(avg_ph)
        }
    
    def _calculate_moisture_score(self, moisture):
        """Calculate moisture health score (0-100)"""
        if 55 <= moisture <= 75:
            return 100
        elif 45 <= moisture < 55 or 75 < moisture <= 85:
            return 80
        elif 35 <= moisture < 45 or 85 < moisture <= 90:
            return 60
        else:
            return 30
    
    def _calculate_temperature_score(self, temperature):
        """Calculate temperature health score (0-100)"""
        if 18 <= temperature <= 28:
            return 100
        elif 15 <= temperature < 18 or 28 < temperature <= 32:
            return 80
        elif 10 <= temperature < 15 or 32 < temperature <= 35:
            return 60
        else:
            return 30
    
    def _calculate_ph_score(self, ph):
        """Calculate pH health score (0-100)"""
        if 6.0 <= ph <= 7.5:
            return 100
        elif 5.5 <= ph < 6.0 or 7.5 < ph <= 8.0:
            return 80
        elif 5.0 <= ph < 5.5 or 8.0 < ph <= 8.5:
            return 60
        else:
            return 30
    
    def _get_moisture_status(self, moisture):
        """Get moisture status description"""
        if moisture < 40:
            return 'Very Dry'
        elif moisture < 55:
            return 'Dry'
        elif moisture <= 75:
            return 'Optimal'
        elif moisture <= 85:
            return 'Moist'
        else:
            return 'Very Wet'
    
    def _get_temperature_status(self, temperature):
        """Get temperature status description"""
        if temperature < 10:
            return 'Very Cold'
        elif temperature < 18:
            return 'Cool'
        elif temperature <= 28:
            return 'Optimal'
        elif temperature <= 32:
            return 'Warm'
        else:
            return 'Hot'
    
    def _get_ph_status(self, ph):
        """Get pH status description"""
        if ph < 5.5:
            return 'Acidic'
        elif ph < 6.0:
            return 'Slightly Acidic'
        elif ph <= 7.5:
            return 'Neutral'
        elif ph <= 8.0:
            return 'Slightly Alkaline'
        else:
            return 'Alkaline'
    
    def simulate_irrigation(self, field_name: str):
        """Simulate irrigation effect on soil moisture"""
        if field_name in self.fields:
            sensor = self.fields[field_name]['moisture']
            current_value = sensor.get_current_value()
            
            # Calculate irrigation amount based on current moisture
            if current_value < 40:
                irrigation_amount = random.uniform(25, 35)  # Heavy irrigation
            elif current_value < 60:
                irrigation_amount = random.uniform(15, 25)  # Medium irrigation
            else:
                irrigation_amount = random.uniform(5, 15)   # Light irrigation
            
            # Apply irrigation
            new_value = min(sensor.max_value, current_value + irrigation_amount)
            
            # Update sensor base value for gradual effect
            sensor.base_value = new_value
            
            # Record irrigation time
            self._last_irrigation[field_name] = datetime.now()
            
            # Nutrients get diluted by irrigation
            dilution_factor = 0.97  # 3% dilution
            for nutrient in self.nutrients.values():
                nutrient.base_value = max(nutrient.min_value, nutrient.base_value * dilution_factor)
            
            return {
                'status': 'success',
                'field': field_name,
                'irrigation_amount': round(irrigation_amount, 1),
                'new_moisture': round(new_value, 1),
                'timestamp': datetime.now()
            }
        
        return {
            'status': 'error',
            'message': f'Field {field_name} not found'
        }
    
    def simulate_fertilization(self, nutrient_type: str = None, amount: float = 20):
        """Simulate fertilization effect on soil nutrients"""
        current_time = datetime.now()
        
        if nutrient_type and nutrient_type in self.nutrients:
            # Apply specific nutrient
            nutrient = self.nutrients[nutrient_type]
            nutrient.base_value = min(nutrient.max_value, nutrient.base_value + amount)
        else:
            # Apply balanced fertilization
            for nutrient_name, nutrient in self.nutrients.items():
                if nutrient_name == 'nitrogen':
                    nutrient.base_value = min(nutrient.max_value, nutrient.base_value + amount)
                elif nutrient_name == 'phosphorus':
                    nutrient.base_value = min(nutrient.max_value, nutrient.base_value + amount * 0.7)
                elif nutrient_name == 'potassium':
                    nutrient.base_value = min(nutrient.max_value, nutrient.base_value + amount * 0.8)
        
        self._last_fertilization = current_time
        
        return {
            'status': 'success',
            'fertilization_type': nutrient_type or 'balanced',
            'amount': amount,
            'timestamp': current_time
        }

class IntelligentCropSimulator:
    """Simulates intelligent crop growth with realistic patterns"""
    
    def __init__(self):
        # Initialize crops with realistic planting dates
        current_date = datetime.now()
        self.crops = {
            'Tomatoes': {
                'planting_date': current_date - timedelta(days=45),
                'harvest_date': current_date + timedelta(days=45),
                'growth_rate': 0.8,  # Daily growth percentage
                'health_base': 92,
                'pest_susceptibility': 0.25,
                'disease_susceptibility': 0.35,
                'water_requirement': 0.7,
                'nutrient_requirement': 0.8,
                'current_stage': 'Flowering',
                'stage_progress': 0.65,
                'yield_potential': 12.5,  # tons/ha
                'color': '#ef4444'  # Red for tomatoes
            },
            'Corn': {
                'planting_date': current_date - timedelta(days=60),
                'harvest_date': current_date + timedelta(days=30),
                'growth_rate': 0.65,
                'health_base': 88,
                'pest_susceptibility': 0.15,
                'disease_susceptibility': 0.25,
                'water_requirement': 0.8,
                'nutrient_requirement': 0.9,
                'current_stage': 'Tasseling',
                'stage_progress': 0.78,
                'yield_potential': 9.5,  # tons/ha
                'color': '#fbbf24'  # Yellow for corn
            },
            'Wheat': {
                'planting_date': current_date - timedelta(days=75),
                'harvest_date': current_date + timedelta(days=15),
                'growth_rate': 0.55,
                'health_base': 85,
                'pest_susceptibility': 0.35,
                'disease_susceptibility': 0.45,
                'water_requirement': 0.6,
                'nutrient_requirement': 0.7,
                'current_stage': 'Heading',
                'stage_progress': 0.82,
                'yield_potential': 6.8,  # tons/ha
                'color': '#d97706'  # Brown for wheat
            },
            'Potatoes': {
                'planting_date': current_date - timedelta(days=50),
                'harvest_date': current_date + timedelta(days=40),
                'growth_rate': 0.75,
                'health_base': 90,
                'pest_susceptibility': 0.4,
                'disease_susceptibility': 0.5,
                'water_requirement': 0.75,
                'nutrient_requirement': 0.85,
                'current_stage': 'Tuber Bulking',
                'stage_progress': 0.62,
                'yield_potential': 25.0,  # tons/ha
                'color': '#7c3aed'  # Purple for potatoes
            }
        }
        
        self._weather_effects = {}
        self._soil_effects = {}
        self._historical_health = {}
        self._growth_history = {}
        
        # Initialize history for all crops
        for crop_name in self.crops.keys():
            self._historical_health[crop_name] = []
            self._growth_history[crop_name] = []
    
    def get_crop_health(self, crop_name: str, soil_data: Dict = None, weather_data: Dict = None):
        """Calculate dynamic crop health with realistic variations"""
        if crop_name not in self.crops:
            return self._get_default_crop_data(crop_name)
        
        crop = self.crops[crop_name]
        current_time = datetime.now()
        days_since_planting = (current_time - crop['planting_date']).days
        
        # Base health from crop characteristics
        base_health = crop['health_base']
        
        # Growth stage effect (health improves as plant matures)
        stage_effect = self._calculate_stage_effect(crop['current_stage'], crop['stage_progress'])
        
        # Weather effects
        weather_effect = self._calculate_weather_effect(weather_data) if weather_data else 0
        
        # Soil effects
        soil_effect = self._calculate_soil_effect(soil_data, crop) if soil_data else 0
        
        # Pest and disease simulation
        pest_effect = self._simulate_pests(crop_name, days_since_planting, weather_data)
        disease_effect = self._simulate_diseases(crop_name, weather_data, soil_data)
        
        # Calculate total health
        total_health = (
            base_health +
            stage_effect +
            weather_effect +
            soil_effect -
            pest_effect -
            disease_effect
        )
        
        # Add realistic daily variations
        hour = current_time.hour
        # Photosynthesis peak hours (more growth, better health)
        if 9 <= hour <= 15:  
            total_health += random.uniform(0, 2)
        # Night time (less growth, stable health)
        elif hour < 6 or hour > 20:  
            total_health += random.uniform(-1, 0)
        
        # Add seasonal effect
        month = current_time.month
        if month in [3, 4, 5]:  # Spring - optimal growth
            total_health += random.uniform(1, 3)
        elif month in [12, 1, 2]:  # Winter - reduced growth
            total_health += random.uniform(-2, 0)
        
        # Add slow trend based on cumulative effects
        trend = self._calculate_health_trend(crop_name, total_health)
        total_health += trend
        
        # Ensure health stays within bounds
        total_health = max(0, min(100, total_health))
        
        # Calculate yield prediction based on health
        yield_prediction = self._calculate_yield_prediction(crop_name, total_health, days_since_planting)
        
        # Store historical data
        health_record = {
            'timestamp': current_time,
            'health': total_health,
            'weather_effect': weather_effect,
            'soil_effect': soil_effect,
            'pest_effect': pest_effect,
            'disease_effect': disease_effect,
            'yield_prediction': yield_prediction
        }
        
        self._historical_health[crop_name].append(health_record)
        
        # Keep only last 30 days of history
        cutoff_time = current_time - timedelta(days=30)
        self._historical_health[crop_name] = [
            h for h in self._historical_health[crop_name]
            if h['timestamp'] > cutoff_time
        ]
        
        # Update stage progress based on time
        self._update_crop_stage(crop_name, days_since_planting)
        
        return {
            'health_score': round(total_health, 1),
            'growth_stage': crop['current_stage'],
            'stage_progress': crop['stage_progress'],
            'days_since_planting': days_since_planting,
            'days_to_harvest': max(0, (crop['harvest_date'] - current_time).days),
            'pest_pressure': round(pest_effect, 1),
            'disease_risk': round(disease_effect, 1),
            'yield_prediction': round(yield_prediction, 1),
            'color': crop.get('color', '#22c55e'),
            'last_updated': current_time
        }
    
    def _calculate_stage_effect(self, stage: str, progress: float):
        """Calculate health effect based on growth stage"""
        stage_effects = {
            'Germination': 0,
            'Seedling': 2,
            'Vegetative': 5,
            'Flowering': 8,
            'Fruiting': 10,
            'Maturation': 12,
            'Tasseling': 7,
            'Silking': 9,
            'Heading': 6,
            'Ripening': 11,
            'Tuber Initiation': 4,
            'Tuber Bulking': 9
        }
        
        return stage_effects.get(stage, 0) * progress
    
    def _calculate_weather_effect(self, weather_data: Dict):
        """Calculate weather effect on crop health"""
        if not weather_data:
            return 0
        
        temp = weather_data.get('temperature', 25)
        humidity = weather_data.get('humidity', 60)
        condition = weather_data.get('condition', 'Clear')
        wind_speed = weather_data.get('wind_speed', 10)
        
        effect = 0
        
        # Temperature effect (optimal 20-28°C for most crops)
        if 22 <= temp <= 26:
            effect += 4  # Optimal range
        elif 18 <= temp <= 30:
            effect += 2  # Good range
        elif temp < 10 or temp > 35:
            effect -= 8  # Stressful conditions
        elif temp < 15 or temp > 32:
            effect -= 4  # Suboptimal conditions
        
        # Humidity effect (optimal 60-70%)
        if 60 <= humidity <= 70:
            effect += 3
        elif humidity > 85:
            effect -= 5  # Too humid - fungal issues
        elif humidity < 40:
            effect -= 4  # Too dry - water stress
        
        # Weather condition effect
        condition_effects = {
            'Clear': 2,
            'Partly Cloudy': 1,
            'Cloudy': 0,
            'Light Rain': 3,  # Good for irrigation
            'Heavy Rain': -4,  # Can damage crops
            'Thunderstorm': -8,
            'Windy': -2,
            'Foggy': -1
        }
        
        effect += condition_effects.get(condition, 0)
        
        # Wind effect (moderate wind is good, strong wind is bad)
        if wind_speed > 25:
            effect -= 3  # Strong wind damage
        elif 5 <= wind_speed <= 15:
            effect += 1  # Good for pollination
        
        return effect
    
    def _calculate_soil_effect(self, soil_data: Dict, crop: Dict):
        """Calculate soil effect on crop health"""
        if not soil_data:
            return 0
        
        effect = 0
        
        # Get soil overall metrics
        soil_overall = soil_data.get('overall', {})
        
        # Moisture effect
        moisture = soil_overall.get('moisture', 60)
        water_req = crop.get('water_requirement', 0.7)
        optimal_moisture = 55 + water_req * 20  # Scale based on crop water requirement
        
        if abs(moisture - optimal_moisture) <= 10:
            effect += 4  # Good moisture
        elif abs(moisture - optimal_moisture) <= 20:
            effect += 2  # Acceptable moisture
        else:
            effect -= 4  # Poor moisture
        
        # pH effect (optimal 6.0-7.0)
        ph = soil_overall.get('ph_level', 6.5)
        if 6.0 <= ph <= 7.0:
            effect += 3
        elif 5.5 <= ph < 6.0 or 7.0 < ph <= 7.5:
            effect += 1
        elif ph < 5.0 or ph > 8.0:
            effect -= 4
        
        # Nutrient effects
        nitrogen = soil_overall.get('nitrogen', 150)
        phosphorus = soil_overall.get('phosphorus', 45)
        potassium = soil_overall.get('potassium', 180)
        
        nutrient_req = crop.get('nutrient_requirement', 0.8)
        
        # Check if nutrients are in optimal ranges (scaled by crop requirement)
        optimal_nitrogen = 140 + nutrient_req * 40
        if abs(nitrogen - optimal_nitrogen) <= 30:
            effect += 2
        elif nitrogen < optimal_nitrogen - 50:
            effect -= 2
        
        optimal_phosphorus = 40 + nutrient_req * 15
        if abs(phosphorus - optimal_phosphorus) <= 10:
            effect += 1
        elif phosphorus < optimal_phosphorus - 20:
            effect -= 1
        
        optimal_potassium = 160 + nutrient_req * 40
        if abs(potassium - optimal_potassium) <= 40:
            effect += 1
        elif potassium < optimal_potassium - 60:
            effect -= 1
        
        return effect
    
    def _simulate_pests(self, crop_name: str, days_since_planting: int, weather_data: Dict = None):
        """Simulate pest pressure with realistic patterns"""
        crop = self.crops[crop_name]
        base_pressure = crop['pest_susceptibility'] * 25
        
        # Pest pressure increases with crop age
        age_factor = min(1.5, days_since_planting / 60)
        
        # Seasonal effect (more pests in warm months)
        month = datetime.now().month
        if 4 <= month <= 9:  # Spring to Fall
            seasonal_factor = 1.2 + (month - 4) * 0.1
        else:
            seasonal_factor = 0.8
        
        # Weather effect
        weather_factor = 1.0
        if weather_data:
            temp = weather_data.get('temperature', 25)
            humidity = weather_data.get('humidity', 60)
            
            # Pests thrive in warm, humid conditions
            if temp > 25 and humidity > 70:
                weather_factor = 1.5
            elif temp > 20 and humidity > 60:
                weather_factor = 1.2
            elif temp < 15:
                weather_factor = 0.7
        
        # Time of day effect (some pests more active at certain times)
        hour = datetime.now().hour
        if 18 <= hour <= 23 or 0 <= hour <= 5:  # Evening/Night
            time_factor = 1.3  # Some pests are nocturnal
        else:
            time_factor = 1.0
        
        # Random outbreaks
        outbreak_chance = random.random()
        if outbreak_chance < 0.03:  # 3% chance of major pest outbreak
            outbreak_factor = 3.0
        elif outbreak_chance < 0.1:  # 10% chance of minor outbreak
            outbreak_factor = 1.5
        else:
            outbreak_factor = 1.0
        
        pest_pressure = base_pressure * age_factor * seasonal_factor * weather_factor * time_factor * outbreak_factor
        
        # Add some random noise
        pest_pressure += random.uniform(-2, 2)
        
        return max(0, min(30, pest_pressure))  # Cap at 30
    
    def _simulate_diseases(self, crop_name: str, weather_data: Dict = None, soil_data: Dict = None):
        """Simulate disease risk with realistic patterns"""
        crop = self.crops[crop_name]
        base_risk = crop['disease_susceptibility'] * 20
        
        # Disease risk based on conditions
        risk_factors = []
        
        # Weather-based disease risk
        if weather_data:
            humidity = weather_data.get('humidity', 60)
            temp = weather_data.get('temperature', 25)
            condition = weather_data.get('condition', 'Clear')
            
            # High humidity increases fungal disease risk
            if humidity > 80:
                risk_factors.append(2.0)
            elif humidity > 70:
                risk_factors.append(1.5)
            elif humidity > 60:
                risk_factors.append(1.2)
            
            # Moderate temperatures favor many diseases
            if 20 <= temp <= 28:
                risk_factors.append(1.3)
            
            # Rain increases disease spread
            if 'Rain' in condition:
                risk_factors.append(1.7)
            elif condition in ['Cloudy', 'Foggy']:
                risk_factors.append(1.2)
        
        # Soil-based disease risk
        if soil_data:
            soil_overall = soil_data.get('overall', {})
            moisture = soil_overall.get('moisture', 60)
            
            # Waterlogged soil increases disease risk
            if moisture > 80:
                risk_factors.append(1.4)
            elif moisture > 70:
                risk_factors.append(1.2)
        
        # Crop stage effect (some stages more susceptible)
        stage = crop['current_stage']
        if stage in ['Flowering', 'Fruiting']:
            risk_factors.append(1.4)
        elif stage in ['Tuber Bulking', 'Ripening']:
            risk_factors.append(1.3)
        
        # Calculate combined risk
        if risk_factors:
            combined_factor = np.prod(risk_factors)
        else:
            combined_factor = 1.0
        
        disease_risk = base_risk * combined_factor
        
        # Add some noise
        disease_risk += random.uniform(-1, 1)
        
        return max(0, min(25, disease_risk))  # Cap at 25
    
    def _calculate_health_trend(self, crop_name: str, current_health: float):
        """Calculate health trend based on historical data"""
        if crop_name not in self._historical_health or len(self._historical_health[crop_name]) < 5:
            return 0
        
        history = self._historical_health[crop_name]
        
        # Get last 24 hours of data
        recent_history = [
            h for h in history[-24:] 
            if (datetime.now() - h['timestamp']).total_seconds() <= 86400
        ]
        
        if len(recent_history) < 3:
            return 0
        
        # Calculate trend using linear regression
        times = [(h['timestamp'] - recent_history[0]['timestamp']).total_seconds() / 3600 
                for h in recent_history]
        healths = [h['health'] for h in recent_history]
        
        # Simple trend calculation
        if len(times) > 1 and times[-1] != times[0]:
            trend = (healths[-1] - healths[0]) / (times[-1] - times[0])
            # Convert hourly trend to daily trend
            daily_trend = trend * 24
            # Normalize and limit
            return max(-2, min(2, daily_trend * 0.1))
        
        return 0
    
    def _calculate_yield_prediction(self, crop_name: str, health: float, days_since_planting: int):
        """Calculate yield prediction based on crop health and growth"""
        crop = self.crops[crop_name]
        base_yield = crop['yield_potential']
        
        # Health factor (0.5 to 1.5)
        health_factor = 0.5 + (health / 100)
        
        # Growth progress factor
        total_days = (crop['harvest_date'] - crop['planting_date']).days
        growth_progress = min(1.0, days_since_planting / total_days)
        progress_factor = 0.7 + (growth_progress * 0.6)  # 0.7 to 1.3
        
        # Calculate predicted yield
        predicted_yield = base_yield * health_factor * progress_factor
        
        # Add some random variation
        predicted_yield += random.uniform(-0.5, 0.5)
        
        return max(0, predicted_yield)
    
    def _update_crop_stage(self, crop_name: str, days_since_planting: int):
        """Update crop growth stage based on time"""
        if crop_name not in self.crops:
            return
        
        crop = self.crops[crop_name]
        total_days = (crop['harvest_date'] - crop['planting_date']).days
        
        if total_days <= 0:
            return
        
        # Calculate progress
        progress = min(1.0, days_since_planting / total_days)
        crop['stage_progress'] = progress
        
        # Update stage based on progress
        if progress < 0.2:
            crop['current_stage'] = 'Germination' if progress < 0.1 else 'Seedling'
        elif progress < 0.4:
            if crop_name == 'Corn':
                crop['current_stage'] = 'Vegetative'
            elif crop_name == 'Wheat':
                crop['current_stage'] = 'Vegetative'
            elif crop_name == 'Potatoes':
                crop['current_stage'] = 'Tuber Initiation'
            else:
                crop['current_stage'] = 'Vegetative'
        elif progress < 0.7:
            if crop_name == 'Corn':
                crop['current_stage'] = 'Tasseling'
            elif crop_name == 'Wheat':
                crop['current_stage'] = 'Heading'
            elif crop_name == 'Potatoes':
                crop['current_stage'] = 'Tuber Bulking'
            else:
                crop['current_stage'] = 'Flowering'
        elif progress < 0.9:
            crop['current_stage'] = 'Fruiting' if crop_name != 'Wheat' else 'Ripening'
        else:
            crop['current_stage'] = 'Maturation'
    
    def get_growth_progress(self, crop_name: str):
        """Generate realistic growth progress data"""
        if crop_name not in self.crops:
            return []
        
        crop = self.crops[crop_name]
        current_time = datetime.now()
        days_since_planting = (current_time - crop['planting_date']).days
        total_days = (crop['harvest_date'] - crop['planting_date']).days
        
        # Generate weekly progress with realistic patterns
        progress_data = []
        total_weeks = min(16, total_days // 7 + 1)
        
        for week in range(1, total_weeks + 1):
            week_days = week * 7
            
            if week_days <= days_since_planting:
                # Historical actual data
                base_progress = min(100, (week_days / total_days) * 100)
                
                # Add realistic variations based on growth stage
                if week <= total_weeks // 4:  # Early growth
                    variation = random.uniform(-2, 4)
                elif week <= total_weeks // 2:  # Mid growth
                    variation = random.uniform(-1, 3)
                else:  # Late growth
                    variation = random.uniform(-0.5, 2)
                
                actual_progress = base_progress + variation
            else:
                # Future expected data (smooth projection)
                actual_progress = None
            
            expected_progress = min(100, (week_days / total_days) * 100)
            
            progress_data.append({
                'week': f"Week {week}",
                'actual': round(actual_progress, 1) if actual_progress is not None else None,
                'expected': round(expected_progress, 1),
                'week_number': week
            })
        
        return progress_data
    
    def get_all_crops_health(self, soil_data: Dict = None, weather_data: Dict = None):
        """Get health data for all crops"""
        results = {}
        for crop_name in self.crops.keys():
            results[crop_name] = self.get_crop_health(crop_name, soil_data, weather_data)
        return results
    
    def _get_default_crop_data(self, crop_name: str):
        """Get default data for unknown crops"""
        return {
            'health_score': 85,
            'growth_stage': 'Vegetative',
            'stage_progress': 0.5,
            'days_since_planting': 45,
            'days_to_harvest': 45,
            'pest_pressure': 5.0,
            'disease_risk': 3.0,
            'yield_prediction': 8.0,
            'color': '#22c55e',
            'last_updated': datetime.now()
        }

class DynamicRainfallSimulator:
    """Simulates dynamic rainfall patterns"""
    
    def __init__(self):
        self._rain_patterns = self._generate_seasonal_patterns()
        self._current_pattern = None
        self._last_update = datetime.now()
        self._regional_variation = random.uniform(0.8, 1.2)
        self._current_season = self._get_current_season()
        self._storm_active = False
        self._storm_end_time = None
        
    def _generate_seasonal_patterns(self):
        """Generate seasonal rainfall patterns"""
        return {
            'winter': {
                'base': 1.5,
                'variation': 3,
                'storm_chance': 0.05,
                'storm_intensity': 1.2,
                'dry_spell_chance': 0.3
            },
            'spring': {
                'base': 4.5,
                'variation': 8,
                'storm_chance': 0.15,
                'storm_intensity': 1.5,
                'dry_spell_chance': 0.2
            },
            'summer': {
                'base': 8.0,
                'variation': 15,
                'storm_chance': 0.25,
                'storm_intensity': 2.0,
                'dry_spell_chance': 0.4
            },
            'monsoon': {
                'base': 18.0,
                'variation': 25,
                'storm_chance': 0.5,
                'storm_intensity': 3.0,
                'dry_spell_chance': 0.1
            },
            'autumn': {
                'base': 3.0,
                'variation': 6,
                'storm_chance': 0.1,
                'storm_intensity': 1.3,
                'dry_spell_chance': 0.25
            }
        }
    
    def _get_current_season(self):
        """Determine current season"""
        month = datetime.now().month
        
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7]:
            return 'summer'
        elif month in [8, 9]:
            return 'monsoon'
        else:
            return 'autumn'
    
    def get_rainfall_predictions(self, days: int = 7):
        """Generate dynamic rainfall predictions"""
        current_time = datetime.now()
        
        # Check if season changed
        current_season = self._get_current_season()
        if current_season != self._current_season:
            self._current_season = current_season
            self._current_pattern = None
            print(f"Season changed to {current_season}")
        
        # Check if we need to update the pattern
        if (current_time - self._last_update).total_seconds() > 21600:  # 6 hours
            self._current_pattern = None
        
        if self._current_pattern is None:
            # Generate new pattern
            pattern = self._rain_patterns[self._current_season]
            self._current_pattern = self._generate_rain_pattern(pattern, days)
            self._last_update = current_time
        
        predictions = []
        
        # Check for active storm
        if self._storm_active and self._storm_end_time:
            if current_time > self._storm_end_time:
                self._storm_active = False
                self._storm_end_time = None
        
        for i in range(days):
            prediction_date = current_time + timedelta(days=i)
            day_name = prediction_date.strftime('%a')
            
            # Get base rainfall for this day
            base_rain = self._current_pattern[i] * self._regional_variation
            
            # Adjust based on day of week (weekends might have different patterns)
            if day_name in ['Sat', 'Sun']:
                base_rain *= random.uniform(0.9, 1.1)
            
            # Check for storm
            if not self._storm_active and random.random() < self._rain_patterns[self._current_season]['storm_chance']:
                self._storm_active = True
                storm_duration = random.randint(1, 3)
                self._storm_end_time = prediction_date + timedelta(days=storm_duration)
                storm_intensity = self._rain_patterns[self._current_season]['storm_intensity']
                base_rain *= storm_intensity
            
            # If storm is active, increase rainfall
            if self._storm_active and prediction_date <= self._storm_end_time:
                base_rain *= 1.5
            
            # Check for dry spell
            if random.random() < self._rain_patterns[self._current_season]['dry_spell_chance']:
                base_rain *= 0.3
            
            # Calculate probability and status
            probability = self._calculate_probability(base_rain)
            status, status_class = self._get_rainfall_status(base_rain)
            
            # Add some random noise
            final_rain = max(0, base_rain + random.uniform(-0.5, 0.5))
            
            predictions.append({
                'day': day_name,
                'date': prediction_date.strftime('%d %b'),
                'rainfall': round(final_rain, 1),
                'probability': probability,
                'status': status,
                'status_class': status_class,
                'storm_active': self._storm_active and prediction_date <= self._storm_end_time,
                'season': self._current_season
            })
        
        return predictions
    
    def _generate_rain_pattern(self, pattern, days):
        """Generate a realistic rainfall pattern"""
        pattern_length = days
        
        # Create base pattern with multiple sine waves for natural variation
        x = np.linspace(0, 3 * np.pi, pattern_length)
        
        # Primary wave (major weather systems)
        primary_wave = np.sin(x) * pattern['variation']
        
        # Secondary wave (minor variations)
        secondary_wave = np.sin(2 * x + 1.5) * (pattern['variation'] * 0.3)
        
        # Tertiary wave (random disturbances)
        tertiary_wave = np.sin(3 * x + 2.7) * (pattern['variation'] * 0.15)
        
        # Combine waves
        base_pattern = primary_wave + secondary_wave + tertiary_wave + pattern['base']
        
        # Add random peaks for storms
        for i in range(pattern_length):
            if random.random() < pattern['storm_chance']:
                storm_intensity = random.uniform(1.3, pattern['storm_intensity'])
                # Make storms affect multiple days
                storm_duration = random.randint(1, 2)
                for j in range(storm_duration):
                    if i + j < pattern_length:
                        base_pattern[i + j] *= storm_intensity
        
        # Ensure no negative values
        base_pattern = np.maximum(base_pattern, 0)
        
        # Smooth the pattern for more realistic distribution
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(base_pattern, sigma=0.8)
        
        # Add occasional dry days
        for i in range(pattern_length):
            if random.random() < pattern['dry_spell_chance']:
                smoothed[i] *= 0.2
        
        return smoothed.tolist()
    
    def _calculate_probability(self, rainfall):
        """Calculate probability of rain based on predicted amount"""
        if rainfall < 0.5:
            return random.randint(10, 30)
        elif rainfall < 2:
            return random.randint(30, 50)
        elif rainfall < 5:
            return random.randint(50, 70)
        elif rainfall < 10:
            return random.randint(70, 85)
        else:
            return random.randint(85, 98)
    
    def _get_rainfall_status(self, rainfall):
        """Get rainfall status and CSS class"""
        if rainfall < 2:
            return 'Light', 'good'
        elif rainfall < 10:
            return 'Moderate', 'warning'
        else:
            return 'Heavy', 'critical'

# =============================================================================
# GLOBAL SIMULATOR INSTANCES
# =============================================================================

# Create singleton instances
weather_simulator = RealisticWeatherSimulator()
soil_simulator = DynamicSoilSimulator()
crop_simulator = IntelligentCropSimulator()
rainfall_simulator = DynamicRainfallSimulator()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_system_status():
    """Get overall system status"""
    return {
        'status': 'operational',
        'simulators': {
            'weather': 'active',
            'soil': 'active',
            'crops': 'active',
            'rainfall': 'active'
        },
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }

def simulate_random_event():
    """Simulate a random farming event"""
    events = [
        {
            'type': 'irrigation',
            'field': random.choice(list(soil_simulator.fields.keys())),
            'amount': random.uniform(10, 30)
        },
        {
            'type': 'fertilization',
            'nutrient': random.choice(['nitrogen', 'phosphorus', 'potassium']),
            'amount': random.uniform(15, 25)
        },
        {
            'type': 'weather_change',
            'change': random.choice(['temperature_increase', 'temperature_decrease', 
                                   'rain_start', 'rain_stop'])
        },
        {
            'type': 'pest_outbreak',
            'crop': random.choice(list(crop_simulator.crops.keys())),
            'severity': random.choice(['low', 'medium', 'high'])
        }
    ]
    
    return random.choice(events)

# =============================================================================
# DATA EXPORT FUNCTIONS
# =============================================================================

def export_sensor_data(hours: int = 24):
    """Export historical sensor data for analysis"""
    current_time = datetime.now()
    start_time = current_time - timedelta(hours=hours)
    
    # This would normally fetch from a database
    # For now, generate simulated historical data
    
    data = {
        'weather': [],
        'soil': [],
        'crops': [],
        'metadata': {
            'start_time': start_time.isoformat(),
            'end_time': current_time.isoformat(),
            'data_points': 24,
            'exported_at': current_time.isoformat()
        }
    }
    
    # Generate hourly data points
    for i in range(hours):
        timestamp = start_time + timedelta(hours=i)
        
        # Weather data
        temp = 24 + np.sin(i * np.pi / 12) * 8 + random.uniform(-2, 2)
        humidity = 65 + np.sin(i * np.pi / 12) * 15 + random.uniform(-5, 5)
        
        data['weather'].append({
            'timestamp': timestamp.isoformat(),
            'temperature': round(temp, 1),
            'humidity': round(humidity, 1),
            'hour': timestamp.hour
        })
        
        # Soil data (average across fields)
        avg_moisture = 65 + np.sin(i * np.pi / 12) * 10 + random.uniform(-3, 3)
        avg_temp = 22 + np.sin(i * np.pi / 12) * 6 + random.uniform(-1, 1)
        
        data['soil'].append({
            'timestamp': timestamp.isoformat(),
            'moisture': round(avg_moisture, 1),
            'temperature': round(avg_temp, 1),
            'ph': 6.8 + random.uniform(-0.1, 0.1)
        })
        
        # Crop data (average health)
        avg_health = 85 + np.sin(i * np.pi / 24) * 5 + random.uniform(-2, 2)
        
        data['crops'].append({
            'timestamp': timestamp.isoformat(),
            'avg_health': round(avg_health, 1),
            'hour': timestamp.hour
        })
    
    return data

# =============================================================================
# INITIALIZATION
# =============================================================================

if __name__ == "__main__":
    print("🌱 Farm Monitoring Simulators Initialized")
    print(f"Weather Simulator: {weather_simulator.__class__.__name__}")
    print(f"Soil Simulator: {soil_simulator.__class__.__name__}")
    print(f"Crop Simulator: {crop_simulator.__class__.__name__}")
    print(f"Rainfall Simulator: {rainfall_simulator.__class__.__name__}")
    
    # Test each simulator
    print("\n🧪 Testing simulators...")
    
    # Test weather simulator
    weather = weather_simulator.get_current_weather()
    print(f"🌤️  Current Weather: {weather['temperature']}°C, {weather['condition']}")
    
    # Test soil simulator
    soil = soil_simulator.get_soil_data()
    print(f"🌱 Overall Soil Moisture: {soil['overall']['moisture']}%")
    
    # Test crop simulator
    crop_health = crop_simulator.get_crop_health('Tomatoes', soil['overall'], weather)
    print(f"🍅 Tomato Health: {crop_health['health_score']}%")
    
    # Test rainfall simulator
    rainfall = rainfall_simulator.get_rainfall_predictions(3)
    print(f"🌧️  Rainfall Prediction (Today): {rainfall[0]['rainfall']}mm")
    
    print("\n✅ All simulators are working correctly!")