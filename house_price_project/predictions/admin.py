from django.contrib import admin
from .models import PredictionHistory


@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):
    """Admin interface for PredictionHistory model"""
    list_display = ['user', 'location', 'square_feet', 'bedrooms', 'bathrooms', 
                    'parking_spaces', 'year_built', 'property_type', 'condition',
                    'predicted_price', 'created_at']
    list_filter = ['location', 'property_type', 'condition', 'bedrooms', 'bathrooms', 'created_at']
    search_fields = ['user__username', 'location']
    readonly_fields = ['created_at']
    
    fieldsets = (
        ('User Information', {
            'fields': ('user',)
        }),
        ('House Details', {
            'fields': ('location', 'square_feet', 'bedrooms', 'bathrooms', 
                      'parking_spaces', 'year_built', 'lot_size_sqm',
                      'property_type', 'condition', 'neighborhood_rating', 'distance_to_cbd_km')
        }),
        ('Prediction Result', {
            'fields': ('predicted_price',)
        }),
        ('Metadata', {
            'fields': ('created_at',)
        }),
    )
