from django.db import models
from django.contrib.auth.models import User


class PredictionHistory(models.Model):
    """
    Model to store house price prediction history for users.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='predictions')
    
    # House features
    location = models.CharField(max_length=100)
    square_feet = models.IntegerField()
    lot_size_sqm = models.IntegerField(null=True, blank=True)
    bedrooms = models.IntegerField()
    bathrooms = models.IntegerField()
    parking_spaces = models.IntegerField()
    year_built = models.IntegerField()

    # Additional (optional) features to improve accuracy
    property_type = models.CharField(max_length=50, blank=True, default="")
    condition = models.CharField(max_length=50, blank=True, default="")
    neighborhood_rating = models.IntegerField(null=True, blank=True)
    distance_to_cbd_km = models.FloatField(null=True, blank=True)
    
    # Prediction result
    predicted_price = models.DecimalField(max_digits=12, decimal_places=2)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = 'Prediction Histories'
    
    def __str__(self):
        return f"{self.user.username} - {self.location} - ₱{self.predicted_price}"
