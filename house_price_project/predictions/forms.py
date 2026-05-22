from django import forms
from .models import PredictionHistory
from datetime import datetime


class PredictionForm(forms.Form):
    """
    Form for collecting house features for price prediction.
    """
    # Philippines-focused locations (must match the values used during training)
    LOCATION_CHOICES = [
        ("Metro Manila", "Metro Manila"),
        ("Cebu", "Cebu"),
        ("Davao", "Davao"),
        ("Baguio", "Baguio"),
        ("Iloilo", "Iloilo"),
        ("Bacolod", "Bacolod"),
        ("Cagayan de Oro", "Cagayan de Oro"),
        ("General Santos", "General Santos"),
    ]
    
    location = forms.ChoiceField(
        choices=LOCATION_CHOICES,
        widget=forms.Select(attrs={
            'class': 'form-control',
            'placeholder': 'Select location'
        })
    )
    
    square_feet = forms.IntegerField(
        min_value=100,
        max_value=10000,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter floor area (sqm)'
        })
    )

    lot_size_sqm = forms.IntegerField(
        required=False,
        min_value=0,
        max_value=10000,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Lot size (sqm) (optional)'
        })
    )

    PROPERTY_TYPE_CHOICES = [
        ("Condo", "Condo"),
        ("House & Lot", "House & Lot"),
        ("Townhouse", "Townhouse"),
    ]

    property_type = forms.ChoiceField(
        required=False,
        choices=PROPERTY_TYPE_CHOICES,
        widget=forms.Select(attrs={
            'class': 'form-control',
        })
    )

    CONDITION_CHOICES = [
        ("New", "New"),
        ("Good", "Good"),
        ("Needs Repair", "Needs Repair"),
    ]

    condition = forms.ChoiceField(
        required=False,
        choices=CONDITION_CHOICES,
        widget=forms.Select(attrs={
            'class': 'form-control',
        })
    )
    
    bedrooms = forms.IntegerField(
        min_value=0,
        max_value=10,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Number of bedrooms'
        })
    )
    
    bathrooms = forms.IntegerField(
        min_value=1,
        max_value=10,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Number of bathrooms'
        })
    )
    
    parking_spaces = forms.IntegerField(
        min_value=0,
        max_value=10,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Parking spaces'
        })
    )
    
    year_built = forms.IntegerField(
        min_value=1900,
        max_value=datetime.now().year,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Year built'
        })
    )

    neighborhood_rating = forms.IntegerField(
        required=False,
        min_value=1,
        max_value=10,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Neighborhood rating (1-10) (optional)'
        })
    )

    distance_to_cbd_km = forms.FloatField(
        required=False,
        min_value=0,
        max_value=200,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.1',
            'placeholder': 'Distance to CBD (km) (optional)'
        })
    )
    
    def clean_square_feet(self):
        square_feet = self.cleaned_data.get('square_feet')
        if square_feet < 100:
            raise forms.ValidationError('Floor area must be at least 100 sqm.')
        return square_feet

    def clean_lot_size_sqm(self):
        lot_size = self.cleaned_data.get('lot_size_sqm')
        if lot_size is None:
            return lot_size
        if lot_size < 0:
            raise forms.ValidationError('Lot size cannot be negative.')
        return lot_size

    def clean_neighborhood_rating(self):
        rating = self.cleaned_data.get('neighborhood_rating')
        if rating is None:
            return rating
        if rating < 1 or rating > 10:
            raise forms.ValidationError('Neighborhood rating must be between 1 and 10.')
        return rating

    def clean_distance_to_cbd_km(self):
        d = self.cleaned_data.get('distance_to_cbd_km')
        if d is None:
            return d
        if d < 0:
            raise forms.ValidationError('Distance to CBD cannot be negative.')
        return d
    
    def clean_year_built(self):
        year_built = self.cleaned_data.get('year_built')
        if year_built > datetime.now().year:
            raise forms.ValidationError('Year built cannot be in the future.')
        return year_built
