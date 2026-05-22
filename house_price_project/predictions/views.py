from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.db.models import Avg, Max, Min
from django.http import HttpResponse
from datetime import datetime
import csv
import json
import joblib
import os
import pandas as pd
from .models import PredictionHistory
from .forms import PredictionForm


# Load the trained model pipeline (joblib)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml_models", "model.pkl")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "ml_models", "model_metrics.json")

_cached_model = None
_cached_model_mtime = None
_cached_metrics = None
_cached_metrics_mtime = None


def _load_model():
    """
    Lazily load model.pkl and auto-reload if the file changes (helps during development/retraining).
    """
    global _cached_model, _cached_model_mtime
    try:
        mtime = os.path.getmtime(MODEL_PATH)
        if _cached_model is None or _cached_model_mtime != mtime:
            _cached_model = joblib.load(MODEL_PATH)
            _cached_model_mtime = mtime
    except Exception:
        _cached_model = None
        _cached_model_mtime = None
    return _cached_model


def _load_metrics():
    """
    Load model_metrics.json (contains CV MAE which we use as a simple error band).
    """
    global _cached_metrics, _cached_metrics_mtime
    try:
        mtime = os.path.getmtime(METRICS_PATH)
        if _cached_metrics is None or _cached_metrics_mtime != mtime:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                _cached_metrics = json.load(f)
            _cached_metrics_mtime = mtime
    except Exception:
        _cached_metrics = None
        _cached_metrics_mtime = None
    return _cached_metrics


def home(request):
    """Home page view"""
    return render(request, 'predictions/home.html')


def register(request):
    """User registration view"""
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Account created for {username}!')
            login(request, user)
            return redirect('home')
    else:
        form = UserCreationForm()
    return render(request, 'predictions/register.html', {'form': form})


@login_required
def predict_price(request):
    """
    House price prediction view.
    Accepts house features via form and returns predicted price.
    """
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            # Extract form data
            location = form.cleaned_data['location']
            square_feet = form.cleaned_data['square_feet']
            bedrooms = form.cleaned_data['bedrooms']
            bathrooms = form.cleaned_data['bathrooms']
            parking_spaces = form.cleaned_data['parking_spaces']
            year_built = form.cleaned_data['year_built']
            lot_size_sqm = form.cleaned_data.get('lot_size_sqm')
            property_type = form.cleaned_data.get('property_type') or ""
            condition = form.cleaned_data.get('condition') or ""
            neighborhood_rating = form.cleaned_data.get('neighborhood_rating')
            distance_to_cbd_km = form.cleaned_data.get('distance_to_cbd_km')

            # Derived feature used by the model
            property_age = datetime.now().year - year_built
            
            model = _load_model()
            # Make prediction if model is loaded
            if model is not None:
                try:
                    # Prepare features for prediction (use the same column names as training)
                    features_df = pd.DataFrame(
                        [
                            {
                                "Location": location,
                                "FloorAreaSqm": square_feet,
                                "LotSizeSqm": lot_size_sqm,
                                "Bedrooms": bedrooms,
                                "Bathrooms": bathrooms,
                                "ParkingSpaces": parking_spaces,
                                "PropertyAge": property_age,
                                "Condition": condition,
                                "PropertyType": property_type,
                                "NeighborhoodRating": neighborhood_rating,
                                "DistanceToCBDKm": distance_to_cbd_km,
                            }
                        ]
                    )

                    # Predict price
                    predicted_price = float(model.predict(features_df)[0])

                    # Confidence band (simple): predicted ± CV MAE (if available)
                    metrics = _load_metrics() or {}
                    cv_mae = None
                    try:
                        cv_mae = float(metrics.get("cv", {}).get("mae_mean"))
                    except Exception:
                        cv_mae = None

                    lower_price = max(0.0, predicted_price - cv_mae) if cv_mae else None
                    upper_price = predicted_price + cv_mae if cv_mae else None
                    
                    # Save prediction to database
                    prediction = PredictionHistory.objects.create(
                        user=request.user,
                        location=location,
                        square_feet=square_feet,
                        lot_size_sqm=lot_size_sqm,
                        bedrooms=bedrooms,
                        bathrooms=bathrooms,
                        parking_spaces=parking_spaces,
                        year_built=year_built,
                        property_type=property_type,
                        condition=condition,
                        neighborhood_rating=neighborhood_rating,
                        distance_to_cbd_km=distance_to_cbd_km,
                        predicted_price=predicted_price
                    )
                    
                    messages.success(request, f'Price predicted successfully: ₱{predicted_price:,.2f}')
                    return render(request, 'predictions/result.html', {
                        'prediction': prediction,
                        'lower_price': lower_price,
                        'upper_price': upper_price,
                        'cv_mae': cv_mae,
                        'model_name': metrics.get('best_model'),
                        'trained_at_utc': metrics.get('trained_at_utc'),
                        'form': form
                    })
                except Exception as e:
                    messages.error(request, f'Error making prediction: {str(e)}')
            else:
                messages.error(request, 'ML model not loaded. Please train the model first (ml_training/train_model.py).')
    else:
        form = PredictionForm()
    
    return render(request, 'predictions/predict.html', {'form': form})


@login_required
def prediction_history(request):
    """
    View to display user's prediction history with chart visualization.
    """
    predictions = PredictionHistory.objects.filter(user=request.user)
    
    # Prepare data for chart
    labels = [pred.created_at.strftime('%Y-%m-%d') for pred in predictions]
    prices = [float(pred.predicted_price) for pred in predictions]
    
    context = {
        'predictions': predictions,
        'labels': labels,
        'prices': prices
    }
    
    return render(request, 'predictions/history.html', context)


@login_required
def export_history_csv(request):
    """
    Export user's prediction history as CSV.
    """
    predictions = PredictionHistory.objects.filter(user=request.user).order_by("-created_at")
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="prediction_history.csv"'

    writer = csv.writer(response)
    writer.writerow(
        [
            "Date",
            "Location",
            "FloorAreaSqm",
            "LotSizeSqm",
            "Bedrooms",
            "Bathrooms",
            "ParkingSpaces",
            "YearBuilt",
            "PropertyType",
            "Condition",
            "NeighborhoodRating",
            "DistanceToCBDKm",
            "PredictedPricePHP",
        ]
    )
    for p in predictions:
        writer.writerow(
            [
                p.created_at.isoformat(),
                p.location,
                p.square_feet,
                p.lot_size_sqm,
                p.bedrooms,
                p.bathrooms,
                p.parking_spaces,
                p.year_built,
                p.property_type,
                p.condition,
                p.neighborhood_rating,
                p.distance_to_cbd_km,
                float(p.predicted_price),
            ]
        )

    return response


@login_required
def profile(request):
    """
    User profile view showing statistics and recent predictions.
    """
    predictions = PredictionHistory.objects.filter(user=request.user)
    
    # Calculate statistics
    total_predictions = predictions.count()
    avg_price = predictions.aggregate(Avg('predicted_price'))['predicted_price__avg'] or 0
    max_price = predictions.aggregate(Max('predicted_price'))['predicted_price__max'] or 0
    min_price = predictions.aggregate(Min('predicted_price'))['predicted_price__min'] or 0
    
    recent_predictions = predictions[:5]
    
    context = {
        'total_predictions': total_predictions,
        'avg_price': avg_price,
        'max_price': max_price,
        'min_price': min_price,
        'recent_predictions': recent_predictions
    }
    
    return render(request, 'predictions/profile.html', context)
