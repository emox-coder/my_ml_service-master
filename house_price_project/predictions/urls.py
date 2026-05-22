from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    # Home
    path('', views.home, name='home'),
    
    # Authentication
    path('register/', views.register, name='register'),
    path('login/', auth_views.LoginView.as_view(template_name='predictions/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(template_name='predictions/logout.html'), name='logout'),
    
    # Prediction
    path('predict/', views.predict_price, name='predict'),
    
    # History and Profile
    path('history/', views.prediction_history, name='history'),
    path('history/export/', views.export_history_csv, name='history_export'),
    path('profile/', views.profile, name='profile'),
]
