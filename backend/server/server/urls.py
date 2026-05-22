from django.contrib import admin
from django.urls import path, include
from django.http import JsonResponse

from apps.endpoints.urls import urlpatterns as endpoints_urlpatterns

def welcome(request):
    """Welcome page for the ML Service API"""
    return JsonResponse({
        'message': 'Welcome to Django ML Service API',
        'version': '1.0',
        'endpoints': {
            'admin': '/admin/',
            'api': '/api/v1/',
            'endpoints': '/api/v1/endpoints',
            'algorithms': '/api/v1/mlalgorithms',
            'predict': '/api/v1/{endpoint_name}/predict',
        }
    })

urlpatterns = [
    path('', welcome, name='welcome'),
    path('admin/', admin.site.urls),
]

urlpatterns += endpoints_urlpatterns
