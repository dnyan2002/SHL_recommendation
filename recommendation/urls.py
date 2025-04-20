from django.urls import path
from .views import HealthCheckView, RecommendationView

urlpatterns = [
    path('health/', HealthCheckView.as_view(), name='health_check'),
    path('recommend/', RecommendationView.as_view(), name='recommend'),
]