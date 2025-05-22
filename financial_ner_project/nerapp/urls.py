from django.urls import path
from .views import extract_entities

urlpatterns = [
    path('', extract_entities, name='home'),
]
