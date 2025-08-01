from django.urls import path, include
from django.contrib import admin
from . import views
from .views import predict_view

app_name = "polls"
urlpatterns = [
    path('', views.predict_view, name='predict'),
]
