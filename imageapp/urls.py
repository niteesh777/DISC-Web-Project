from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),                        # Home page
    path('upload/', views.upload_images, name='upload_images'),  # Upload page
    path('result/<int:pk>/', views.result, name='result'),    # Result page
    path('check_status/<int:pk>/', views.check_processing_status, name='check_status'),  # Status check
]


# from django.urls import path
# from . import views

# urlpatterns = [
    
#     # ... other URL patterns ...
# ]

