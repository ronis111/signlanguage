from django.conf import settings
from django.contrib import admin
from django.urls import path
from django.conf.urls.static import static
from signapp import views

urlpatterns = [
    path('admin/', admin.site.urls),

    path('home/', views.home, name='home'),
    path('', views.index, name='index'),

    path('texttosign/', views.TextToSign, name='texttosign'),
    path('tutorials/', views.tutorials, name='tutorials'),
    path('alphabets/', views.alphabets, name='alphabets'),
    path('numbers/', views.numbers, name='numbers'),

path('signtotext/', views.upload_image, name='signtotext'),
    path('api/detect/', views.detect, name='detect'),

    path('login/', views.loginr, name='login'),
    path('register/', views.registerr, name='register'),
]

if settings.DEBUG:
    urlpatterns += static(
        settings.STATIC_URL,
        document_root=settings.STATIC_ROOT
    )