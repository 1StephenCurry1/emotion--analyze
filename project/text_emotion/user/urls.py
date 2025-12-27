from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    path('user', views.user),
    path('login_check', views.login_check),
    path('register',views.register),
    path('change_password', views.change_password),
    path('get_users',views.get_user),
    path('edit_user',views.edit_user),
    path('del_user',views.del_user),
    path('password', views.password),
]
