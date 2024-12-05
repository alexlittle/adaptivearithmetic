from django.urls import path, include
from django.contrib.auth.views import LoginView
from adaptarith import views
from adaptarith.forms import UserLoginForm

app_name = 'adaptarith'
urlpatterns = [
    path("", views.HomeView.as_view(), name="index"),
    path('login/', LoginView.as_view(
            template_name="adaptarith/login.html",
            authentication_form=UserLoginForm
            ), name='login'),
    path('logout/', views.UserLogoutView.as_view(), name='logout'),

]