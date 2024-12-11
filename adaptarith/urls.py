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
    path('pretest/start/', views.start_pretest, name='pretest_start'),
    path('pretest/question/', views.PreTestQuestionView.as_view(), name='pretest_question'),
    path('pretest/complete/', views.PreTestCompleteView.as_view(), name='pretest_complete'),

    path('run/', views.RunView.as_view(), name='run'),
    path('passed/', views.PassedView.as_view(), name='passed'),

]