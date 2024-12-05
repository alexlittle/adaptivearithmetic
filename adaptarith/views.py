from django.views.generic import TemplateView
from django.contrib.auth.views import LogoutView
from django.urls import reverse_lazy
from django.conf import settings



class HomeView(TemplateView):
    template_name = 'adaptarith/home.html'


class UserLogoutView(LogoutView):
    next_page = reverse_lazy('adaptarith:index')