from django import forms
from django.contrib.auth.forms import AuthenticationForm, UsernameField

class UserLoginForm(AuthenticationForm):
    def __init__(self, *args, **kwargs):
        super(UserLoginForm, self).__init__(*args, **kwargs)

    username = UsernameField(widget=forms.TextInput(
        attrs={'class': 'form-control', 'placeholder': ''}))
    password = forms.CharField(widget=forms.PasswordInput(
        attrs={
            'class': 'form-control',
            'placeholder': ''
        }
))


class AnswerForm(forms.Form):
    response = forms.IntegerField(
        widget=forms.NumberInput(attrs={'step': 1}),
        label="Your Answer",
        help_text="Enter your answer.",
    )