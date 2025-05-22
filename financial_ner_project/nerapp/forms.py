from django import forms

class TextInputForm(forms.Form):
    user_input = forms.CharField(label='Enter Financial Text', widget=forms.Textarea)
