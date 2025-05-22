from django.shortcuts import render
from .forms import TextInputForm
from .ner_model import financial_ner

from django.shortcuts import render, redirect
from django.contrib import messages
from .forms import TextInputForm
from .ner_model import financial_ner

def extract_entities(request):
    print("extract_entities view called")  
    
    if request.method == 'POST':
        form = TextInputForm(request.POST)
        print("Form submitted with data:", form)
        if form.is_valid():
            text = form.cleaned_data['user_input']
            
            # Extract entities
            result = {}
            result['companies'] = financial_ner._extract_companies(text)
            result['symbols'] = financial_ner._extract_symbols(text)
            result['industries'] = financial_ner._extract_industries(text)
            result['market_caps'] = financial_ner._extract_market_caps(text)
            result['exchanges'] = financial_ner._extract_exchanges(text)
            
            # Store results in session to persist across redirect
            request.session['extraction_results'] = result
            request.session['user_input'] = text
            
            return redirect('home')  # Redirect to GET request
    
    # Handle GET request (initial page load or after redirect)
    form = TextInputForm()
    result = {}
    
    # Check if we have results from a recent extraction
    if 'extraction_results' in request.session:
        result = request.session['extraction_results']
        # Repopulate form with previous input
        if 'user_input' in request.session:
            form = TextInputForm(initial={'user_input': request.session['user_input']})
    
    return render(request, 'nerapp/home.html', {'form': form, 'result': result})