from django.shortcuts import render
from .forms import TextInputForm
from .ner_model import financial_ner

def extract_entities(request):
    print("extract_entities view called")  
    result = {}
    form = TextInputForm()  # Default empty form
    
    if request.method == 'POST':
        form = TextInputForm(request.POST)
        print("Form submitted with data:", form)
        if form.is_valid():
            text = form.cleaned_data['user_input']
            # Keep the form populated with the submitted data
            form = TextInputForm(request.POST)
            
            # Extract entities
            result['companies'] = financial_ner._extract_companies(text)
            result['symbols'] = financial_ner._extract_symbols(text)
            result['industries'] = financial_ner._extract_industries(text)
            result['market_caps'] = financial_ner._extract_market_caps(text)
            result['exchanges'] = financial_ner._extract_exchanges(text)
    
    # For GET requests, result will be empty dict, so results won't show
    return render(request, 'nerapp/home.html', {'form': form, 'result': result})
