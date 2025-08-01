from django.shortcuts import render
from django.db.models import F
from django.http import HttpResponseRedirect, HttpResponse
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views import generic
from django.utils import timezone
from django.shortcuts import render
from .forms import PredictForm
from .utils import format_amenities_from_string
from .forms import PredictForm
import pandas as pd
import numpy as np
import joblib
import os
from django.conf import settings

class IndexView(generic.TemplateView):
    template_name = "index.html"

APP_DIR = os.path.dirname(os.path.abspath(__file__))

preprocessor_path = os.path.join(APP_DIR, "preprocessor.pkl")
model_path = os.path.join(APP_DIR, "model.pkl")
clusterer_path = os.path.join(APP_DIR, "clusterer.pkl")

preprocessor = joblib.load(preprocessor_path)
preprocessor.fitted = True 
model = joblib.load(model_path)
clusterer = joblib.load(clusterer_path)

def predict_view(request):
    price = None
    formatted_amenities = None

    if request.method == 'POST':
        form = PredictForm(request.POST)
        
        if form.is_valid():
            data = form.cleaned_data
            print(data)

            data['amenities'] = format_amenities_from_string(data.get('amenities', ''))

            df = pd.DataFrame([data])

            processed = preprocessor.transform(df)
            cluster_labels = clusterer.predict(processed)
            prediction = model.predicts(processed, cluster_labels)[0]
            price = round(np.exp(prediction), 2)
            print(price)

        else:
            print(form.errors)  

    else:
        form = PredictForm()

    return render(request, 'predict.html', {
        'form': form,
        'price': price
    })
