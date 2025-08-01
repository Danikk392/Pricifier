from django import forms

class PredictForm(forms.Form):
    room_type = forms.ChoiceField(
        choices=[('Entire home/apt', 'Entire home/apt'), ('Private room', 'Private room'), ('Shared room', 'Shared room')],
        label="Room Type"
    )
    
    accommodates = forms.IntegerField(
        label="Number of Guests",
        min_value=1,
        max_value=16,
        required=True
    )

    beds = forms.IntegerField(
        min_value=0,
        label="Number of Beds"
    )
    
    latitude = forms.FloatField(
        label="Latitude"
    )
    
    longitude = forms.FloatField(
        label="Longitude"
    )
    
    city = forms.ChoiceField(
        choices=[('NYC', 'New York City'), ('LA', 'Los Angeles'), ('SF', 'San Francisco'),
                 ('DC', 'Washington DC'), ('Chicago', 'Chicago'), ('Boston', 'Boston')],
        label="City"
    )
    
    description = forms.CharField(
        widget=forms.Textarea,
        required=False,
        label="Listing Description",
        help_text="Briefly describe your listing (e.g., cozy studio in downtown with great lighting and fast WiFi)"
    )
    
    amenities = forms.CharField(
        required=False,
        label="Amenities (comma-separated)",
        help_text="e.g., Smoke detector, Air conditioning, Family/kid friendly, First aid kit, Cable TV"
    )

    
