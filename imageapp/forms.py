from django import forms
from .models import ImageUpload

class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = ImageUpload
        fields = ['image1', 'image2']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Make file inputs optional, as they are not used if captured images are provided.
        self.fields['image1'].required = False
        self.fields['image2'].required = False
