from django.db import models

class ImageUpload(models.Model):
    image1 = models.ImageField(upload_to='uploads/')
    image2 = models.ImageField(upload_to='uploads/')
    # result_heatmap = models.ImageField(upload_to='results/', null=True, blank=True)
    result_heatmap = models.ImageField(upload_to='results/Cropped/OVERLAY/Heatmap/', null=True, blank=True)
    result_score_cheek = models.FloatField(null=True, blank=True)
    result_score_forehead = models.FloatField(null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
