from celery import shared_task
from .models import ImageUpload
from .processing_script import your_processing_function  # Import your processing function

@shared_task
def process_images(image_upload_id):
    image_upload = ImageUpload.objects.get(id=image_upload_id)
    image1_path = image_upload.image1.path
    image2_path = image_upload.image2.path

    # Call your existing Python script functions here
    result_heatmap_path, result_score = your_processing_function(image1_path, image2_path)
    # Update the ImageUpload instance
    cheek_score, forehead_score = result_score
    image_upload.result_heatmap = '/results/Cropped/OVERLAY/Heatmap/cropped_1000001_heatmap.png'
    image_upload.result_score_cheek = float(cheek_score)
    image_upload.result_score_forehead = float(forehead_score)
    image_upload.save()

