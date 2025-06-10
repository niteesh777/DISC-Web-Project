# imageapp/views.py
import base64
from django.core.files.base import ContentFile
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from .forms import ImageUploadForm
from .models import ImageUpload
from .tasks import process_images
from .utils import validate_face_alignment

def home(request):
    return render(request, 'imageapp/home.html')

def upload_images(request):
    if request.method == 'POST':
        # Check if captured image data exists (from the capture form)
        captured_image1 = request.POST.get('capturedImage1')
        captured_image2 = request.POST.get('capturedImage2')
        if captured_image1 and captured_image2:
            try:
                # Convert Base64 captured images into Django file objects
                format1, imgstr1 = captured_image1.split(';base64,')
                ext1 = format1.split('/')[-1]
                image_data1 = ContentFile(base64.b64decode(imgstr1), name='capture1.' + ext1)

                format2, imgstr2 = captured_image2.split(';base64,')
                ext2 = format2.split('/')[-1]
                image_data2 = ContentFile(base64.b64decode(imgstr2), name='capture2.' + ext2)
            except Exception as e:
                print("DEBUG: Error processing captured images:", e)
                return render(request, 'imageapp/upload_or_capture.html', {
                    'form': ImageUploadForm(),
                    'error': 'Error processing captured images. Please try again.'
                })

            image_upload = ImageUpload.objects.create(image1=image_data1, image2=image_data2)
            if not validate_face_alignment(image_upload.image1.path, image_upload.image2.path):
                image_upload.delete()
                return render(request, 'imageapp/upload_or_capture.html', {
                    'form': ImageUploadForm(),
                    'error': 'The faces in the captured images do not match in position or size. Please capture again using the same framing.'
                })
            process_images.delay(image_upload.id)
            return redirect('result', pk=image_upload.id)
        else:
            # Process file uploads
            form = ImageUploadForm(request.POST, request.FILES)
            if form.is_valid():
                image_upload = form.save()
                # Now check the uploaded images using face alignment verification
                if not validate_face_alignment(image_upload.image1.path, image_upload.image2.path):
                    image_upload.delete()
                    return render(request, 'imageapp/upload_or_capture.html', {
                        'form': form,
                        'error': 'The faces in the uploaded images do not match in position or size. Please upload again.'
                    })
                process_images.delay(image_upload.id)
                return redirect('result', pk=image_upload.id)
            else:
                return render(request, 'imageapp/upload_or_capture.html', {'form': form})
    else:
        form = ImageUploadForm()
    return render(request, 'imageapp/upload_or_capture.html', {'form': form})

def result(request, pk):
    image_upload = ImageUpload.objects.get(pk=pk)
    return render(request, 'imageapp/result.html', {'image_upload': image_upload})

def check_processing_status(request, pk):
    image_upload = get_object_or_404(ImageUpload, pk=pk)
    processing_complete = bool(image_upload.result_heatmap)
    return JsonResponse({'processing_complete': processing_complete})
