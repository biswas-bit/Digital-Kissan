from django.shortcuts import render


def  disease_detection(request):
    return render(request, "Ai/disease_detection.html")