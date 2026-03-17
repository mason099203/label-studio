import os
import django
import sys

sys.path.append(r"d:\ai_test\project\label-studio\label_studio")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings.label_studio")
django.setup()

from tasks.models import Task, Annotation
from training.datasets import detect_training_interface, _extract_classification_label, _resolve_local_image_path
from projects.models import Project

project_id = 9
project = Project.objects.get(pk=project_id)
spec = detect_training_interface(project)

with open(r"d:\ai_test\project\label-studio\test_class_out.txt", "w", encoding="utf-8") as f:
    f.write(f"Spec: {spec}\n")
    tasks = Task.objects.filter(project_id=project_id)
    f.write(f"Total tasks: {tasks.count()}\n")

    for t in tasks[:3]:
        f.write(f"Task ID: {t.id}\n")
        f.write(f"Data: {t.data}\n")
        
        img_val = t.data.get(spec['data_key'])
        img_path = _resolve_local_image_path(img_val)
        f.write(f"Img: {img_val} Path: {img_path}\n")
        
        for a in t.annotations.all():
            f.write(f"  Anno ID: {a.id} cancelled: {a.was_cancelled}\n")
            f.write(f"  Result: {a.result}\n")
            label = _extract_classification_label([a], spec['control_name'])
            f.write(f"  Label extracted: {label}\n")
