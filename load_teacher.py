import torchvision.models.detection as detection

def load_teacher_model():
    teacher = detection.fasterrcnn_resnet50_fpn(pretrained=True)
    teacher.eval()  # Freeze the teacher
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher

if __name__ == "__main__":
    teacher = load_teacher_model()
    print("Teacher model loaded and frozen.")
