if __name__ == '__main__':
    from model import Darknet53
    import torch
    from torchvision.transforms import transforms
    import cv2

    class_list = ['Cervical', 'Lumbar']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = 'logs/CSP_SGD/160223_144706/checkpoint_best_loss_0.15974382683634758.pth'
    model = Darknet53(3, 2, csp=True).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    img = cv2.imread('datasets/val/Lumbar/193_02082014_LS_1.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224))
        ]
    )

    img = torch.unsqueeze(transform(img), dim=0).to(device)
    out = torch.argmax(model(img))
    print(class_list[out])