import argparse
import torch

from model import Darknet53

def main(
    pred_path,
    pretrained_path,
    device,
    csp    
    ):

    if device is not None:
        device = torch.device(device)
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')    

    if csp:
        model = Darknet53(num_classes=2, csp=csp)


    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', '-pp', help ='Predict Folder', type=str, default='datasets/pred')
    parser.add_argument('--pretrained_path', '-p', help='Pretrained Weights Path', type=str, default=None)
    parser.add_argument('--device', '-d', help='Device', type=str, default=None)
    parser.add_argument('--csp', help='Activate CSP', default=False)
    args = parser.parse_args()

    main(**args)
