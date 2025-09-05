# ResNet18 + RetinaNet Model Definition

from torchvision.models.detection import RetinaNet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

def create_custom_retinanet(num_classes , backbone_name = "resnet18") :
    backbone = resnet_fpn_backbone(backbone_name , weights = None , trainable_layers = 3)
    
    model = RetinaNet(backbone , num_classes = num_classes)
    
    
    return model