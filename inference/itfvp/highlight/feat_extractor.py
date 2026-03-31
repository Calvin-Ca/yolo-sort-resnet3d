import torch
import torchvision
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names


class ImageFeatExactor(object):
    def __init__(self):
        pass

    def feature(self, image):
        raise NotImplementedError("Should have implemented this")


class ResNet18FeatExtractor(ImageFeatExactor):
    def __init__(self, device):
        super().__init__()
        model = create_feature_extractor(
            resnet18(pretrained=True),
            return_nodes={"flatten": "feat"}
            )
        self._model = model.eval().to(device)
        self.device = device
        normalize = transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
            )
        self._preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
            ])
    
    def feature(self, image):
        """
        image: numpy.ndarray, shape (H, W, 3) or (B, H, W, 3)
        """
        image = self._preprocess(image).to(self.device)
        single_image = False
        if image.dim() == 3:
            single_image = True
            image = image.unsqueeze(0)
        with torch.no_grad():
            feat = self._model(image)["feat"]
        if single_image:
            feat = feat.squeeze(0)
        return feat.cpu().numpy()