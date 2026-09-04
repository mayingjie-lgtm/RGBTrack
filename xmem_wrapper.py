from estimater import *
from datareader import *
import os
# from ultralytics import YOLO
# from ultralytics.utils.plotting import Annotator, colors

XMEM_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "XMem")
sys.path.append(XMEM_PATH)
from inference.data.test_datasets import LongTestDataset, DAVISTestDataset, YouTubeVOSTestDataset
from inference.data.mask_mapper import MaskMapper
from model.network import XMem
from inference.inference_core import InferenceCore
from inference.interact.interactive_utils import image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask, overlay_davis
from torch.cuda.amp import autocast

# default configuration
config = {
    'top_k': 30,
    'mem_every': 5,
    'deep_update_every': -1,
    'enable_long_term': True,
    'enable_long_term_count_usage': True,
    'num_prototypes': 128,
    'min_mid_term_frames': 5,
    'max_mid_term_frames': 10,
    'max_long_term_elements': 10000,
}

torch.set_grad_enabled(False)
torch.cuda.empty_cache()


class XMemMaskTracker:
    """Propagate a single binary object mask with the bundled XMem model."""

    def __init__(self, model_path=None, device="cuda"):
        """Load XMem weights and create an inference processor for one object."""
        if model_path is None:
            model_path = os.path.join(XMEM_PATH, "saves", "XMem.pth")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"XMem weights not found: {model_path}")

        self.device = device
        self.config = dict(config)
        self.network = XMem(self.config, model_path).eval().to(device)
        self.processor = InferenceCore(self.network, config=self.config)
        self.processor.set_all_labels([1])
        self.initialized = False

    def initialize(self, rgb, mask):
        """Seed XMem memory from an RGB frame and its binary object mask."""
        mask = np.asarray(mask, dtype=bool)
        if rgb.shape[:2] != mask.shape:
            raise ValueError(
                f"XMem RGB shape {rgb.shape[:2]} does not match mask shape {mask.shape}"
            )
        if not np.any(mask):
            raise ValueError("XMem initialization mask is empty")

        frame_torch, _ = image_to_torch(rgb, device=self.device)
        mask_index = mask.astype(np.uint8)
        mask_torch = index_numpy_to_one_hot_torch(mask_index, 2).to(self.device)
        with autocast(enabled=self.device.startswith("cuda")):
            prediction = self.processor.step(frame_torch, mask_torch[1:])
        self.initialized = True
        return torch_prob_to_numpy_mask(prediction) == 1

    def propagate(self, rgb):
        """Propagate the initialized object mask to the next RGB frame."""
        if not self.initialized:
            raise RuntimeError("XMemMaskTracker must be initialized before propagation")

        frame_torch, _ = image_to_torch(rgb, device=self.device)
        with autocast(enabled=self.device.startswith("cuda")):
            prediction = self.processor.step(frame_torch)
        return torch_prob_to_numpy_mask(prediction) == 1


# class yolo_wrapper:
#     def __init__(self):
#         self.model= YOLO("yolo11n-seg.pt")  # segmentation model

#     def initialize(self, rgb, mask):
#         annotator=Annotator(rgb, line_width=2)