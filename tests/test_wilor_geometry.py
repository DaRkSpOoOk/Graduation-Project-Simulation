import unittest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from pose.wilor.geometry import cam_crop_to_full


@unittest.skipUnless(HAS_TORCH, "torch not installed (heavy optional dependency, see pose/wilor/requirements.txt)")
class TestCamCropToFull(unittest.TestCase):
    def test_matches_hand_computed_reference(self) -> None:
        # box centered in a 100x100 image, size 50, cam_bbox=[scale=1, tx=0, ty=0]
        cam_bbox = torch.tensor([[1.0, 0.0, 0.0]])
        box_center = torch.tensor([[50.0, 50.0]])
        box_size = torch.tensor([50.0])
        img_size = torch.tensor([[100.0, 100.0]])

        out = cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length=5000.0)

        # bs = 50*1 = 50; tz = 2*5000/50 = 200; tx=ty=0 since box is centered
        self.assertEqual(out.shape, (1, 3))
        self.assertAlmostEqual(out[0, 0].item(), 0.0, places=4)
        self.assertAlmostEqual(out[0, 1].item(), 0.0, places=4)
        self.assertAlmostEqual(out[0, 2].item(), 200.0, places=2)

    def test_offset_box_center_shifts_translation(self) -> None:
        cam_bbox = torch.tensor([[1.0, 0.0, 0.0]])
        box_center = torch.tensor([[60.0, 50.0]])  # 10px right of center
        box_size = torch.tensor([50.0])
        img_size = torch.tensor([[100.0, 100.0]])

        out = cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length=5000.0)
        # tx = 2*(60-50)/50 = 0.4
        self.assertAlmostEqual(out[0, 0].item(), 0.4, places=4)


if __name__ == "__main__":
    unittest.main()
