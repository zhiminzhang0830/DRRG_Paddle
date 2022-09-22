import paddle
import paddle.nn as nn
from paddle.utils.cpp_extension import load
custom_ops = load(
    name="custom_jit_ops",
    sources=[
        "ppocr/ext_op/roi_align_rotated/roi_align_rotated.cc",
        "ppocr/ext_op/roi_align_rotated/roi_align_rotated.cu"])  # 先测试CPU

roi_align_rotated = custom_ops.roi_align_rotated



class RoIAlignRotated(nn.Layer):
    """RoI align pooling layer for rotated proposals.

    """

    def __init__(self,
                 out_size,
                 spatial_scale,
                 sample_num=0,
                 aligned=True,
                 clockwise=False):
        super(RoIAlignRotated, self).__init__()

        if isinstance(out_size, int):
            self.out_h = out_size
            self.out_w = out_size
        elif isinstance(out_size, tuple):
            assert len(out_size) == 2
            assert isinstance(out_size[0], int)
            assert isinstance(out_size[1], int)
            self.out_h, self.out_w = out_size
        else:
            raise TypeError(
                '"out_size" must be an integer or tuple of integers')

        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)
        self.aligned = aligned
        self.clockwise = clockwise
    
    def forward(self, feats, rois):
        output = roi_align_rotated(feats, rois, self.out_h, self.out_w, self.spatial_scale, self.sample_num, self.aligned,
                                             self.clockwise)
        return output