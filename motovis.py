import os.path as osp
import numpy as np
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import mmcv

@DATASETS.register_module()
class MotovisDataset(CustomDataset):
    """Motovis数据集 - 自动驾驶道路分割数据"""

    # 类别定义 - 基于实际的道路元素
    CLASSES = (
        'background',     # 0: 背景
        # 12类车道线标注
        'curb',          # 1: 路缘石 
        'laneline',      # 2: 车道线
        'solid',         # 3: 实线
        'dashed',        # 4: 虚线
        'white',         # 5: 白色线
        'yellow',        # 6: 黄色线
        'fishbone',      # 7: 鱼骨线
        'variable',      # 8: 可变线
        'waitingline',   # 9: 等待线
        'stopline',      # 10: 停止线
        'wide',          # 11: 宽线
        'double',        # 12: 双线
        # 4类车道多边形
        'zebra',         # 13: 斑马线
        'diversion',     # 14: 导流区
        'split_point',   # 15: 分流点
        'merge_point'    # 16: 汇流点
    )

    # 专业的道路元素调色板 (BGR格式)
    PALETTE = [
        [0, 0, 0],         # 0: background - 黑色
        # 12类车道线标注 - 使用不同色调区分
        [128, 128, 128],   # 1: curb - 灰色 (路缘石)
        [255, 255, 255],   # 2: laneline - 白色 (车道线)
        [0, 255, 0],       # 3: solid - 绿色 (实线)
        [0, 255, 255],     # 4: dashed - 青色 (虚线)
        [220, 220, 220],   # 5: white - 浅灰 (白色线)
        [0, 255, 255],     # 6: yellow - 黄色 (黄色线)
        [128, 0, 255],     # 7: fishbone - 紫色 (鱼骨线)
        [255, 128, 0],     # 8: variable - 橙色 (可变线)
        [0, 128, 255],     # 9: waitingline - 天蓝 (等待线)
        [255, 0, 0],       # 10: stopline - 红色 (停止线)
        [64, 128, 64],     # 11: wide - 深绿 (宽线)
        [128, 64, 128],    # 12: double - 紫灰 (双线)
        # 4类车道多边形 - 使用明显区分的颜色
        [255, 255, 0],     # 13: zebra - 明黄 (斑马线)
        [255, 128, 128],   # 14: diversion - 粉红 (导流区)
        [128, 255, 128],   # 15: split_point - 浅绿 (分流点)
        [128, 128, 255]    # 16: merge_point - 浅蓝 (汇流点)
    ]

    def __init__(self, **kwargs):
        super(MotovisDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.npz',  # 使用npz文件
            reduce_zero_label=False,  # 不减少标签
            **kwargs
        )

    def get_gt_seg_map_by_idx(self, idx):
        """专门为evaluation提供GT，复用LoadMotovisNPZ逻辑"""

        ann_info = self.get_ann_info(idx)
        results = {
            'ann_info': ann_info,
            'seg_prefix': self.ann_dir,
        }

        # 复用pipeline中的加载逻辑
        npz_loader = LoadMotovisNPZ(crop_height_start=480, crop_height_end=980)
        results = npz_loader(results)
        #(results.keys())
        return results['gt_semantic_seg']

class LoadMotovisNPZ:
    """
    加载NPZ文件并精确复现target_parser逻辑
    """

    def __init__(self, 
                 crop_height_start=480, 
                 crop_height_end=980,
                 reduce_zero_label=False):
        self.crop_height_start = crop_height_start
        self.crop_height_end = crop_height_end
        self.reduce_zero_label = reduce_zero_label

        # 16个通道的类别映射
        self.class_mapping = {i: i+1 for i in range(16)}

    def __call__(self, results):
        """加载NPZ并转换为分割标注"""
        # print(results.keys())
        if 'ann_info' in results:
            filename = results['ann_info']['seg_map']
        elif 'ann' in results['img_info']:
            filename = results['img_info']['ann']['seg_map']
        else:
            raise ValueError('Cannot find seg_map in results')
        
        full_path = results['seg_prefix'] +  '/' + filename

        # 加载NPZ文件
        data = np.load(full_path)
        annotations = data['arr_0']  # shape: (16, 1080, 1920)

        # 验证shape
        assert annotations.shape == (16, 1080, 1920), f"期望(16,1080,1920)，得到{annotations.shape}"

        # 步骤1: 裁剪 [:, 480:980, :] -> (16, 500, 1920)
        annotations_cropped = annotations[:, self.crop_height_start:self.crop_height_end, :]

        # 步骤2: 转换为分割图 (复现target_parser逻辑)
        segmap = self.npz_to_segmap(annotations_cropped)
        # segmap = np.ascontiguousarray(segmap.astype(np.uint8))

        # 存储结果
        results['gt_semantic_seg'] = segmap
        results['seg_fields'] = ['gt_semantic_seg']

        for key in results.get('seg_fields', []):
            if key in results and results[key] is not None:
                seg = results[key]

                # 裁剪分割图 [:, 480:980]
                if seg.shape[0] == 1080:  # 如果是原始尺寸
                    seg_cropped = seg[self.crop_height_start:self.crop_height_end, :]
                else:
                    seg_cropped = seg  # 已经是裁剪后的尺寸

                # Resize分割图 (最近邻插值)
                seg_resized = mmcv.imresize(
                    seg_cropped,
                    (896,896),
                    interpolation='nearest',
                    return_scale=False
                )

                results[key] = seg_resized

        return results

    def npz_to_segmap(self, target):
        """
        精确复现原始target_parser逻辑：
        将16通道NPZ转换为单通道分割图
        """
        ch, h, w = target.shape  # (16, 500, 1920)
        segmap = np.zeros((h, w), dtype=np.uint8)

        # 初始化background mask
        background_mask = np.ones((h, w), dtype=bool)

        # 处理所有16个通道
        for channel_id in range(16):
            class_id = channel_id + 1  # 通道0->类别1, ..., 通道15->类别16

            # 获取当前通道的mask
            channel_mask = (target[channel_id] == 1)

            # 分配类别标签
            segmap[channel_mask] = class_id

            # 更新background mask
            background_mask = background_mask & (~channel_mask)

        # 设置background区域为0
        segmap[background_mask] = 0

        return segmap