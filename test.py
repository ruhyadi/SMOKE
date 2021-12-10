# code from: https://github.com/xiaoMrzhang/SMOKE

import cv2
import errno
import logging
import numpy as np
import os
import torch
import tqdm

from smoke.config import cfg
from smoke.data import make_data_loader, build_test_loader
from smoke.solver.build import make_optimizer, make_lr_scheduler
from smoke.utils.check_point import DetectronCheckpointer
from smoke.engine import default_argument_parser, default_setup,launch
from smoke.utils import comm
from smoke.engine.trainer import do_train
from smoke.modeling.detector import build_detection_model

def setup(args):
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    return cfg

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def load_checkpoint(model:torch.nn.Module, model_path:str, pdb=False):
    if not os.path.exists(model_path):
        return None
    pretrained_dict = torch.load(model_path, map_location=lambda storage, loc: storage)["model"]
    pretrained_dict = {'.'.join(k.split('.')[1:]): v for k, v in pretrained_dict.items()}

    if pdb:
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                        if k in model_dict.keys() and model_dict[k].size() == v.size()}

        import pdb; pdb.set_trace()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        model.load_state_dict(pretrained_dict)
        
    model.eval()
    return model

def load_test_data(cfg):
    output_folders = []
    dataset_names = cfg.DATASETS.TEST  #tuple like('kitti_test', )
    if cfg.OUTPUT_DIR:
        for dataset_name in dataset_names:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders.append(output_folder)
    data_loaders_val = build_test_loader(cfg, is_train=False)
    return output_folders, data_loaders_val

def inference(cfg, images, targets, model, device, logger):
    if type(images) == np.ndarray:
        images = torch.from_numpy(images).float().to(device)
        if images.dim() == 3:
            images = images.permute(2, 0, 1).unsqueeze(0)
        elif images.dim() == 4:
            images = images.permute(0, 3, 1, 2)
        else:
            logger.warning("Dimision Not equal 3 or 4, please check the input images")
    # import pdb; pdb.set_trace()
    with torch.no_grad():
        output = model(images, targets)
    return output

def rad_to_matrix(rotys, N):
    device = rotys.device
    cos, sin = rotys.cos(), rotys.sin()

    i_temp = torch.tensor([[1, 0, 1],
                            [0, 1, 0],
                            [-1, 0, 1]]).to(dtype=torch.float32,
                                            device=device)
    ry = i_temp.repeat(N, 1).view(N, -1, 3)

    ry[:, 0, 0] *= cos
    ry[:, 0, 2] *= sin
    ry[:, 2, 0] *= sin
    ry[:, 2, 2] *= cos

    return ry

# Get 8 corner 3D bounding box in the camera frame, detail 
# can see the paper Eq.6
def encode_box3d(rotys, dims, locs, K, image_size):
    '''
    construct 3d bounding box for each object.
    Args:
        rotys: rotation in shape N
        dims: dimensions of objects
        locs: locations of objects

    Returns:
        box_3d in camera frame, shape(b, 2, 8)
    '''
    if len(rotys.shape) == 2:
        rotys = rotys.flatten()
    if len(dims.shape) == 3:
        dims = dims.view(-1, 3)
    if len(locs.shape) == 3:
        locs = locs.view(-1, 3)

    device = rotys.device
    N = rotys.shape[0]
    ry = rad_to_matrix(rotys, N)

    dims = dims.contiguous().view(-1, 1).repeat(1, 8)
    dims[::3, :4], dims[2::3, :4] = 0.5 * dims[::3, :4], 0.5 * dims[2::3, :4]
    dims[::3, 4:], dims[2::3, 4:] = -0.5 * dims[::3, 4:], -0.5 * dims[2::3, 4:]
    dims[1::3, :4], dims[1::3, 4:] = 0., -dims[1::3, 4:]
    index = torch.tensor([[4, 0, 1, 2, 3, 5, 6, 7],
                            [4, 5, 0, 1, 6, 7, 2, 3],
                            [4, 5, 6, 0, 1, 2, 3, 7]]).repeat(N, 1).to(device=device)
    box_3d_object = torch.gather(dims, 1, index)
    box_3d = torch.matmul(ry, box_3d_object.view(N, 3, -1))
    box_3d += locs.unsqueeze(-1).repeat(1, 1, 8)

    K = K.to(device=device)
    box3d_image = torch.matmul(K, box_3d)
    box3d_image = box3d_image[:, :2, :] / box3d_image[:, 2, :].view(
        box_3d.shape[0], 1, box_3d.shape[2]
    )

    box3d_image = (box3d_image).int()
    box3d_image[:, 0] = box3d_image[:, 0].clamp(0, image_size[1])
    box3d_image[:, 1] = box3d_image[:, 1].clamp(0, image_size[0])
    return box3d_image

# Trans camrea 3D bounding box to image 3D and 2D
def encode_box2d(K, rotys, dims, locs, img_size):
    device = rotys.device
    K = K.to(device=device)

    img_size = img_size.flatten()

    box3d = encode_box3d(rotys, dims, locs)
    box3d_image = torch.matmul(K, box3d)
    box3d_image = box3d_image[:, :2, :] / box3d_image[:, 2, :].view(
        box3d.shape[0], 1, box3d.shape[2]
    )

    xmins, _ = box3d_image[:, 0, :].min(dim=1)
    xmaxs, _ = box3d_image[:, 0, :].max(dim=1)
    ymins, _ = box3d_image[:, 1, :].min(dim=1)
    ymaxs, _ = box3d_image[:, 1, :].max(dim=1)

    xmins = xmins.clamp(0, img_size[0])
    xmaxs = xmaxs.clamp(0, img_size[0])
    ymins = ymins.clamp(0, img_size[1])
    ymaxs = ymaxs.clamp(0, img_size[1])

    bboxfrom3d = torch.cat((xmins.unsqueeze(1), ymins.unsqueeze(1),
                            xmaxs.unsqueeze(1), ymaxs.unsqueeze(1)), dim=1)

    return bboxfrom3d   

def encoder_decodr(output, K, image_size):
    clses, pred_alphas, box2d = output[:, 0], output[:, 1], output[:, 2:6]
    pred_dimensions, pred_locations = output[:, 6:9], output[:, 9:12]
    # dimensions has changed from l,h,w to h,w,l ; stupied
    # smoke/modeling/heads/smoke_head/inference line 99
    pred_dimensions = pred_dimensions.roll(shifts=1, dims=1)
    pred_rotys, scores = output[:, 12], output[:, 13]
    bbox3d = encode_box3d(pred_rotys, pred_dimensions, pred_locations, K, image_size)
    logging.info(box2d)
    return bbox3d

def draw_box_3d(image, corners, color=(0, 0, 255)):
    ''' Draw 3d bounding box in image
        corners: (8,2) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
            4 -------- 5
           /|         /|
          1 -------- 0 .
          | |        | |
          . 3 -------- 6
          |/         |/
          2 -------- 7
    '''

    # face_idx = [[0, 1, 5, 4],
    #             [1, 2, 6, 5],
    #             [2, 3, 7, 6],
    #             [3, 0, 4, 7]]
    face_idx = [[5, 4, 3, 6],
                [1, 2, 3, 4],
                [1, 0, 7, 2],
                [0, 5, 6, 7]]
    for ind_f in range(3, -1, -1):
        f = face_idx[ind_f]
        for j in range(4):
            cv2.line(image, (corners[f[j], 0], corners[f[j], 1]),
                     (corners[f[(j + 1) % 4], 0], corners[f[(j + 1) % 4], 1]), color, 2, lineType=cv2.LINE_AA)
        if ind_f == 0:
            cv2.line(image, (corners[f[0], 0], corners[f[0], 1]),
                     (corners[f[2], 0], corners[f[2], 1]), color, 1, lineType=cv2.LINE_AA)
            cv2.line(image, (corners[f[1], 0], corners[f[1], 1]),
                     (corners[f[3], 0], corners[f[3], 1]), color, 1, lineType=cv2.LINE_AA)

    return image

def main(args):
    cfg = setup(args)
    logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    model = load_checkpoint(model, "./tools/logs/model_final.pth")

    output_folder, test_data_set = load_test_data(cfg)
    for batch in test_data_set:
        images, targets, image_ids = batch["images"], batch["targets"], batch["img_ids"]
        images = images.to(device)
        output = inference(cfg, images, targets, model, device, logger)
        bbox3d = encoder_decodr(output, targets[0].get_field("K"), images.image_sizes[0])

        img = cv2.imread(os.path.join("datasets/kitti/testing/image_2",
                        image_ids[0]+".png"), 1)
        # img = cv2.resize(img, (images.image_sizes[0][1], images.image_sizes[0][0]))
        for idx in range(bbox3d.size(0)):
            bbox = bbox3d[idx]
            bbox = bbox.transpose(1,0).cpu().data.numpy()
            img = draw_box_3d(img, bbox)
        import pdb;pdb.set_trace()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)