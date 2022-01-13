from __future__ import print_function

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from retinaface.data import cfg_mnet
from retinaface.layers.functions.prior_box import PriorBox
from retinaface.loader import load_model
from retinaface.utils.box_utils import decode, decode_landm
from retinaface.utils.nms.py_cpu_nms import py_cpu_nms

import pdb


class RetinafaceDetector:
    def __init__(self, net="mnet", type="cuda"):
        cudnn.benchmark = True
        self.net = net
        self.device = torch.device(type)
        self.model = load_model(net).to(self.device)
        self.model.eval()
        # net = 'mnet'
        # type = 'cuda'
        # self.model -- RetinaFace(...)

    def detect_faces(self, img_raw, confidence_threshold=0.9, top_k=5000, nms_threshold=0.4, keep_top_k=750, resize=1):
        # confidence_threshold = 0.9
        # top_k = 5000
        # nms_threshold = 0.4
        # keep_top_k = 750
        # resize = 1

        # img_raw.shape -- (250, 250, 3)
        img = np.float32(img_raw)
        im_height, im_width = img.shape[:2]
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        # img.size(), img.min(), img.max() -- [1, 3, 250, 250], -119., 151.0

        scale = scale.to(self.device)
        # scale -- tensor([250., 250., 250., 250.], device='cuda:0')

        # tic = time.time()
        with torch.no_grad():
            loc, conf, landms = self.model(img)  # forward pass
            # print('net forward time: {:.4f}'.format(time.time() - tic))

        # (Pdb) loc.size() -- [1, 2688, 4]
        # (Pdb) conf.size() -- [1, 2688, 2]
        # (Pdb) landms.size() -- [1, 2688, 10]

        # cfg_mnet --
        # {'name': 'mobilenet0.25',
        # 'min_sizes': [[16, 32], [64, 128], [256, 512]],
        # 'steps': [8, 16, 32], 'variance': [0.1, 0.2],
        # 'clip': False, 'loc_weight': 2.0, 'gpu_train': True,
        # 'batch_size': 32, 'ngpu': 1, 'epoch': 250, 'decay1': 190, 'decay2': 220,
        # 'image_size': 640, 'pretrain': False,
        # 'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
        # 'in_channel': 32, 'out_channel': 64}

        # im_height, im_width -- (250, 250)
        priorbox = PriorBox(cfg_mnet, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data

        # cfg_mnet['variance'] -- [0.1, 0.2]
        # prior_data.size() -- torch.Size([2688, 4])
        boxes = decode(loc.data.squeeze(0), prior_data, cfg_mnet["variance"])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        # boxes.shape -- (2688, 4)

        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg_mnet["variance"])
        scale1 = torch.Tensor(
            [
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
            ]
        )
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        # confidence_threshold -- 0.9
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]
        # print(landms.shape)
        landms = landms.reshape(-1, 5, 2)
        # print(landms.shape)
        landms = landms.transpose(0, 2, 1)
        # print(landms.shape)
        landms = landms.reshape(-1, 10)
        # print(landms.shape)

        # dets.shape, landms.shape -- ((1, 5), (1, 10))
        # dets -- array([[ 78.220116  ,  79.84813   , 173.14445   , 195.16386   ,0.99955505]]
        # -----------------------------------------------------------------------------------
        # landms -- array([[102.47213 , 145.46236 , 125.38177 , 106.445854, 146.62794 ,
        #         118.52239 , 115.24955 , 140.87106 , 159.90822 , 156.94913 ]]

        return dets, landms


detector = RetinafaceDetector(net="mnet")
