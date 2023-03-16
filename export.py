import argparse
import sys
import time
import warnings

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile
import torchvision.transforms

import models
from models.experimental import attempt_load, End2End
from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size
from utils.torch_utils import select_device
from utils.add_nms import RegisterNMS
import onnxruntime
import numpy as np


class ExportModel(nn.Module):
    def __init__(self, m):
        super(ExportModel, self).__init__()
        self.m = m
        
        self.denormalize = torchvision.transforms.Normalize(
                mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std = [1/0.229, 1/0.224,1/0.225]
                )
    def __len__(self):
        return len(self.m)

    def forward(self, x):
        _, _, input_h, input_w = x.shape
        
        x = self.denormalize(x) 
        outputs = self.m(
            x)  # [batch, num, 5 + num_classes] xmin, ymin, width, height, object_score, cls1_score, cls2_score ...
        print(f"output.shape:{outputs.shape}")
        '''
        box = outputs[...,0:4]
        score = outputs[...,4]
        cls = outputs[...,5]
        ymin = (box[:, 1] - box[:, 3] / 2) / input_h
        xmin = (box[:, 0] - box[:, 2] / 2) /input_w
        ymax = (box[:, 1] + box[:, 3]) /input_h
        xmax = (box[:, 0] + box[:, 2]) / input_w
        bbox = torch.stack([ymin, xmin, ymax, xmax], dim=2)
        '''
        ymin = (outputs[..., 1] - outputs[..., 3] / 2) / input_h
        xmin = (outputs[..., 0] - outputs[..., 2] / 2) / input_w
        ymax = (outputs[..., 1] + outputs[..., 3] / 2) / input_h
        xmax = (outputs[..., 0] + outputs[..., 2] / 2) / input_w
        bbox = torch.stack([ymin, xmin, ymax, xmax], dim=2)
        class_score, class_id = torch.max(outputs[..., 5:], -1)

        class_id = (class_id + 1).to(torch.float)  # add dummy background classes
        objectness = outputs[..., 4]
        return bbox, objectness, class_score #bbox, score, cls
        #return torch.Tensor([bbox[:100,]]), score[:100], cls[:100]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolor-csp-c.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes')
    parser.add_argument('--dynamic-batch', action='store_true', help='dynamic batch onnx for tensorrt and onnx-runtime')
    parser.add_argument('--grid', action='store_true', help='export Detect() layer grid')
    parser.add_argument('--end2end', action='store_true', help='export end2end onnx')
    parser.add_argument('--max-wh', type=int, default=None, help='None for tensorrt nms, int value for onnx-runtime nms')
    parser.add_argument('--topk-all', type=int, default=100, help='topk objects for every images')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='iou threshold for NMS')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='conf threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--simplify', action='store_true', help='simplify onnx model')
    parser.add_argument('--include-nms', action='store_true', help='export end2end onnx')
    parser.add_argument('--fp16', action='store_true', help='CoreML FP16 half-precision export')
    parser.add_argument('--int8', action='store_true', help='CoreML INT8 quantization')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    opt.dynamic = opt.dynamic and not opt.end2end
    opt.dynamic = False if opt.dynamic_batch else opt.dynamic
    print(opt)
    set_logging()
    t = time.time()

    # Load PyTorch model
    device = select_device(opt.device)
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    labels = model.names

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.img_size).to(device)  # image size(1,3,320,192) iDetection

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        # elif isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # assign forward (optional)
    model.model[-1].export = not opt.grid  # set Detect() layer grid export
    y = model(img)  # dry run
    if opt.include_nms:
        model.model[-1].include_nms = True
        y = None

    # TorchScript export
    try:
        print('\nStarting TorchScript export with torch %s...' % torch.__version__)
        f = opt.weights.replace('.pt', '.torchscript.pt')  # filename
        ts = torch.jit.trace(model, img, strict=False)
        ts.save(f)
        print('TorchScript export success, saved as %s' % f)
    except Exception as e:
        print('TorchScript export failure: %s' % e)

    # CoreML export
    try:
        import coremltools as ct

        print('\nStarting CoreML export with coremltools %s...' % ct.__version__)
        # convert model from torchscript and apply pixel scaling as per detect.py
        ct_model = ct.convert(ts, inputs=[ct.ImageType('image', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])])
        bits, mode = (8, 'kmeans_lut') if opt.int8 else (16, 'linear') if opt.fp16 else (32, None)
        if bits < 32:
            if sys.platform.lower() == 'darwin':  # quantization only supported on macOS
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning)  # suppress numpy==1.20 float warning
                    ct_model = ct.models.neural_network.quantization_utils.quantize_weights(ct_model, bits, mode)
            else:
                print('quantization only supported on macOS, skipping...')

        f = opt.weights.replace('.pt', '.mlmodel')  # filename
        ct_model.save(f)
        print('CoreML export success, saved as %s' % f)
    except Exception as e:
        print('CoreML export failure: %s' % e)
                     
    # TorchScript-Lite export
    try:
        print('\nStarting TorchScript-Lite export with torch %s...' % torch.__version__)
        f = opt.weights.replace('.pt', '.torchscript.ptl')  # filename
        tsl = torch.jit.trace(model, img, strict=False)
        tsl = optimize_for_mobile(tsl)
        tsl._save_for_lite_interpreter(f)
        print('TorchScript-Lite export success, saved as %s' % f)
    except Exception as e:
        print('TorchScript-Lite export failure: %s' % e)

    # ONNX export
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights.replace('.pt', '.onnx')  # filename
        model.eval()
        output_names = ['classes', 'boxes'] if y is None else ['output']
        dynamic_axes = None
        if opt.dynamic:
            dynamic_axes = {'images': {0: 'batch', 2: 'height', 3: 'width'},  # size(1,3,640,640)
             'output': {0: 'batch', 2: 'y', 3: 'x'}}
        if opt.dynamic_batch:
            opt.batch_size = 'batch'
            dynamic_axes = {
                'images': {
                    0: 'batch',
                }, }
            if opt.end2end and opt.max_wh is None:
                output_axes = {
                    'num_dets': {0: 'batch'},
                    'det_boxes': {0: 'batch'},
                    'det_scores': {0: 'batch'},
                    'det_classes': {0: 'batch'},
                }
            else:
                output_axes = {
                    'output': {0: 'batch'},
                }
            dynamic_axes.update(output_axes)
        if opt.grid:
            if opt.end2end:
                print('\nStarting export end2end onnx model for %s...' % 'TensorRT' if opt.max_wh is None else 'onnxruntime')
                model = End2End(model,opt.topk_all,opt.iou_thres,opt.conf_thres,opt.max_wh,device,len(labels))
                if opt.end2end and opt.max_wh is None:
                    output_names = ['num_dets', 'det_boxes', 'det_scores', 'det_classes']
                    shapes = [opt.batch_size, 1, opt.batch_size, opt.topk_all, 4,
                              opt.batch_size, opt.topk_all, opt.batch_size, opt.topk_all]
                else:
                    output_names = ['output']
            else:
                model.model[-1].concat = True
        output_names = ['detection_boxes', 'detection_scores', 'detection_classes']
        model = ExportModel(model)
        torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['input_tensor'],
                          output_names=output_names,
                          dynamic_axes=dynamic_axes)

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model

        if opt.end2end and opt.max_wh is None:
            for i in onnx_model.graph.output:
                for j in i.type.tensor_type.shape.dim:
                    j.dim_param = str(shapes.pop(0))

        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model

        # # Metadata
        # d = {'stride': int(max(model.stride))}
        # for k, v in d.items():
        #     meta = onnx_model.metadata_props.add()
        #     meta.key, meta.value = k, str(v)
        # onnx.save(onnx_model, f)
        onnx.save(onnx_model,f)
        print('ONNX export success, saved as %s' % f)
        import cv2
        img = cv2.imread("test.png")
        img = img / 255.0
        img = img.transpose(2, 0, 1)
        input_tensor = img[np.newaxis, :, :, :].astype(np.float32)
        print(input_tensor.shape)
        session = onnxruntime.InferenceSession(f, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        model_inputs = session.get_inputs()
        input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        model_outputs = session.get_outputs()
        output_names = [model_outputs[i].name for i in range(len(model_outputs))]
        print(output_names)
        outputs = session.run(output_names, {input_names[0]:input_tensor})
        #outputs= outputs.tolist()
        print(outputs)
        print(outputs[0].shape)
        log = open("log.txt", 'w')
        print(outputs[0][0][0])
        print(outputs[0].shape)
        print(outputs[1].shape)
         
        print(outputs[2].shape)
        #for (box, score, cls) in zip(outputs[0][0], outputs[1][0], outputs[2][0]):
        size = len(outputs[1][0].tolist())
        for ii in range(size):
            if outputs[1][0][ii] > 0.5:
                print(outputs[0][0][ii])
        if opt.simplify:
            try:
                import onnxsim

                print('\nStarting to simplify ONNX...')
                onnx_model, check = onnxsim.simplify(onnx_model)
                assert check, 'assert check failed'
            except Exception as e:
                print(f'Simplifier failure: {e}')

        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        # onnx.save(onnx_model,f)
        # print('ONNX export success, saved as %s' % f)

        if opt.include_nms:
            print('Registering NMS plugin for ONNX...')
            mo = RegisterNMS(f)
            mo.register_nms()
            mo.save(f)

    except Exception as e:
        print('ONNX export failure: %s' % e)

    # Finish
    print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))
