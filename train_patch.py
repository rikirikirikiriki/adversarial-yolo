"""
Training code for Adversarial patch training


"""
import argparse
import gc
import subprocess
import sys
import time
from pathlib import Path
import PIL
import load_data
from tqdm import tqdm

from load_data import *
import gc
import matplotlib.pyplot as plt
from torch import autograd
from torchvision import transforms
from tensorboardX import SummaryWriter


import patch_config

import torch
from torch.cuda.amp import GradScaler, autocast
from ultralytics import YOLO

class PatchTrainer(object):
    def __init__(self, mode):
        self.config = patch_config.patch_configs[mode]()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.darknet_model = Darknet(self.config.cfgfile)
        # self.darknet_model.load_weights(self.config.weightfile)
        # self.darknet_model = self.darknet_model.eval().cuda() # TODO: Why eval?
        # YOLOv11 仅使用单一的 .pt 权重文件，不再像 YOLOv3 那样需要 cfg/weights 两段式加载；
        # 配置中的 weightfile 会直接传给 ultralytics.YOLO 以保持自定义模型（如 yolo11-visdrone）的灵活性。
        self.model_path = Path(self.config.weightfile)
        self.darknet_model = YOLO(self.model_path)  # 通过 YOLO 类替代 Darknet(cfg+weights) 的旧加载方式
        # ultralytics.YOLO 不是 nn.Module；实际的可训练网络位于 .model，下行确保网络驻留在 GPU 并冻结参数避免不必要的显存开销。
        self.darknet_model.model.to(self.device).eval().requires_grad_(False)
        self.scaler = GradScaler(enabled=torch.cuda.is_available())
        self.target_cls = getattr(self.config, 'target_cls', 0)
        raw_imgsz = getattr(self.darknet_model.model, 'args', None)
        if isinstance(raw_imgsz, dict):
            imgsz = raw_imgsz.get('imgsz', 640)
        else:
            imgsz = getattr(raw_imgsz, 'imgsz', 640)
        if isinstance(imgsz, (list, tuple)):
            self.model_height = imgsz[0]
            self.model_width = imgsz[1] if len(imgsz) > 1 else imgsz[0]
        else:
            self.model_height = self.model_width = imgsz

        self.patch_applier = PatchApplier().cuda()
        self.patch_transformer = PatchTransformer().cuda()
        #self.prob_extractor = MaxProbExtractor(0, 80, self.config).cuda()
        self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).cuda()
        self.total_variation = TotalVariation().cuda()

        self.writer = self.init_tensorboard(mode)

    def init_tensorboard(self, name=None):
        subprocess.Popen(['tensorboard', '--logdir=runs'])
        if name is not None:
            time_str = time.strftime("%Y%m%d-%H%M%S")
            return SummaryWriter(f'runs/{time_str}_{name}')
        else:
            return SummaryWriter()

    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """

        #img_size = self.darknet_model.height
        img_size = self.config.img_size # 随配置读取，匹配 yolo11
        img_size = self.model_height
        batch_size = self.config.batch_size
        n_epochs = self.config.max_epoch
        # 使用配置中的 max_lab 限制每张图参与贴纸叠加的目标数，避免在密集场景中过多复制 patch 导致显存暴涨。
        max_lab = getattr(self.config, 'max_lab', 200)

        time_str = time.strftime("%Y%m%d-%H%M%S")
        save_dir = Path("saved_patches")
        save_dir.mkdir(parents=True, exist_ok=True)

        # Generate stating point
        adv_patch_cpu = self.generate_patch("gray").to(self.device)
        #adv_patch_cpu = self.read_image("saved_patches/patchnew0.jpg")

        adv_patch_cpu.requires_grad_(True)

        train_loader = torch.utils.data.DataLoader(InriaDataset(self.config.img_dir, self.config.lab_dir, max_lab, img_size,
                                                                shuffle=True),
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=10)
        self.epoch_length = len(train_loader)
        print(f'One epoch is {len(train_loader)}')

        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)

        et0 = time.time()
        for epoch in range(n_epochs):
            ep_det_loss = 0
            ep_nps_loss = 0
            ep_tv_loss = 0
            # ep_loss 用于日志/调度器，必须存储数值而非带计算图的 Tensor，
            # 否则会在多个 batch 间累积梯度历史，快速占满显存。
            ep_loss = 0.0
            bt0 = time.time()
            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                        total=self.epoch_length):
                with autograd.detect_anomaly():
                    img_batch = img_batch.to(self.device, non_blocking=True)
                    lab_batch = lab_batch.to(self.device, non_blocking=True)
                    #print('TRAINING EPOCH %i, BATCH %i'%(epoch, i_batch))
                    adv_patch = adv_patch_cpu.to(self.device)
                    # PyTorch<2.0 的 `torch.cuda.amp.autocast` 不接受 device_type 参数，
                    # 在此保持最小兼容写法仅通过 enabled 控制 AMP 开关。
                    with autocast(enabled=torch.cuda.is_available()):
                        adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=False)
                        p_img_batch = self.patch_applier(img_batch, adv_batch_t)
                        #p_img_batch = F.interpolate(p_img_batch, (self.darknet_model.height, self.darknet_model.width))
                        p_img_batch = F.interpolate(p_img_batch, (img_size, img_size))

                        img = p_img_batch[1, :, :,]
                        img = transforms.ToPILImage()(img.detach().cpu())
                        #img.show()


                        output = self.darknet_model.model(p_img_batch)

                    # YOLOv11 混合精度下偶发 NaN/Inf，若预测包含非有限值则跳过本 batch，避免 SigmoidBackward 报错。
                    def _get_output_tensor(raw_out):
                        if isinstance(raw_out, torch.Tensor):
                            return raw_out
                        if isinstance(raw_out, (list, tuple)) and len(raw_out) > 0:
                            first = raw_out[0]
                            if isinstance(first, torch.Tensor):
                                return first
                            if hasattr(first, 'boxes') and hasattr(first.boxes, 'data'):
                                return first.boxes.data
                        return None

                    out_tensor = _get_output_tensor(output)
                    if out_tensor is not None and not torch.isfinite(out_tensor).all():
                        print(f"[warn] skip batch {i_batch} due to non-finite model output")
                        optimizer.zero_grad(set_to_none=True)
                        del adv_batch_t, output, p_img_batch
                        torch.cuda.empty_cache()
                        continue
                        #max_prob = self.prob_extractor(output)

                        results = None
                        if isinstance(output, list) and len(output) > 0 and hasattr(output[0], 'boxes'):
                            results = output
                        elif hasattr(output, 'boxes'):
                            results = [output]

                        if results is not None:
                            boxes = results[0].boxes
                            obj_scores = boxes.conf.to(p_img_batch.device)
                            probs = getattr(results[0], 'probs', None)

                            if probs is not None:
                                cls_scores = probs.data.to(p_img_batch.device)
                                combined_scores = self.config.loss_target(obj_scores.unsqueeze(-1), cls_scores)
                                max_prob = combined_scores.max(dim=1).values
                            else:
                                # yolo11 输出格式变化：此处手动取 conf/cls 构造对抗损失
                                max_cls_score = torch.ones_like(obj_scores)
                                combined_scores = self.config.loss_target(obj_scores, max_cls_score)
                                max_prob = combined_scores
                        else:
                            # 当 YOLOv11 前向仅返回原始预测张量（而非 Results 对象）时，直接从输出张量中取 obj 与 cls 打分，避免旧版 prob_extractor 依赖。
                            raw_preds = output[0] if isinstance(output, (list, tuple)) else output
                            if isinstance(raw_preds, torch.Tensor):
                                obj_scores = raw_preds[..., 4]
                                cls_scores = raw_preds[..., 5:] if raw_preds.shape[-1] > 5 else torch.ones_like(obj_scores).unsqueeze(-1)
                                combined_scores = self.config.loss_target(obj_scores.unsqueeze(-1), cls_scores)
                                max_prob = combined_scores.max(dim=-1).values
                            else:
                                raise RuntimeError(f"Unsupported YOLO output type: {type(raw_preds)}")

                        nps = self.nps_calculator(adv_patch)
                        tv = self.total_variation(adv_patch)


                        nps_loss = nps*0.01
                        tv_loss = tv*2.5
                        det_loss = torch.mean(max_prob)
                        loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1, device=self.device))

                    # YOLOv11 的 head 在半精度下偶发 NaN（见 SigmoidBackward0 报错），若当前 batch 出现非有限值，
                    # 则跳过本 batch 以防训练中断，同时释放中间显存。
                    if not torch.isfinite(loss).all():
                        print(f"[warn] skip batch {i_batch} due to non-finite loss")
                        optimizer.zero_grad(set_to_none=True)
                        del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss
                        torch.cuda.empty_cache()
                        continue

                    ep_det_loss += det_loss.detach().cpu().numpy()
                    ep_nps_loss += nps_loss.detach().cpu().numpy()
                    ep_tv_loss += tv_loss.detach().cpu().numpy()
                    # 仅保存当前 batch 的标量值，避免把完整的 autograd 图堆叠到 ep_loss 上。
                    ep_loss += loss.detach().item()

                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
                    adv_patch_cpu.data.clamp_(0,1)       #keep patch in image range

                    bt1 = time.time()
                    if i_batch%5 == 0:
                        iteration = self.epoch_length * epoch + i_batch

                        self.writer.add_scalar('total_loss', loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/nps_loss', nps_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/tv_loss', tv_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('misc/epoch', epoch, iteration)
                        self.writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)

                        self.writer.add_image('patch', adv_patch_cpu, iteration)
                    if i_batch + 1 >= len(train_loader):
                        print('\n')
                    else:
                        del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss
                        torch.cuda.empty_cache()
                    bt0 = time.time()
            et1 = time.time()
            ep_det_loss = ep_det_loss/len(train_loader)
            ep_nps_loss = ep_nps_loss/len(train_loader)
            ep_tv_loss = ep_tv_loss/len(train_loader)
            ep_loss = ep_loss/len(train_loader)

            #im = transforms.ToPILImage('RGB')(adv_patch_cpu)
            #plt.imshow(im)
            #plt.savefig(f'pics/{time_str}_{self.config.patch_name}_{epoch}.png')

            scheduler.step(ep_loss)
            if True:
                print('  EPOCH NR: ', epoch),
                print('EPOCH LOSS: ', ep_loss)
                print('  DET LOSS: ', ep_det_loss)
                print('  NPS LOSS: ', ep_nps_loss)
                print('   TV LOSS: ', ep_tv_loss)
                print('EPOCH TIME: ', et1-et0)
                #im = transforms.ToPILImage('RGB')(adv_patch_cpu)
                #plt.imshow(im)
                #plt.show()
                #im.save("saved_patches/patchnew1.jpg")
                patch_image = transforms.ToPILImage('RGB')(adv_patch_cpu.detach().cpu())
                patch_filename = f"{time_str}_{self.config.patch_name}_{epoch}.png"
                patch_path = save_dir / patch_filename
                patch_image.save(patch_path)
                print(f"Saved patch to {patch_path}")
                del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss
                torch.cuda.empty_cache()
            et0 = time.time()

    
    def extract_max_confidence(self, results):
        if results is None or len(results) == 0:
            return torch.tensor(0.0, device=next(self.darknet_model.model.parameters()).device)

        result = results[0]
        boxes = result.boxes
        probs = result.probs

        if boxes is None or boxes.data.numel() == 0:
            return torch.tensor(0.0, device=next(self.darknet_model.model.parameters()).device)

        confs = boxes.conf
        cls_ids = boxes.cls

        if probs is not None and hasattr(probs, 'shape') and probs.shape[0] == cls_ids.shape[0]:
            target_scores = probs[:, int(self.target_cls)].to(confs.device)
            confs = confs * target_scores

        target_mask = cls_ids == self.target_cls
        target_confs = confs[target_mask] if target_mask.any() else confs
        return target_confs.max() if target_confs.numel() > 0 else torch.tensor(0.0, device=confs.device)

    def generate_patch(self, type):
        """
        Generate a random patch as a starting point for optimization.

        :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
        :return:
        """
        if type == 'gray':
            adv_patch_cpu = torch.full((3, self.config.patch_size, self.config.patch_size), 0.5)
        elif type == 'random':
            adv_patch_cpu = torch.rand((3, self.config.patch_size, self.config.patch_size))

        return adv_patch_cpu

    def read_image(self, path):
        """
        Read an input image to be used as a patch

        :param path: Path to the image to be read.
        :return: Returns the transformed patch as a pytorch Tensor.
        """
        patch_img = Image.open(path).convert('RGB')
        tf = transforms.Resize((self.config.patch_size, self.config.patch_size))
        patch_img = tf(patch_img)
        tf = transforms.ToTensor()

        adv_patch_cpu = tf(patch_img)
        return adv_patch_cpu


def main():
    # if len(sys.argv) != 2:
    #     print('You need to supply (only) a configuration mode.')
    #     print('Possible modes are:')
    #     print(patch_config.patch_configs)
    parser = argparse.ArgumentParser(description='Train adversarial patches.')
    parser.add_argument('--config', default='my_visdrone', choices=patch_config.patch_configs.keys(),
                        help='Configuration key to use from patch_config.patch_configs')
    args = parser.parse_args()


    #trainer = PatchTrainer(sys.argv[1])
    trainer = PatchTrainer(args.config)
    trainer.train()

if __name__ == '__main__':
    main()


