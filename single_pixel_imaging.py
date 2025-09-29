import os
import torch
from tqdm import tqdm
from configs import paths_config, hyperparameters, global_config
import numpy as np
from forward_model.hadamard_measure import CCHadamardSensor
from utils.models_utils import load_old_G
import copy
from lpips import LPIPS
from criteria.localitly_regulizer import Space_Regulizer
from PIL import Image
import torch.nn.functional as F
from scipy.io import savemat
import os

def clear_files(file_paths):
    """
    清空指定路径列表中的所有文件内容

    参数：
        file_paths (list): 文件路径的列表
    """
    for file_path in file_paths:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"文件不存在，跳过: {file_path}")
            continue

        try:
            # 打开文件并清空内容
            with open(file_path, 'w') as file:
                file.truncate(0)
            print(f"文件已清空: {file_path}")
        except Exception as e:
            print(f"清空文件时出错: {file_path}, 错误信息: {str(e)}")

_lpips_model = LPIPS(net='alex').eval()
def log_loss_to_file(loss, filename="loss_log.txt"):
    # 构建日志内容
    log_line = f"{loss:.6f}"
    # 写入文件
    try:
        with open(filename, "a") as f:  # 'a'模式表示追加写入
            f.write(log_line + "\n")
    except Exception as e:
        print(f"写入文件失败: {str(e)}")

def preprocess_for_lpips(pil_image):
    """将PIL图像转换为LPIPS所需的张量"""
    img = np.array(pil_image).astype(np.float32) / 255.0
    img = img * 2 - 1  # 归一化到[-1, 1]
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

def image_save_1(image, path):
    image_save = np.squeeze(image) * 255
    image_save = image_save.astype(np.uint8)
    image_save = np.transpose(image_save, (1, 2, 0))
    image_target = Image.fromarray(image_save)
    image_target.save(path)

def image_save_2(image, path):
    image_save = np.squeeze(image / 2 + 0.5) * 255
    image_save = image_save.astype(np.uint8)
    image_save = np.transpose(image_save, (1, 2, 0))
    image_target = Image.fromarray(image_save)
    image_target.save(path)
def normalize_per_channel(img: torch.Tensor) -> torch.Tensor:
    """每个通道独立归一化"""
    # Assumes input shape is (C, H, W)
    mins = img.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
    maxs = img.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    ranges = torch.where(maxs - mins < 1e-8, torch.tensor(1.0, device=img.device), maxs - mins)
    return torch.clamp((img - mins) / ranges, 0.0, 1.0)
class MultiscaleLPIPS:
    def __init__(
            self,
            min_loss_res: int = 16,  # 最小尺度分辨率
            level_weights=hyperparameters.muti_lpips_weight,  # 各分辨率权重
            l1_weight: float = 0.0  # L1正则化权重
    ):
        super().__init__()
        self.min_loss_res = min_loss_res
        self.weights = level_weights
        self.l1_weight = l1_weight
        self.lpips_network = LPIPS(net="vgg", verbose=False).cuda()  # 初始化感知网络

    def measure_lpips(self, x, y):
        return self.lpips_network(x, y, normalize=True).mean()

    def __call__(self, x, y):
        losses = []
        for weight in self.weights:
            if y.shape[-1] <= self.min_loss_res:
                break
            if weight > 0:
                loss = self.measure_lpips(x, y)
                losses.append(weight * loss)
            x = F.avg_pool2d(x, 2)
            y = F.avg_pool2d(y, 2)
        # 将所有尺度上的损失求和，得到总损失。
        total = torch.stack(losses).sum(dim=0) if len(losses) > 0 else 0.0
        l1 = self.l1_weight * F.l1_loss(x, y)

        return total + l1


class CS_slover(object):
    def __init__(self, sample_num, image, mission='project'):
        assert image.shape[0] == 3
        image_save_2(image, "E:\\PTI-main-old\\result\\target_image_cs.jpg")

        if hyperparameters.gray_to_rgb:
            image = 0.2989 * image[0, :, :] + 0.5870 * image[1, :, :] + 0.1140 * image[2, :, :]
            image = np.concatenate([np.expand_dims(image, axis=0)] * 3, axis=0)

        self.target = torch.tensor(image, dtype=torch.float32, device='cuda')
        savemat('E:\\PTI-main-old\\result\\target_cs.mat',
                {'target_cs': np.transpose(np.squeeze(self.target.detach().cpu().numpy()), [1, 2, 0])})
        self.sensor = CCHadamardSensor(sample_num, (image.shape[1], image.shape[2]))
        mea_np = self.sensor._np(image)
        self.mea_true = torch.tensor(mea_np, dtype=torch.float32, device=global_config.device)

        image_ATy = self.sensor.trans_np_3(mea_np)
        self.image_ATy = torch.tensor(image_ATy, device=global_config.device, dtype=torch.float32)
        self.image_ATy = normalize_per_channel(self.image_ATy)

        image_save_1(self.image_ATy.detach().cpu().numpy(), "E:\\PTI-main-old\\result\\hadamard_image_cs.jpg")
        savemat('E:\\PTI-main-old\\result\\target_cs.mat',
                {'target_cs': np.transpose(np.squeeze(self.target.detach().cpu().numpy()), [1, 2, 0])})

        if mission == 'prior':
            self.G = torch.load(paths_config.G_prior_path_cs).to(global_config.device)
        if mission == 'project':
            self.G = load_old_G()

        w_path_dir = f'{paths_config.solve_base_dir}/{paths_config.input_data_id}'
        os.makedirs(w_path_dir, exist_ok=True)
        os.makedirs(f'{w_path_dir}/{paths_config.pti_results_keyword}', exist_ok=True)

        lpips_loss = LPIPS(net=hyperparameters.lpips_type).to(global_config.device).eval()
        self.space_regulizer = Space_Regulizer(self.G, lpips_loss)
        file_paths = [
            'E:\\PTI-main-old\\result\\loss_total_phase1.txt',
            'E:\\PTI-main-old\\result\\loss_total_phase2.txt',
            'E:\\PTI-main-old\\result\\loss_total_phase3.txt',
            'E:\\PTI-main-old\\result\\loss_measure_phase1.txt',
            'E:\\PTI-main-old\\result\\loss_measure_phase2.txt',
            'E:\\PTI-main-old\\result\\loss_measure_phase3.txt',
            'E:\\PTI-main-old\\result\\loss_lpip_phase1.txt',
            'E:\\PTI-main-old\\result\\loss_lpip_phase2.txt',
            'E:\\PTI-main-old\\result\\loss_lpip_phase3.txt',
        ]
    def solver1(self,
                num_steps=hyperparameters.solver1_step,
                initial_w=None,  # 要么None，要么输入必须为[1,1,512]
                noise_bufs_step1=None,
                initial_learning_rate=hyperparameters.solver1_lr,
                measure_loss_weight=hyperparameters.solver_mea_weight,
                lpip_loss_weight=hyperparameters.solver_lpip_weight,
                regularize_noise_weight=1e5,
                w_avg_samples=10000,
                initial_noise_factor=0.005,
                lr_rampdown_length=0.75,
                lr_rampup_length=0.02,
                noise_ramp_length=0.75,
                device=global_config.device
                ):

        G = copy.deepcopy(self.G).eval().requires_grad_(False).to(device).float()
        # Compute w stats.
        z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
        w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
        w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
        w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
        w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5
        start_w = initial_w if initial_w is not None else w_avg
        w_opt = torch.tensor(start_w, dtype=torch.float32, device=device, requires_grad=True)
        num_ws = G.mapping.num_ws

        # Init noise.
        noise_bufs = {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}
        for buf in noise_bufs.values():
            buf.data[:] = torch.randn_like(buf)
            buf.requires_grad = True
        if noise_bufs_step1 is not None:
            for name, buf in G.synthesis.named_buffers():
                if name in noise_bufs_step1:
                    buf.data.copy_(noise_bufs_step1[name].data)

        optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999),
                                     lr=initial_learning_rate)
        lpips_mea = MultiscaleLPIPS()

        for step in tqdm(range(num_steps)):
            # Learning rate schedule.
            t = step / num_steps
            w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
            lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
            lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
            lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
            lr = initial_learning_rate * (lr_ramp + 1)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            ws = w_opt.repeat([1, num_ws, 1])
            ws = ws + torch.randn_like(ws) * w_noise_scale
            synth_images = G.synthesis(ws, noise_mode='const', force_fp32=True)
            filename = f'E:\\PTI-main-old\\result\\phase1\\output_{step}.mat'
            savemat(filename,
                    {'output_cs': np.transpose(np.squeeze(synth_images.detach().cpu().numpy()), [1, 2, 0])})
            if step % 20 == 0:
                image_save_2(synth_images.detach().cpu().numpy(), "E:\\PTI-main-old\\result\\proj_image_cs.jpg")
                savemat('E:\\PTI-main-old\\result\\output_cs.mat',
                        {'output_cs': np.transpose(np.squeeze(synth_images.detach().cpu().numpy()), [1, 2, 0])})
            if hyperparameters.gray_to_rgb:
                synth_images = 0.2989 * synth_images[:, 0, :, :] + 0.5870 * synth_images[:, 1, :, :] + 0.1140 * synth_images[:, 2, :, :]
                synth_images = torch.cat([synth_images] * 3, dim=0).unsqueeze(0)

            downsampled_image = F.interpolate(synth_images, size=(self.target.shape[2], self.target.shape[2]),
                                              mode='bilinear', align_corners=False)

            mea_synth = self.sensor(downsampled_image)
            image_ATy_synth = self.sensor.trans_torch_3(mea_synth)
            image_ATy_synth = normalize_per_channel(image_ATy_synth)

            if step % 20 == 0:
                image_save_1(image_ATy_synth.detach().cpu().numpy(), "E:\\PTI-main-old\\result\\proj_imageATy_cs.jpg")

            # Noise regularization.
            reg_loss = 0.0
            for v in noise_bufs.values():
                noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
                while True:
                    reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                    reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2)

            loss_measure = F.mse_loss(mea_synth, self.mea_true, reduction='mean')
            loss_lpip = lpips_mea(image_ATy_synth, self.image_ATy)
            loss = measure_loss_weight * loss_measure + lpip_loss_weight * loss_lpip + reg_loss * regularize_noise_weight
            log_loss_to_file(loss.detach().cpu().numpy(), filename="E:\\PTI-main-old\\result\\loss_total_phase1.txt")
            log_loss_to_file(loss_measure.detach().cpu().numpy(), filename="E:\\PTI-main-old\\result\\loss_measure_phase1.txt")
            log_loss_to_file(loss_lpip.detach().cpu().numpy(), filename="E:\\PTI-main-old\\result\\loss_lpip_phase1.txt")
            if step % 20 == 0:
                print('loss_total:',loss.detach().cpu().numpy(),
                      'loss_mea:',  loss_measure.detach().cpu().numpy(),
                      'loss_lpips', loss_lpip.detach().cpu().numpy(),
                      )
                print('learning_rate:', lr)
            # Step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Normalize noise.
            with torch.no_grad():
                for buf in noise_bufs.values():
                    buf -= buf.mean()
                    buf *= buf.square().mean().rsqrt()

        del G
        return w_opt.repeat([1, num_ws, 1]), noise_bufs

    def solver2(self,
                num_steps=hyperparameters.solver2_step,
                initial_w=None,  # 输入为[1, num, 512]或[1, 1, 512]或None
                noise_bufs_step1=None,
                initial_learning_rate=hyperparameters.solver2_lr,
                measure_loss_weight=hyperparameters.solver_mea_weight,
                lpip_loss_weight=hyperparameters.solver_lpip_weight,
                cut_point1=hyperparameters.solver2_cutpoint1,
                cut_point2=hyperparameters.solver2_cutpoint2,
                w_avg_samples=10000,
                lr_rampdown_length=0.75,
                lr_rampup_length=0.02,
                regularize_noise_weight=1e5,
                initial_noise_factor=0.005,
                noise_ramp_length=0.75,
                device=global_config.device,
                ):

        if initial_w is None:
            cut_point1 = 0
            cut_point2 = 0
        G = copy.deepcopy(self.G).eval().requires_grad_(False).to(device).float()
        # Compute w stats.
        z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
        w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
        w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
        w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
        w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5
        start_w = initial_w if initial_w is not None else w_avg
        start_w = torch.tensor(start_w, dtype=torch.float32, device=device, requires_grad=False)
        if start_w.shape[1] == 1:
            start_w = torch.tile(start_w, (1, G.mapping.num_ws, 1))

        w_var = torch.tensor(start_w[:, cut_point1:(G.mapping.num_ws-cut_point2), :].detach().cpu().numpy(),
                             dtype=torch.float32, device=device, requires_grad=True)
        w_prior1 = torch.tensor(start_w[:, :cut_point1, :].detach().cpu().numpy(),
                                dtype=torch.float32, device=device, requires_grad=False)
        w_prior2 = torch.tensor(start_w[:, (G.mapping.num_ws-cut_point2):, :].detach().cpu().numpy(),
                                dtype=torch.float32, device=device, requires_grad=False)

        noise_bufs = {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}
        for buf in noise_bufs.values():
            buf.data[:] = torch.randn_like(buf)
            buf.requires_grad = True
        if noise_bufs_step1 is not None:
            for name, buf in G.synthesis.named_buffers():
                if name in noise_bufs_step1:
                    buf.data.copy_(noise_bufs_step1[name].data)

        optimizer = torch.optim.Adam([w_var] + list(noise_bufs.values()), betas=(0.9, 0.999),
                                     lr=initial_learning_rate)
        lpips_mea = MultiscaleLPIPS()

        for step in tqdm(range(num_steps)):
            t = step / num_steps
            w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
            lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
            lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
            lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
            lr = initial_learning_rate * (lr_ramp + 1)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            ws = torch.cat((w_prior1, w_var, w_prior2), dim=1)
            ws = ws + torch.randn_like(ws) * w_noise_scale
            synth_images = G.synthesis(ws, noise_mode='const', force_fp32=True)
            filename = f'E:\\PTI-main-old\\result\\phase2\\output_{step}.mat'
            savemat(filename,
                    {'output_cs': np.transpose(np.squeeze(synth_images.detach().cpu().numpy()), [1, 2, 0])})
            if step % 20 == 0:
                image_save_2(synth_images.detach().cpu().numpy(), "E:\\PTI-main-old\\result\\proj_image_cs.jpg")
                savemat('E:\\PTI-main-old\\result\\output_cs.mat',
                        {'output_cs': np.transpose(np.squeeze(synth_images.detach().cpu().numpy()), [1, 2, 0])})
            if hyperparameters.gray_to_rgb:
                synth_images = 0.2989 * synth_images[:, 0, :, :] + 0.5870 * synth_images[:, 1, :, :] + 0.1140 * synth_images[:, 2, :, :]
                synth_images = torch.cat([synth_images] * 3, dim=0).unsqueeze(0)

            downsampled_image = F.interpolate(synth_images, size=(self.target.shape[2], self.target.shape[2]),
                                              mode='bilinear', align_corners=False)
            mea_synth = self.sensor(downsampled_image)
            image_ATy_synth = self.sensor.trans_torch_3(mea_synth)
            image_ATy_synth = normalize_per_channel(image_ATy_synth)

            if step % 20 == 0:
                image_save_1(image_ATy_synth.detach().cpu().numpy(), "E:\\PTI-main-old\\result\\proj_imageATy_cs.jpg")

            # Noise regularization.
            reg_loss = 0.0
            for v in noise_bufs.values():
                noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
                while True:
                    reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                    reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2)

            loss_measure = F.mse_loss(mea_synth, self.mea_true, reduction='mean')
            loss_lpips = lpips_mea(image_ATy_synth, self.image_ATy)
            loss = measure_loss_weight * loss_measure + lpip_loss_weight * loss_lpips + reg_loss * regularize_noise_weight
            log_loss_to_file(loss.detach().cpu().numpy(), filename="E:\\PTI-main-old\\result\\loss_total_phase2.txt")
            log_loss_to_file(loss_measure.detach().cpu().numpy(), filename="E:\\PTI-main-old\\result\\loss_measure_phase2.txt")
            log_loss_to_file(loss_lpips.detach().cpu().numpy(), filename="E:\\PTI-main-old\\result\\loss_lpip_phase2.txt")
            if step % 20 == 0:
                target_dist = F.mse_loss(self.target, torch.squeeze(downsampled_image))
                print('loss_total:', loss.detach().cpu().numpy(),
                      'loss_mea:', measure_loss_weight * loss_measure.detach().cpu().numpy(),
                      'loss_lpips', lpip_loss_weight * loss_lpips.detach().cpu().numpy(),
                      'loss_reg', regularize_noise_weight * reg_loss.detach().cpu().numpy())
                print('target_dist:', target_dist.detach().cpu().numpy())
                print('learning_rate:', lr)

            # Step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Normalize noise.
            with torch.no_grad():
                for buf in noise_bufs.values():
                    buf -= buf.mean()
                    buf *= buf.square().mean().rsqrt()
        del G
        return torch.cat((w_prior1, w_var, w_prior2), dim=1), noise_bufs

    def solver3(self,
                num_steps=hyperparameters.solver3_step,
                initial_w=None,
                noise_bufs_step1=None,
                lr=hyperparameters.solver3_lr,
                weight_measure=hyperparameters.solver_mea_weight,
                weight_lpips=hyperparameters.solver_lpip_weight,
                w_avg_samples=10000,
                device=global_config.device,
                ):

        use_ball_holder = True
        G = copy.deepcopy(self.G).requires_grad_(True).to(device).float()

        z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
        w_samples = self.G.mapping(torch.from_numpy(z_samples).to(device), None)
        w_samples = w_samples[:, :1, :].detach().cpu().numpy().astype(np.float32)
        w_avg = np.mean(w_samples, axis=0, keepdims=True)
        start_w = initial_w if initial_w is not None else w_avg
        start_w = torch.tensor(start_w, dtype=torch.float32, device=device, requires_grad=False)
        if start_w.shape[1] == 1:
            start_w = torch.tile(start_w, (1, G.mapping.num_ws, 1))

        noise_bufs = {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}
        for buf in noise_bufs.values():
            buf.data[:] = torch.randn_like(buf)
            buf.requires_grad = True
        if noise_bufs_step1 is not None:
            for name, buf in G.synthesis.named_buffers():
                if name in noise_bufs_step1:
                    buf.data.copy_(noise_bufs_step1[name].data)

        optimizer = torch.optim.Adam(G.parameters(), lr=lr)
        lpips_mea = MultiscaleLPIPS()

        for step in tqdm(range(num_steps)):
            generated_images = G.synthesis(start_w, noise_mode='const', force_fp32=True)
            filename = f'E:\\PTI-main-old\\result\\phase3\\output_{step}.mat'
            savemat(filename,
                    {'output_cs': np.transpose(np.squeeze(generated_images.detach().cpu().numpy()), [1, 2, 0])})
            if step % 20 == 0:
                image_save_2(generated_images.detach().cpu().numpy(), "E:\\PTI-main-old\\result\\proj_image_cs.jpg")
                savemat('E:\\PTI-main-old\\result\\output_cs.mat',
                        {'output_cs': np.transpose(np.squeeze(generated_images.detach().cpu().numpy()), [1, 2, 0])})

            if hyperparameters.gray_to_rgb:
                generated_images = 0.2989 * generated_images[:, 0, :, :] + 0.5870 * generated_images[:, 1, :, :] + 0.1140 * generated_images[:, 2, :, :]
                generated_images = torch.cat([generated_images] * 3, dim=0).unsqueeze(0)

            downsampled_image = F.interpolate(generated_images, size=(self.target.shape[2], self.target.shape[2]),
                                              mode='bilinear', align_corners=False)

            mea_synth = self.sensor(downsampled_image)
            image_ATy_synth = self.sensor.trans_torch_3(mea_synth)
            image_ATy_synth = normalize_per_channel(image_ATy_synth)

            if step % 20 == 0:
                image_save_1(image_ATy_synth.detach().cpu().numpy(), "E:\\PTI-main-old\\result\\proj_imageATy_cs.jpg")

            loss_measure = F.mse_loss(mea_synth, self.mea_true, reduction='mean')
            loss_lpips = lpips_mea(image_ATy_synth, self.image_ATy)

            if use_ball_holder and hyperparameters.use_locality_regularization:
                ball_holder_loss_val = self.space_regulizer.space_regulizer_loss(G, start_w, use_wandb=False)
            else:
                ball_holder_loss_val = 0

            loss = loss_measure * weight_measure + loss_lpips * weight_lpips + ball_holder_loss_val

            log_loss_to_file(loss.detach().cpu().numpy(), filename="E:\\PTI-main-old\\result\\loss_total_phase3.txt")
            log_loss_to_file(loss_measure.detach().cpu().numpy(), filename="E:\\PTI-main-old\\result\\loss_measure_phase3.txt")
            log_loss_to_file(loss_lpips.detach().cpu().numpy(), filename="E:\\PTI-main-old\\result\\loss_lpip_phase3.txt")

            if step % 20 == 0:
                target_dist = F.mse_loss(self.target, torch.squeeze(downsampled_image))
                print('loss_total:', loss.detach().cpu().numpy(),
                      'loss_mea:', weight_measure * loss_measure.detach().cpu().numpy(),
                      'loss_lpips', weight_lpips * loss_lpips.detach().cpu().numpy())
                print('target_dist:', target_dist.detach().cpu().numpy())
                if not hyperparameters.use_prior:
                    torch.save(G, f'{paths_config.checkpoints_dir}/model_{global_config.run_name}_prior.pt')
                    torch.save(start_w, f'E:\\PTI-main-old\\w_solve\\barcelona\\PTI\\w_prior.pt')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0
            global_config.training_step += 1


    def solver4(self, initial_w=None, noise_bufs_step1=None):
        wsolver1, noise1 = self.solver1(initial_w=initial_w, noise_bufs_step1=noise_bufs_step1)
        wsolver2, noise2 = self.solver2(initial_w=wsolver1, noise_bufs_step1=noise1)
        self.solver3(initial_w=wsolver2, noise_bufs_step1=noise2)

    def solver_prior(self, initial_w=None, noise_bufs_step1=None):
        wsolver2, noise2 = self.solver2(initial_w=initial_w, noise_bufs_step1=noise_bufs_step1)
        self.solver3(initial_w=wsolver2, noise_bufs_step1=noise2)




