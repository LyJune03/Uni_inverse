import os
import copy
from random import choice
from string import ascii_uppercase
from types import SimpleNamespace
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
hyperparameters = SimpleNamespace(
    first_inv_type="w",
    optim_type="adam",

    latent_ball_num_of_samples=1,
    locality_regularization_interval=1,
    use_locality_regularization=False,
    regulizer_l2_lambda=1,
    regulizer_alpha=30,

    poi_loss_weight=1.0,

    first_inv_steps=200,
    second_inv_steps=40,
    max_pti_steps=200,
    max_images_to_invert=30,
    second_inv_model=True,

    pti_learning_rate=3e-4,
    first_inv_lr=3e-2,
    second_inv_lr=5e-3,
    train_batch_size=1,

    n_learning_rate=2e-1,

    gray_to_rgb=False,
    super_resolution=True,

    add_noise=True,

    poisson_photon_true=5,

    N_init_quantile=0.99,
    N_init_min=1e-3,
    N_init_max_samples=2_000_000,

    use_prior=False,
    use_last_w_pivots=False,
    prior_run_id="project",
    prior_name="man1",
    prior_wplus_steps=40,
)

global_config = SimpleNamespace(
    cuda_visible_devices="0",
    device="cuda:0",
    training_step=1,
    pivotal_training_steps=0,
    run_name="",
    run_solve_name="",
)

paths_config = SimpleNamespace(
    checkpoints_dir=r"C:\Users\vipl\Desktop\PTI-main-old\checkpoints",
    embedding_base_dir=r"C:\Users\vipl\Desktop\PTI-main-old\embeddings",
    input_data_path=r"C:\Users\vipl\Desktop\PTI-main-old\image",
    input_data_id="barcelona",
    pti_results_keyword="PTI",
)

from utils.models_utils import toogle_grad, load_old_G, load_tuned_G
from utils.data_utils import make_dataset
def l2_loss(real_images, generated_images):
    l2_criterion = torch.nn.MSELoss(reduction='mean')
    loss = l2_criterion(real_images, generated_images)
    return loss

def inv_softplus(x: float) -> float:
    x = max(float(x), 1e-6)
    return float(np.log(np.expm1(x)))

def to_01(x_m11: torch.Tensor) -> torch.Tensor:
    return (x_m11 + 1.0) / 2.0

def rgb_to_gray01(x01: torch.Tensor) -> torch.Tensor:
    gray = 0.2989 * x01[:, 0] + 0.5870 * x01[:, 1] + 0.1140 * x01[:, 2]
    return gray[:, None].repeat(1, 3, 1, 1)

@torch.no_grad()
def make_poisson_observation_counts(clean01: torch.Tensor, N_true: float):
    clean01 = torch.clamp(clean01, 0.0, 1.0)
    lam = clean01 * float(N_true)
    y_counts = torch.poisson(lam)
    return y_counts

@torch.no_grad()
def estimate_N_init_quantile(
    y_counts: torch.Tensor,
    q: float = 0.99,
    minN: float = 1e-3,
    max_samples: int = 2_000_000
) -> float:
    flat = y_counts.detach().float().reshape(-1)
    n = flat.numel()
    if n == 0:
        return float(minN)

    if n > int(max_samples):
        idx = torch.randint(0, n, (int(max_samples),), device=flat.device)
        flat = flat[idx]

    q_t = torch.tensor(float(q), device=flat.device, dtype=flat.dtype)
    N0 = torch.quantile(flat, q_t).item()
    return max(float(N0), float(minN))

def poisson_nll_from_rate(rate: torch.Tensor, y_counts: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    rate = torch.clamp(rate, min=eps)
    return torch.mean(rate - y_counts * torch.log(rate))

def intensity_from_counts(y_counts: torch.Tensor, N: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.clamp(y_counts / torch.clamp(N, min=eps), 0.0, 1.0)

class Space_Regulizer:
    def __init__(self, original_G):
        self.original_G = original_G
        self.morphing_regulizer_alpha = hyperparameters.regulizer_alpha

    def get_morphed_w_code(self, new_w_code, fixed_w):
        interpolation_direction = new_w_code - fixed_w
        interpolation_direction_norm = torch.norm(interpolation_direction, p=2)
        direction_to_move = hyperparameters.regulizer_alpha * interpolation_direction / interpolation_direction_norm
        result_w = fixed_w + direction_to_move
        return result_w

    def ball_holder_loss_lazy(self, new_G, num_of_sampled_latents, w_batch):
        loss = 0.0
        z_samples = np.random.randn(num_of_sampled_latents, self.original_G.z_dim)
        w_samples = self.original_G.mapping(
            torch.from_numpy(z_samples).to(global_config.device),
            None,
            truncation_psi=0.5
        )
        territory_indicator_ws = [self.get_morphed_w_code(w_code.unsqueeze(0), w_batch) for w_code in w_samples]

        for w_code in territory_indicator_ws:
            new_img = new_G.synthesis(w_code, noise_mode='none', force_fp32=True)
            with torch.no_grad():
                old_img = self.original_G.synthesis(w_code, noise_mode='none', force_fp32=True)

            if hyperparameters.regulizer_l2_lambda > 0:
                loss += l2_loss.l2_loss(old_img, new_img) * hyperparameters.regulizer_l2_lambda

        return loss / len(territory_indicator_ws)

    def space_regulizer_loss(self, new_G, w_batch):
        return self.ball_holder_loss_lazy(new_G, hyperparameters.latent_ball_num_of_samples, w_batch)

def _get_N_from_logN(logN: torch.Tensor) -> torch.Tensor:
    return F.softplus(logN) + 1e-6

def project(
    G,
    target01: torch.Tensor,
    y_counts: torch.Tensor = None,
    *,
    num_steps=1000,
    w_avg_samples=30000,
    initial_learning_rate=hyperparameters.first_inv_lr,
    initial_noise_factor=0.005,
    lr_rampdown_length=0.25,
    lr_rampup_length=0.05,
    noise_ramp_length=0.75,
    regularize_noise_weight=1e5,
    verbose=False,
    device: torch.device,
    initial_w=None,
    initial_N: Optional[float] = None,
):
    assert target01.shape[1] == target01.shape[2]
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device).float()

    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)
    w_samples = w_samples[:, :1, :].detach().cpu().numpy().astype(np.float32)
    w_avg = np.mean(w_samples, axis=0, keepdims=True)
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    start_w = initial_w if initial_w is not None else w_avg
    num_ws = G.mapping.num_ws

    noise_bufs = {name: buf for (name, buf) in G.synthesis.named_buffers() if "noise_const" in name}
    target_images = target01.unsqueeze(0).to(device).to(torch.float32)

    assert hyperparameters.add_noise, "This script is Poisson-only: set add_noise=True"
    assert y_counts is not None, "add_noise=True: y_counts must be provided"

    initN = float(initial_N) if (initial_N is not None) else float(hyperparameters.N_init_min)
    logN = torch.tensor(
        inv_softplus(max(initN, hyperparameters.N_init_min)),
        device=device, dtype=torch.float32, requires_grad=True
    )

    w_opt = torch.tensor(start_w, dtype=torch.float32, device=device, requires_grad=True)

    optimizer = torch.optim.Adam(
        [
            {"params": [w_opt] + list(noise_bufs.values()), "lr": float(initial_learning_rate)},
            {"params": [logN], "lr": float(hyperparameters.n_learning_rate)},
        ],
        betas=(0.9, 0.999),
    )

    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    pbar = tqdm(range(num_steps), desc="Project-W", leave=False)
    for step in pbar:
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2

        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp *= min(1.0, t / lr_rampup_length)
        lr = float(initial_learning_rate) * lr_ramp
        optimizer.param_groups[0]["lr"] = lr

        ws = (w_opt + torch.randn_like(w_opt) * w_noise_scale).repeat([1, num_ws, 1])
        synth = G.synthesis(ws, noise_mode="const", force_fp32=True)
        synth01 = torch.clamp(to_01(synth), 0.0, 1.0)

        if hyperparameters.super_resolution:
            synth01 = F.interpolate(
                synth01, size=(target_images.shape[2], target_images.shape[3]),
                mode="bilinear", align_corners=False
            )

        if hyperparameters.gray_to_rgb:
            synth01 = rgb_to_gray01(synth01)

        N_pred = _get_N_from_logN(logN)
        rate = synth01 * N_pred
        loss_poi = poisson_nll_from_rate(rate, y_counts)

        reg = 0.0
        for v in noise_bufs.values():
            n = v[None, None, :, :]
            while True:
                reg += (n * torch.roll(n, shifts=1, dims=3)).mean() ** 2
                reg += (n * torch.roll(n, shifts=1, dims=2)).mean() ** 2
                if n.shape[2] <= 8:
                    break
                n = F.avg_pool2d(n, kernel_size=2)

        loss = reg * regularize_noise_weight + loss_poi * hyperparameters.poi_loss_weight

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

        pbar.set_postfix(
            N=f"{float(N_pred.detach().cpu().item()):.4f}",
            poi=f"{float(loss_poi.detach().cpu().item()):.4f}",
        )

    N_est = float(_get_N_from_logN(logN).detach().cpu().item())
    del G
    return w_opt.repeat([1, num_ws, 1]), noise_bufs, N_est


def project_wplus(
    G,
    target01: torch.Tensor,
    y_counts: torch.Tensor = None,
    *,
    num_steps=1000,
    w_avg_samples=30000,
    initial_learning_rate=hyperparameters.second_inv_lr,
    initial_noise_factor=0.005,
    lr_rampdown_length=0.25,
    lr_rampup_length=0.05,
    noise_ramp_length=0.75,
    regularize_noise_weight=1e5,
    verbose=False,
    device: torch.device,
    initial_w=None,
    noise_bufs_step1=None,
    initial_N: Optional[float] = None,
):
    assert target01.shape[1] == target01.shape[2]
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device).float()

    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)
    w_samples = w_samples[:, :1, :].detach().cpu().numpy().astype(np.float32)
    w_avg = np.mean(w_samples, axis=0, keepdims=True)
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    start_w = initial_w if initial_w is not None else w_avg
    num_ws = G.mapping.num_ws

    noise_bufs = {name: buf for (name, buf) in G.synthesis.named_buffers() if "noise_const" in name}
    target_images = target01.unsqueeze(0).to(device).to(torch.float32)

    assert hyperparameters.add_noise, "This script is Poisson-only: set add_noise=True"
    assert y_counts is not None, "add_noise=True: y_counts must be provided"

    w_mid = torch.tensor(start_w, dtype=torch.float32, device=device, requires_grad=False)
    if w_mid.ndim == 2:
        w_mid = w_mid[:, None, :]
    if w_mid.shape[1] != num_ws:
        w_mid = w_mid.repeat(1, num_ws, 1)
    w_opt = w_mid.detach().clone().requires_grad_(True)

    initN = float(initial_N) if (initial_N is not None) else float(hyperparameters.N_init_min)
    logN = torch.tensor(
        inv_softplus(max(initN, hyperparameters.N_init_min)),
        device=device, dtype=torch.float32, requires_grad=True
    )

    optimizer = torch.optim.Adam(
        [
            {"params": [w_opt] + list(noise_bufs.values()), "lr": float(initial_learning_rate)},
            {"params": [logN], "lr": float(hyperparameters.n_learning_rate)},
        ],
        betas=(0.9, 0.999),
    )

    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    if noise_bufs_step1 is not None:
        for name, buf in G.synthesis.named_buffers():
            if name in noise_bufs_step1:
                buf.data.copy_(noise_bufs_step1[name].data)

    pbar = tqdm(range(num_steps), desc="Project-W+", leave=False)
    for step in pbar:
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2

        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp *= min(1.0, t / lr_rampup_length)
        lr = float(initial_learning_rate) * lr_ramp
        optimizer.param_groups[0]["lr"] = lr

        ws = w_opt + torch.randn_like(w_opt) * w_noise_scale
        synth = G.synthesis(ws, noise_mode="const", force_fp32=True)
        synth01 = torch.clamp(to_01(synth), 0.0, 1.0)

        if hyperparameters.super_resolution:
            synth01 = F.interpolate(
                synth01, size=(target_images.shape[2], target_images.shape[3]),
                mode="bilinear", align_corners=False
            )

        if hyperparameters.gray_to_rgb:
            synth01 = rgb_to_gray01(synth01)

        N_pred = _get_N_from_logN(logN)
        rate = synth01 * N_pred
        loss_poi = poisson_nll_from_rate(rate, y_counts)

        reg = 0.0
        for v in noise_bufs.values():
            n = v[None, None, :, :]
            while True:
                reg += (n * torch.roll(n, shifts=1, dims=3)).mean() ** 2
                reg += (n * torch.roll(n, shifts=1, dims=2)).mean() ** 2
                if n.shape[2] <= 8:
                    break
                n = F.avg_pool2d(n, kernel_size=2)

        loss = reg * regularize_noise_weight + loss_poi * hyperparameters.poi_loss_weight

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

        pbar.set_postfix(
            N=f"{float(N_pred.detach().cpu().item()):.4f}",
            poi=f"{float(loss_poi.detach().cpu().item()):.4f}",
        )

    N_est = float(_get_N_from_logN(logN).detach().cpu().item())
    del G
    return w_opt, N_est


class ImagesDataset(Dataset):
    def __init__(self, source_root, source_transform=None):
        self.source_paths = sorted(make_dataset(source_root))
        self.source_transform = source_transform

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        fname, path = self.source_paths[index]
        img = Image.open(path).convert("RGB")
        if self.source_transform:
            img = self.source_transform(img)
        return fname, img


class SingleIDCoach:
    def __init__(self, data_loader, use_wandb: bool):
        self.use_wandb = bool(use_wandb)
        self.data_loader = data_loader

        self.w_pivots = {}
        self.image_counter = 0

        self.restart_training()
        os.makedirs(paths_config.checkpoints_dir, exist_ok=True)

    def restart_training(self):
        if hyperparameters.use_prior:
            self.G = load_tuned_G(hyperparameters.prior_run_id, hyperparameters.prior_name)
        else:
            self.G = load_old_G()

        toogle_grad(self.G, True)
        self.original_G = load_old_G()
        self.space_regulizer = Space_Regulizer(self.original_G)

        self.log_N = None
        self.optimizer = None

    def _build_optimizer(self):
        params_G = list(self.G.parameters())
        self.optimizer = torch.optim.Adam(
            params_G,
            lr=float(hyperparameters.pti_learning_rate),
            betas=(0.9, 0.999),
        )

    def init_pti_N_and_optimizer(self, N_init_for_pti: Optional[float]):
        if not hyperparameters.add_noise:
            self.log_N = None
            self._build_optimizer()
            return

        if N_init_for_pti is None:
            N_init_for_pti = float(hyperparameters.N_init_min)

        N_init_for_pti = max(float(N_init_for_pti), float(hyperparameters.N_init_min))

        self.log_N = nn.Parameter(
            torch.tensor(inv_softplus(N_init_for_pti), device=global_config.device, dtype=torch.float32),
            requires_grad=False,
        )
        self._build_optimizer()

    def get_N_pred(self) -> torch.Tensor:
        assert self.log_N is not None
        return _get_N_from_logN(self.log_N)

    def load_inversions(self, w_path_dir, image_name):
        if image_name in self.w_pivots:
            return self.w_pivots[image_name]
        w_path = f"{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}/w.pt"
        if not os.path.isfile(w_path):
            return None
        w = torch.load(w_path).to(global_config.device)
        self.w_pivots[image_name] = w
        return w

    def calc_inversions(self, target01, image_name, y_counts=None, w_plus_prior=None, N_init_obs: Optional[float] = None):
        id_image01 = torch.squeeze(target01.to(global_config.device))

        if (not hyperparameters.use_prior) or (w_plus_prior is None):
            w, noise_bufs, N_w = project(
                self.G,
                id_image01,
                y_counts=y_counts,
                device=torch.device(global_config.device),
                w_avg_samples=30000,
                num_steps=hyperparameters.first_inv_steps,
                initial_N=N_init_obs,
            )

            if hyperparameters.second_inv_model:
                w_input = w[:, 0, :].detach().cpu().numpy()
                w_plus, N_wplus = project_wplus(
                    self.G,
                    id_image01,
                    y_counts=y_counts,
                    device=torch.device(global_config.device),
                    num_steps=hyperparameters.second_inv_steps,
                    noise_bufs_step1=noise_bufs,
                    initial_w=w_input,
                    initial_N=N_w,
                )
                return w_plus, N_wplus

            return w, N_w

        w_plus, N_wplus = project_wplus(
            self.G,
            id_image01,
            y_counts=y_counts,
            device=torch.device(global_config.device),
            num_steps=hyperparameters.prior_wplus_steps,
            initial_w=w_plus_prior,
            initial_N=N_init_obs,
        )
        return w_plus, N_wplus

    def calc_loss(self, I_pred01, new_G, use_ball_holder, w_batch, y_counts=None):
        assert hyperparameters.add_noise
        loss = 0.0

        with torch.no_grad():
            N_pred = self.get_N_pred()

        rate = torch.clamp(I_pred01, 0.0, 1.0) * N_pred
        loss_poi = poisson_nll_from_rate(rate, y_counts)
        loss = loss + loss_poi * hyperparameters.poi_loss_weight

        if hyperparameters.pt_l2_lambda > 0:
            I_obs01 = intensity_from_counts(y_counts, N_pred)
            loss = loss + F.mse_loss(torch.clamp(I_pred01, 0.0, 1.0), I_obs01) * hyperparameters.pt_l2_lambda

        if use_ball_holder and hyperparameters.use_locality_regularization:
            loss = loss + self.space_regulizer.space_regulizer_loss(new_G, w_batch)

        return loss, loss_poi

    def forward(self, w):
        return self.G.synthesis(w, noise_mode="const", force_fp32=True)

    def train(self):
        w_path_dir = f"{paths_config.embedding_base_dir}/{paths_config.input_data_id}"
        os.makedirs(w_path_dir, exist_ok=True)
        os.makedirs(f"{w_path_dir}/{paths_config.pti_results_keyword}", exist_ok=True)

        use_ball_holder = True

        for fname, image in tqdm(self.data_loader, desc="Images", leave=True):
            image_name = fname[0]
            self.restart_training()

            if self.image_counter >= hyperparameters.max_images_to_invert:
                break

            image = image.to(global_config.device)

            clean01 = torch.clamp(to_01(image), 0.0, 1.0)

            if hyperparameters.gray_to_rgb:
                clean01 = rgb_to_gray01(clean01)

            if hyperparameters.add_noise:
                y_counts = make_poisson_observation_counts(clean01, N_true=hyperparameters.poisson_photon_true)

                N_init_obs = estimate_N_init_quantile(
                    y_counts,
                    q=float(hyperparameters.N_init_quantile),
                    minN=float(hyperparameters.N_init_min),
                    max_samples=int(hyperparameters.N_init_max_samples),
                )

                real01 = torch.clamp(y_counts / float(hyperparameters.poisson_photon_true), 0.0, 1.0)
            else:
                raise RuntimeError("This script is Poisson-only: set add_noise=True")

            embedding_dir = f"{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}"
            os.makedirs(embedding_dir, exist_ok=True)
            if hyperparameters.use_prior:
                w_plus_prior = self.load_inversions(w_path_dir, hyperparameters.prior_name)
                if w_plus_prior is None:
                    raise FileNotFoundError(
                        f"use_prior=True 但没找到 prior w：{w_path_dir}/{paths_config.pti_results_keyword}/{hyperparameters.prior_name}/w.pt"
                    )
                w_plus_prior = w_plus_prior.detach().cpu().numpy()

                w_pivot, N_inv_final = self.calc_inversions(
                    real01, image_name,
                    y_counts=y_counts,
                    w_plus_prior=w_plus_prior,
                    N_init_obs=N_init_obs,
                )
            else:
                w_pivot, N_inv_final = self.calc_inversions(
                    real01, image_name,
                    y_counts=y_counts,
                    N_init_obs=N_init_obs,
                )

            w_pivot = w_pivot.to(global_config.device)

            self.init_pti_N_and_optimizer(N_inv_final)

            real_images_batch = real01.to(global_config.device)
            pbar_pti = tqdm(range(hyperparameters.max_pti_steps), desc=f"PTI[{image_name}]", leave=False)

            for _ in pbar_pti:
                generated_images = self.forward(w_pivot)
                I_pred01 = torch.clamp(to_01(generated_images), 0.0, 1.0)

                if hyperparameters.super_resolution:
                    H, W = real_images_batch.shape[2], real_images_batch.shape[3]
                    I_pred01 = F.interpolate(I_pred01, size=(H, W), mode="bilinear", align_corners=False)

                if hyperparameters.gray_to_rgb:
                    I_pred01 = rgb_to_gray01(I_pred01)

                loss, loss_poi = self.calc_loss(
                    I_pred01,
                    self.G, use_ball_holder, w_pivot,
                    y_counts=y_counts
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    N_pti = float(self.get_N_pred().detach().cpu().item())
                pbar_pti.set_postfix(N=f"{N_pti:.4f}", poi=f"{float(loss_poi.detach().cpu().item()):.4f}")

                use_ball_holder = (global_config.training_step % hyperparameters.locality_regularization_interval == 0)
                global_config.training_step += 1

            with torch.no_grad():
                N_pti = float(self.get_N_pred().detach().cpu().item())
            torch.save(torch.tensor([N_pti], dtype=torch.float32), f"{embedding_dir}/N_pti.pt")

            self.image_counter += 1
            torch.save(self.G, f"{paths_config.checkpoints_dir}/model_{global_config.run_name}_{image_name}.pt")


def run(run_name: str = "", use_wandb: bool = False):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = global_config.cuda_visible_devices

    global_config.run_name = (run_name if run_name != "" else "".join(choice(ascii_uppercase) for _ in range(12)))
    global_config.pivotal_training_steps = 1
    global_config.training_step = 1

    embedding_dir_path = f"{paths_config.embedding_base_dir}/{paths_config.input_data_id}/{paths_config.pti_results_keyword}"
    os.makedirs(embedding_dir_path, exist_ok=True)

    dataset = ImagesDataset(
        paths_config.input_data_path,
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]),
    )
    dataloader = DataLoader(dataset, batch_size=hyperparameters.train_batch_size, shuffle=False)

    coach = SingleIDCoach(dataloader, use_wandb=use_wandb)
    coach.train()


if __name__ == "__main__":
    run(run_name="project", use_wandb=False)
