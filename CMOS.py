import os
import copy
from random import choice
from string import ascii_uppercase
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.io import savemat
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
from lpips import LPIPS
from utils.models_utils import toogle_grad, load_old_G, load_tuned_G
from utils.data_utils import make_dataset

hyperparameters = SimpleNamespace(
    lpips_type="vgg",
    first_inv_type="w",
    optim_type="adam",

    latent_ball_num_of_samples=1,
    locality_regularization_interval=1,
    use_locality_regularization=False,
    regulizer_l2_lambda=10,
    regulizer_lpips_lambda=10,
    regulizer_alpha=30,

    pt_l2_lambda=1,
    pt_lpips_lambda=0.2,
    muti_lpips_weight=[1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
    muti_l1=0.0,

    add_gaussian=True,
    gaussian_std=0.3,

    sigma_est_min=1e-4,
    sigma_est_max=0.4,
    sigma_lr_mult=0.2,
    gauss_nll_weight=1.0,

    first_inv_steps=200,
    second_inv_steps=40,
    max_pti_steps=200,
    max_images_to_invert=30,
    second_inv_model=True,

    pti_learning_rate=3e-4,
    first_inv_lr=3e-2,
    second_inv_lr=5e-3,
    train_batch_size=1,

    gray_to_rgb=False,
    super_resolution=True,

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
    checkpoints_dir=r"/home/vipl/vipl_hlj/PTI-main-old/checkpoints",
    embedding_base_dir=r"/home/vipl/vipl_hlj/PTI-main-old/embeddings",
    input_data_path=r"/home/vipl/vipl_hlj/PTI-main-old/image",
    input_data_id="barcelona",
    pti_results_keyword="PTI",
)

def l2_loss(real_images, generated_images):
    l2_criterion = torch.nn.MSELoss(reduction='mean')
    loss = l2_criterion(real_images, generated_images)
    return loss


def soft_clamp01(u: torch.Tensor, beta: float = 20.0) -> torch.Tensor:
    return F.softplus(u, beta=beta) - F.softplus(u - 1.0, beta=beta)

def add_true_gaussian_hardclamp(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    return (x + float(sigma) * torch.randn_like(x)).clamp(0.0, 1.0)

def add_true_gaussian_softclamp(x: torch.Tensor, sigma: torch.Tensor, beta: float = 20.0) -> torch.Tensor:
    if sigma is None:
        return x
    return soft_clamp01(x + sigma * torch.randn_like(x), beta=beta)


@torch.no_grad()
def estimate_gaussian_std_from_observation(x01: torch.Tensor) -> float:
    if x01.ndim == 3:
        x01 = x01.unsqueeze(0)
    assert x01.ndim == 4, f"expect 4D tensor, got {x01.shape}"

    if x01.shape[1] == 3:
        gray = 0.2989 * x01[:, 0] + 0.5870 * x01[:, 1] + 0.1140 * x01[:, 2]
    else:
        gray = x01[:, 0]

    v = gray.reshape(-1)
    med = v.median()
    mad = (v - med).abs().median()
    sigma = (mad / 0.6745).clamp(min=1e-8).item()
    return float(sigma)


def _safe_logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    return torch.log(p) - torch.log(1.0 - p)

def make_learnable_sigma(
    sigma_init: float,
    sigma_min: float,
    sigma_max: float,
    device: torch.device
):
    sigma_init = float(np.clip(sigma_init, sigma_min, sigma_max))
    p0 = (sigma_init - sigma_min) / (sigma_max - sigma_min + 1e-12)
    p0 = torch.tensor([p0], dtype=torch.float32, device=device)
    rho0 = _safe_logit(p0)
    rho = rho0.detach().clone().requires_grad_(True)

    def rho_to_sigma(rho_tensor: torch.Tensor) -> torch.Tensor:
        return float(sigma_min) + (float(sigma_max) - float(sigma_min)) * torch.sigmoid(rho_tensor)

    return rho, rho_to_sigma


def gaussian_nll(x: torch.Tensor, y: torch.Tensor, sigma: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    sigma2 = sigma * sigma + eps
    res2 = (x - y).pow(2)
    return 0.5 * (res2 / sigma2 + torch.log(sigma2)).mean()


class MultiscaleLPIPS:
    def __init__(
        self,
        min_loss_res: int = 16,
        level_weights=hyperparameters.muti_lpips_weight,
        l1_weight: float = hyperparameters.muti_l1
    ):
        super().__init__()
        self.min_loss_res = int(min_loss_res)
        self.weights = level_weights
        self.l1_weight = float(l1_weight)
        self.lpips_network = LPIPS(net="vgg", verbose=False).cuda().eval()

    def measure_lpips(self, x, y):
        return self.lpips_network(x, y, normalize=True).mean()

    def __call__(self, x, y):
        losses = []
        x_cur, y_cur = x, y
        for w in self.weights:
            if y_cur.shape[-1] <= self.min_loss_res:
                break
            if w > 0:
                losses.append(w * self.measure_lpips(x_cur, y_cur))
            x_cur = F.avg_pool2d(x_cur, 2)
            y_cur = F.avg_pool2d(y_cur, 2)

        total = torch.stack(losses).sum(dim=0) if len(losses) > 0 else torch.tensor(0.0, device=x.device)
        l1 = self.l1_weight * F.l1_loss(x_cur, y_cur)
        return total + l1


class Space_Regulizer:
    def __init__(self, original_G, lpips_net):
        self.original_G = original_G
        self.morphing_regulizer_alpha = hyperparameters.regulizer_alpha
        self.lpips_loss = lpips_net

    def get_morphed_w_code(self, new_w_code, fixed_w):
        interpolation_direction = new_w_code - fixed_w
        interpolation_direction_norm = torch.norm(interpolation_direction, p=2)
        direction_to_move = hyperparameters.regulizer_alpha * interpolation_direction / interpolation_direction_norm
        result_w = fixed_w + direction_to_move
        return result_w

    def get_image_from_ws(self, w_codes, G):
        return torch.cat([G.synthesis(w_code, noise_mode='none', force_fp32=True) for w_code in w_codes])

    def ball_holder_loss_lazy(self, new_G, num_of_sampled_latents, w_batch, use_wandb=False):
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
                loss += l2_loss(old_img, new_img) * hyperparameters.regulizer_l2_lambda

            if hyperparameters.regulizer_lpips_lambda > 0:
                loss_lpips = self.lpips_loss(old_img, new_img)
                loss_lpips = torch.mean(torch.squeeze(loss_lpips))
                loss += loss_lpips * hyperparameters.regulizer_lpips_lambda

        return loss / len(territory_indicator_ws)

    def space_regulizer_loss(self, new_G, w_batch, use_wandb):
        return self.ball_holder_loss_lazy(new_G, hyperparameters.latent_ball_num_of_samples, w_batch, use_wandb)


def project(
    G,
    target: torch.Tensor,
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
    sigma_init: float = None,
    learn_sigma: bool = True,
):
    assert target.shape[1] == target.shape[2]
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device).float()

    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)
    w_samples = w_samples[:, :1, :].detach().cpu().numpy().astype(np.float32)
    w_avg = np.mean(w_samples, axis=0, keepdims=True)
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    start_w = initial_w if initial_w is not None else w_avg
    num_ws = G.mapping.num_ws

    noise_bufs = {name: buf for (name, buf) in G.synthesis.named_buffers() if "noise_const" in name}
    lpips_mea = MultiscaleLPIPS()
    target_images = target.unsqueeze(0).to(device).to(torch.float32)

    if sigma_init is None:
        sigma_init = 0.01

    rho, rho_to_sigma = make_learnable_sigma(
        sigma_init=sigma_init,
        sigma_min=hyperparameters.sigma_est_min,
        sigma_max=hyperparameters.sigma_est_max,
        device=device
    )

    w_opt = torch.tensor(start_w, dtype=torch.float32, device=device, requires_grad=True)

    params_main = [w_opt] + list(noise_bufs.values())
    param_groups = [{"params": params_main, "lr": initial_learning_rate}]
    if learn_sigma:
        param_groups.append({"params": [rho], "lr": initial_learning_rate * float(hyperparameters.sigma_lr_mult)})

    optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.999))

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
        lr = initial_learning_rate * lr_ramp
        optimizer.param_groups[0]["lr"] = lr
        if learn_sigma:
            optimizer.param_groups[1]["lr"] = lr * float(hyperparameters.sigma_lr_mult)

        ws = (w_opt + torch.randn_like(w_opt) * w_noise_scale).repeat([1, num_ws, 1])
        synth = G.synthesis(ws, noise_mode="const", force_fp32=True)
        synth = (synth + 1) / 2

        if hyperparameters.super_resolution:
            synth = F.interpolate(
                synth,
                size=(target_images.shape[2], target_images.shape[3]),
                mode="bilinear",
                align_corners=False,
            )

        if hyperparameters.gray_to_rgb:
            gray = 0.2989 * synth[:, 0] + 0.5870 * synth[:, 1] + 0.1140 * synth[:, 2]
            synth = gray[:, None].repeat(1, 3, 1, 1)

        sigma = rho_to_sigma(rho)
        sigma2_val = float((sigma * sigma).detach().item())

        synth_lp = add_true_gaussian_softclamp(synth, sigma, beta=20.0) if hyperparameters.add_gaussian else synth
        dist_lpips = lpips_mea(synth_lp, target_images)

        nll = gaussian_nll(synth, target_images, sigma) * float(hyperparameters.gauss_nll_weight)

        reg = 0.0
        for v in noise_bufs.values():
            n = v[None, None, :, :]
            while True:
                reg += (n * torch.roll(n, shifts=1, dims=3)).mean() ** 2
                reg += (n * torch.roll(n, shifts=1, dims=2)).mean() ** 2
                if n.shape[2] <= 8:
                    break
                n = F.avg_pool2d(n, kernel_size=2)

        loss = dist_lpips + nll + reg * regularize_noise_weight

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

        pbar.set_postfix(
            sigma2=sigma2_val,
            lpips=float(dist_lpips.detach().item()),
            nll=float(nll.detach().item()),
            loss=float(loss.detach().item()),
        )

    sigma_final = rho_to_sigma(rho).detach()
    del G
    return w_opt.repeat([1, num_ws, 1]), noise_bufs, sigma_final


def project_wplus(
    G,
    target: torch.Tensor,
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
    sigma_init: float = None,
    learn_sigma: bool = True,
):
    assert target.shape[1] == target.shape[2]
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device).float()

    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)
    w_samples = w_samples[:, :1, :].detach().cpu().numpy().astype(np.float32)
    w_avg = np.mean(w_samples, axis=0, keepdims=True)
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    start_w = initial_w if initial_w is not None else w_avg
    num_ws = G.mapping.num_ws

    noise_bufs = {name: buf for (name, buf) in G.synthesis.named_buffers() if "noise_const" in name}
    lpips_mea = MultiscaleLPIPS()
    target_images = target.unsqueeze(0).to(device).to(torch.float32)

    if sigma_init is None:
        sigma_init = 0.01

    rho, rho_to_sigma = make_learnable_sigma(
        sigma_init=sigma_init,
        sigma_min=hyperparameters.sigma_est_min,
        sigma_max=hyperparameters.sigma_est_max,
        device=device
    )

    w_mid = torch.tensor(start_w, dtype=torch.float32, device=device, requires_grad=False)
    if w_mid.ndim == 2:
        w_mid = w_mid[:, None, :]
    if w_mid.shape[1] != num_ws:
        w_mid = w_mid.repeat(1, num_ws, 1)

    w_opt = w_mid.detach().clone().requires_grad_(True)

    params_main = [w_opt] + list(noise_bufs.values())
    param_groups = [{"params": params_main, "lr": initial_learning_rate}]
    if learn_sigma:
        param_groups.append({"params": [rho], "lr": initial_learning_rate * float(hyperparameters.sigma_lr_mult)})

    optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.999))

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
        lr = initial_learning_rate * lr_ramp
        optimizer.param_groups[0]["lr"] = lr
        if learn_sigma:
            optimizer.param_groups[1]["lr"] = lr * float(hyperparameters.sigma_lr_mult)

        ws = w_opt + torch.randn_like(w_opt) * w_noise_scale
        synth = G.synthesis(ws, noise_mode="const", force_fp32=True)
        synth = (synth + 1) / 2

        if hyperparameters.super_resolution:
            synth = F.interpolate(
                synth,
                size=(target_images.shape[2], target_images.shape[3]),
                mode="bilinear",
                align_corners=False,
            )

        if hyperparameters.gray_to_rgb:
            gray = 0.2989 * synth[:, 0] + 0.5870 * synth[:, 1] + 0.1140 * synth[:, 2]
            synth = gray[:, None].repeat(1, 3, 1, 1)

        sigma = rho_to_sigma(rho)
        sigma2_val = float((sigma * sigma).detach().item())

        synth_lp = add_true_gaussian_softclamp(synth, sigma, beta=20.0) if hyperparameters.add_gaussian else synth
        dist_lpips = lpips_mea(synth_lp, target_images)

        nll = gaussian_nll(synth, target_images, sigma) * float(hyperparameters.gauss_nll_weight)

        reg = 0.0
        for v in noise_bufs.values():
            n = v[None, None, :, :]
            while True:
                reg += (n * torch.roll(n, shifts=1, dims=3)).mean() ** 2
                reg += (n * torch.roll(n, shifts=1, dims=2)).mean() ** 2
                if n.shape[2] <= 8:
                    break
                n = F.avg_pool2d(n, kernel_size=2)

        loss = dist_lpips + nll + reg * regularize_noise_weight

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

        pbar.set_postfix(
            sigma2=sigma2_val,
            lpips=float(dist_lpips.detach().item()),
            nll=float(nll.detach().item()),
            loss=float(loss.detach().item()),
        )

    sigma_final = rho_to_sigma(rho).detach()
    del G
    return w_opt, sigma_final


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

        self.lpips_mea = MultiscaleLPIPS()
        self.lpips_loss = LPIPS(net=hyperparameters.lpips_type).to(global_config.device).eval()

        self.restart_training()
        os.makedirs(paths_config.checkpoints_dir, exist_ok=True)

        self.sigma_fixed_map = {}

    def restart_training(self):
        if hyperparameters.use_prior:
            self.G = load_tuned_G(hyperparameters.prior_run_id, hyperparameters.prior_name)
        else:
            self.G = load_old_G()

        toogle_grad(self.G, True)
        self.original_G = load_old_G()
        self.space_regulizer = Space_Regulizer(self.original_G, self.lpips_loss)
        self.optimizer = self.configure_optimizers()

    def configure_optimizers(self):
        return torch.optim.Adam(self.G.parameters(), lr=hyperparameters.pti_learning_rate)

    def load_inversions(self, w_path_dir, image_name):
        if image_name in self.w_pivots:
            return self.w_pivots[image_name]

        w_path = f"{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}/w.pt"
        if not os.path.isfile(w_path):
            return None

        w = torch.load(w_path).to(global_config.device)
        self.w_pivots[image_name] = w
        return w

    def calc_inversions(self, image01, image_name, sigma_init: float, w_plus_prior=None):
        id_image = torch.squeeze(image01.to(global_config.device))

        if (not hyperparameters.use_prior) or (w_plus_prior is None):
            w, noise_bufs, sigma_hat1 = project(
                self.G,
                id_image,
                device=torch.device(global_config.device),
                w_avg_samples=30000,
                num_steps=hyperparameters.first_inv_steps,
                sigma_init=sigma_init,
                learn_sigma=True,
            )

            sigma_hat = sigma_hat1

            if hyperparameters.second_inv_model:
                w_input = w[:, 0, :].detach().cpu().numpy()
                w_plus, sigma_hat2 = project_wplus(
                    self.G,
                    torch.squeeze(image01.to(global_config.device)),
                    device=torch.device(global_config.device),
                    num_steps=hyperparameters.second_inv_steps,
                    noise_bufs_step1=noise_bufs,
                    initial_w=w_input,
                    sigma_init=float(sigma_hat1.detach().item()),
                    learn_sigma=True,
                )
                sigma_hat = sigma_hat2
                return w_plus, sigma_hat

            return w, sigma_hat

        w_plus, sigma_hat = project_wplus(
            self.G,
            torch.squeeze(image01.to(global_config.device)),
            device=torch.device(global_config.device),
            num_steps=hyperparameters.prior_wplus_steps,
            initial_w=w_plus_prior,
            sigma_init=sigma_init,
            learn_sigma=True,
        )
        return w_plus, sigma_hat

    def calc_loss(self, generated_images01, real_images01, new_G, use_ball_holder, w_batch, sigma_fixed: torch.Tensor):
        loss = 0.0

        if hyperparameters.pt_l2_lambda > 0:
            loss = loss + l2_loss(generated_images01, real_images01) * hyperparameters.pt_l2_lambda

        if hyperparameters.pt_lpips_lambda > 0:
            if hyperparameters.add_gaussian:
                gen_lp = add_true_gaussian_softclamp(generated_images01, sigma_fixed, beta=20.0)
            else:
                gen_lp = generated_images01
            lp = self.lpips_mea(gen_lp, real_images01)
            loss = loss + torch.squeeze(lp) * hyperparameters.pt_lpips_lambda

        loss = loss + gaussian_nll(generated_images01, real_images01, sigma_fixed) * float(hyperparameters.gauss_nll_weight)

        if use_ball_holder and hyperparameters.use_locality_regularization:
            loss = loss + self.space_regulizer.space_regulizer_loss(new_G, w_batch, use_wandb=False)

        return loss

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

            if hyperparameters.gray_to_rgb:
                gray = 0.2989 * image[:, 0] + 0.5870 * image[:, 1] + 0.1140 * image[:, 2]
                image = gray[:, None].repeat(1, 3, 1, 1)

            image = (image + 1) / 2

            if hyperparameters.add_gaussian:
                image = add_true_gaussian_hardclamp(image, hyperparameters.gaussian_std)
            save_image(image.detach().cpu(), f"/home/vipl/vipl_hlj/PTI-main-old/result/observation.png")
            sigma_init = estimate_gaussian_std_from_observation(image.detach())
            print(f"[{image_name}] sigma_init(std) from observation = {sigma_init:.6f}, var = {sigma_init*sigma_init:.6f}")

            embedding_dir = f"{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}"
            os.makedirs(embedding_dir, exist_ok=True)

            if hyperparameters.use_prior:
                w_plus_prior = self.load_inversions(w_path_dir, hyperparameters.prior_name)
                if w_plus_prior is None:
                    raise FileNotFoundError(
                        f"use_prior=True 但没找到 prior w：{w_path_dir}/{paths_config.pti_results_keyword}/{hyperparameters.prior_name}/w.pt"
                    )
                w_plus_prior = w_plus_prior.detach().cpu().numpy()
                w_pivot, sigma_hat = self.calc_inversions(image, image_name, sigma_init=sigma_init, w_plus_prior=w_plus_prior)
            else:
                w_pivot, sigma_hat = self.calc_inversions(image, image_name, sigma_init=sigma_init)

            w_pivot = w_pivot.to(global_config.device)

            torch.save(w_pivot, f"{embedding_dir}/w.pt")
            torch.save(sigma_hat.detach().cpu(), f"{embedding_dir}/sigma_hat.pt")

            sigma_fixed = sigma_hat.detach().clone().to(global_config.device)
            self.sigma_fixed_map[image_name] = sigma_fixed

            real_images_batch = image.to(global_config.device)

            pbar_pti = tqdm(range(hyperparameters.max_pti_steps), desc=f"PTI[{image_name}]", leave=False)
            for _ in pbar_pti:
                generated_images = self.forward(w_pivot)

                savemat(
                    r"/copy_1/rec.mat",
                    {"generated_images": generated_images.detach().cpu().numpy()},
                )

                generated_images = (generated_images + 1) / 2

                if hyperparameters.super_resolution:
                    H, W = real_images_batch.shape[2], real_images_batch.shape[3]
                    generated_images = F.interpolate(
                        generated_images, size=(H, W),
                        mode="bilinear", align_corners=False
                    )

                if hyperparameters.gray_to_rgb:
                    gray = 0.2989 * generated_images[:, 0] + 0.5870 * generated_images[:, 1] + 0.1140 * generated_images[:, 2]
                    generated_images = gray[:, None].repeat(1, 3, 1, 1)

                loss = self.calc_loss(
                    generated_images, real_images_batch,
                    self.G, use_ball_holder, w_pivot,
                    sigma_fixed=sigma_fixed
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                use_ball_holder = (global_config.training_step % hyperparameters.locality_regularization_interval == 0)
                global_config.training_step += 1

                sigma2_val = float((sigma_fixed * sigma_fixed).detach().item())
                pbar_pti.set_postfix(
                    sigma2=sigma2_val,
                    loss=float(loss.detach().item()),
                )

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
