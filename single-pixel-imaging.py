import copy
import numpy as np
import torch
import torch.nn.functional as F
from lpips import LPIPS
from tqdm import tqdm
from PIL import Image
from hadamard_measure import CCHadamardSensor
import pickle

cs_device = 'cuda'
cs_run_name = 'exp1'
cs_training_step = 1

cs_checkpoints_dir = r'checkpoints'
cs_image_solve_path = r'face.jpeg'
cs_old_G_path = r'stylegan2-ffhq-256x256.pkl'
cs_result_mat_path = r"final_rec.mat"

cs_sample_num = 1000
cs_gray_to_rgb = False
cs_lpips_type = 'vgg'
cs_muti_lpips_weight = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
cs_solver_mea_weight = 1e-3
cs_solver_lpip_weight = 10

cs_solver1_step = 200
cs_solver1_lr = 2e-2

cs_solver2_step = 40
cs_solver2_lr = 5e-3
cs_solver2_cutpoint1 = 0
cs_solver2_cutpoint2 = 0

cs_solver3_step = 100
cs_solver3_lr = 3e-4

cs_use_locality_regularization = True
cs_locality_regularization_interval = 1
cs_use_prior = False
cs_w_prior_path = None
cs_G_prior_path = None
cs_noise_path = None

REGULIZER_ALPHA = 30
REGULIZER_L2_LAMBDA = 1
REGULIZER_LPIPS_LAMBDA = 1
LATENT_BALL_NUM_OF_SAMPLES = 1


def l2_loss(real_images, generated_images):
    l2_criterion = torch.nn.MSELoss(reduction='mean')
    loss = l2_criterion(real_images, generated_images)
    return loss


class Space_Regulizer:
    def __init__(self, original_G, lpips_net):
        self.original_G = original_G
        self.morphing_regulizer_alpha = REGULIZER_ALPHA
        self.lpips_loss = lpips_net

    def get_morphed_w_code(self, new_w_code, fixed_w):
        interpolation_direction = new_w_code - fixed_w
        interpolation_direction_norm = torch.norm(interpolation_direction, p=2)
        direction_to_move = self.morphing_regulizer_alpha * interpolation_direction / interpolation_direction_norm
        result_w = fixed_w + direction_to_move
        return result_w

    def get_image_from_ws(self, w_codes, G):
        return torch.cat([G.synthesis(w_code, noise_mode='none', force_fp32=True) for w_code in w_codes])

    def ball_holder_loss_lazy(self, new_G, num_of_sampled_latents, w_batch, use_wandb=False):
        loss = 0.0

        z_samples = np.random.randn(num_of_sampled_latents, self.original_G.z_dim)
        w_samples = self.original_G.mapping(
            torch.from_numpy(z_samples).to(cs_device),
            None,
            truncation_psi=0.5
        )

        territory_indicator_ws = [self.get_morphed_w_code(w_code.unsqueeze(0), w_batch) for w_code in w_samples]

        for w_code in territory_indicator_ws:
            new_img = new_G.synthesis(w_code, noise_mode='none', force_fp32=True)
            with torch.no_grad():
                old_img = self.original_G.synthesis(w_code, noise_mode='none', force_fp32=True)

            if REGULIZER_L2_LAMBDA > 0:
                l2_loss_val = l2_loss(old_img, new_img)
                loss += l2_loss_val * REGULIZER_L2_LAMBDA

            if REGULIZER_LPIPS_LAMBDA > 0:
                loss_lpips = self.lpips_loss(old_img, new_img)
                loss_lpips = torch.mean(torch.squeeze(loss_lpips))
                loss += loss_lpips * REGULIZER_LPIPS_LAMBDA

        return loss / len(territory_indicator_ws)

    def space_regulizer_loss(self, new_G, w_batch, use_wandb):
        return self.ball_holder_loss_lazy(new_G, LATENT_BALL_NUM_OF_SAMPLES, w_batch, use_wandb)


def image_save(image, path):
    image_save = np.squeeze(image / 2 + 0.5) * 255
    image_save = image_save.astype(np.uint8)
    image_save = np.transpose(image_save, (1, 2, 0))
    image_target = Image.fromarray(image_save)
    image_target.save(path)


def load_old_G():
    with open(cs_old_G_path, 'rb') as f:
        old_G = pickle.load(f)['G_ema'].to(cs_device).eval()
        old_G = old_G.float()
    return old_G


def normalize_per_channel(img: torch.Tensor) -> torch.Tensor:
    mins = img.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
    maxs = img.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    ranges = torch.where(
        maxs - mins < 1e-8,
        torch.tensor(1.0, device=img.device),
        maxs - mins,
    )
    return torch.clamp((img - mins) / ranges, 0.0, 1.0)


def compute_w_stats(G, device, w_avg_samples=10000, seed=123):
    z = np.random.RandomState(seed).randn(w_avg_samples, G.z_dim)
    with torch.no_grad():
        w = G.mapping(torch.from_numpy(z).to(device), None)
    w = w[:, :1, :].cpu().numpy().astype(np.float32)
    w_avg = np.mean(w, axis=0, keepdims=True)
    w_std = (np.sum((w - w_avg) ** 2) / w_avg_samples) ** 0.5
    return w_avg, w_std


def init_noise_bufs(G):
    noise_bufs = {
        name: buf
        for (name, buf) in G.synthesis.named_buffers()
        if "noise_const" in name
    }
    for buf in noise_bufs.values():
        buf.data[:] = torch.randn_like(buf)
        buf.requires_grad = True
    return noise_bufs


def maybe_copy_noise_from(noise_bufs_target, noise_bufs_source, G):
    if noise_bufs_source is None:
        return
    for name, buf in G.synthesis.named_buffers():
        if name in noise_bufs_source:
            buf.data.copy_(noise_bufs_source[name].data)


def noise_regularizer(noise_bufs):
    reg_loss = 0.0
    for v in noise_bufs.values():
        noise = v[None, None, :, :]
        while True:
            reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
            reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
            if noise.shape[2] <= 8:
                break
            noise = F.avg_pool2d(noise, kernel_size=2)
    return reg_loss


def normalize_all_noise(noise_bufs):
    with torch.no_grad():
        for buf in noise_bufs.values():
            buf -= buf.mean()
            buf *= buf.square().mean().rsqrt()


def ramped_lr(t, base_lr, lr_rampdown_length=0.75, lr_rampup_length=0.02):
    lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
    lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
    lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
    return base_lr * (lr_ramp + 1)


class MultiscaleLPIPS:
    def __init__(
        self,
        min_loss_res: int = 16,
        level_weights=cs_muti_lpips_weight,
        l1_weight: float = 0.0,
    ):
        super().__init__()
        self.min_loss_res = min_loss_res
        self.weights = level_weights
        self.l1_weight = l1_weight
        self.lpips_network = LPIPS(net=cs_lpips_type, verbose=False).cuda()

    def measure_lpips(self, x, y):
        return self.lpips_network(x, y, normalize=True).mean()

    def __call__(self, x, y):
        losses = []
        x_i, y_i = x, y
        for weight in self.weights:
            if y_i.shape[-1] <= self.min_loss_res:
                break
            if weight > 0:
                loss = self.measure_lpips(x_i, y_i)
                losses.append(weight * loss)
            x_i = F.avg_pool2d(x_i, 2)
            y_i = F.avg_pool2d(y_i, 2)

        total = torch.stack(losses).sum(dim=0) if len(losses) > 0 else 0.0
        l1 = self.l1_weight * F.l1_loss(x_i, y_i)
        return total + l1


class CS_slover(object):
    def __init__(self, sample_num, image, mission="project"):
        assert image.shape[0] == 3

        if cs_gray_to_rgb:
            gray = (
                0.2989 * image[0, :, :] +
                0.5870 * image[1, :, :] +
                0.1140 * image[2, :, :]
            )
            image = np.concatenate([np.expand_dims(gray, axis=0)] * 3, axis=0)

        self.target = torch.tensor(image, dtype=torch.float32, device=cs_device)

        self.sensor = CCHadamardSensor(sample_num, (image.shape[1], image.shape[2]))
        mea_np = self.sensor._np(image)
        self.mea_true = torch.tensor(mea_np, dtype=torch.float32, device=cs_device)

        image_ATy = self.sensor.trans_np_3(mea_np)
        self.image_ATy = torch.tensor(image_ATy, device=cs_device, dtype=torch.float32)
        self.image_ATy = normalize_per_channel(self.image_ATy)

        if cs_use_prior:
            self.G = torch.load(cs_G_prior_path).to(cs_device)
        else:
            self.G = load_old_G()
        lpips_loss = LPIPS(net=cs_lpips_type).to(cs_device).eval()
        self.space_regulizer = Space_Regulizer(self.G, lpips_loss)

    def solver1(
        self,
        num_steps=cs_solver1_step,
        initial_w=None,
        noise_bufs_step1=None,
        initial_learning_rate=cs_solver1_lr,
        measure_loss_weight=cs_solver_mea_weight,
        lpip_loss_weight=cs_solver_lpip_weight,
        regularize_noise_weight=1e5,
        w_avg_samples=10000,
        initial_noise_factor=0.005,
        lr_rampdown_length=0.75,
        lr_rampup_length=0.02,
        noise_ramp_length=0.75,
        device=cs_device,
    ):
        G = copy.deepcopy(self.G).eval().requires_grad_(False).to(device).float()

        w_avg, w_std = compute_w_stats(G, device, w_avg_samples)
        start_w = initial_w if initial_w is not None else w_avg
        w_opt = torch.tensor(start_w, dtype=torch.float32, device=device, requires_grad=True)

        num_ws = G.mapping.num_ws

        noise_bufs = init_noise_bufs(G)
        maybe_copy_noise_from(noise_bufs, noise_bufs_step1, G)

        optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()),
                                     betas=(0.9, 0.999), lr=initial_learning_rate)
        lpips_mea = MultiscaleLPIPS()

        for step in tqdm(range(num_steps)):
            t = step / num_steps
            w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
            lr = ramped_lr(t, initial_learning_rate, lr_rampdown_length, lr_rampup_length)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            ws_base = w_opt.repeat([1, num_ws, 1])
            ws = ws_base + torch.randn_like(ws_base) * w_noise_scale
            synth_images = G.synthesis(ws, noise_mode="const", force_fp32=True)

            if cs_gray_to_rgb:
                si = (
                    0.2989 * synth_images[:, 0, :, :] +
                    0.5870 * synth_images[:, 1, :, :] +
                    0.1140 * synth_images[:, 2, :, :]
                )
                synth_images = torch.cat([si] * 3, dim=0).unsqueeze(0)

            downsampled_image = F.interpolate(
                synth_images,
                size=(self.target.shape[2], self.target.shape[2]),
                mode="bilinear",
                align_corners=False,
            )

            mea_synth = self.sensor(downsampled_image)
            image_ATy_synth = normalize_per_channel(self.sensor.trans_torch_3(mea_synth))

            reg_loss = noise_regularizer(noise_bufs)
            loss_measure = F.mse_loss(mea_synth, self.mea_true, reduction="mean")
            loss_lpip = lpips_mea(image_ATy_synth, self.image_ATy)

            loss = (measure_loss_weight * loss_measure
                    + lpip_loss_weight * loss_lpip
                    + regularize_noise_weight * reg_loss)

            if step % 20 == 0:
                print(
                    "Phase1 -> loss_total:", float(loss.detach().cpu()),
                    "loss_mea:", float(loss_measure.detach().cpu()),
                    "loss_lpips:", float(loss_lpip.detach().cpu()),
                    "lr:", lr,
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            normalize_all_noise(noise_bufs)

        del G
        return w_opt.repeat([1, num_ws, 1]), noise_bufs

    def solver2(
        self,
        num_steps=cs_solver2_step,
        initial_w=None,
        noise_bufs_step1=None,
        initial_learning_rate=cs_solver2_lr,
        measure_loss_weight=cs_solver_mea_weight,
        lpip_loss_weight=cs_solver_lpip_weight,
        cut_point1=cs_solver2_cutpoint1,
        cut_point2=cs_solver2_cutpoint2,
        w_avg_samples=10000,
        lr_rampdown_length=0.75,
        lr_rampup_length=0.02,
        regularize_noise_weight=1e5,
        initial_noise_factor=0.005,
        noise_ramp_length=0.75,
        device=cs_device,
    ):
        if initial_w is None:
            cut_point1 = 0
            cut_point2 = 0

        G = copy.deepcopy(self.G).eval().requires_grad_(False).to(device).float()

        w_avg, w_std = compute_w_stats(G, device, w_avg_samples)
        start_w_np = initial_w if initial_w is not None else w_avg
        start_w = torch.tensor(start_w_np, dtype=torch.float32, device=device, requires_grad=False)
        if start_w.shape[1] == 1:
            start_w = torch.tile(start_w, (1, G.mapping.num_ws, 1))

        w_var = torch.tensor(
            start_w[:, cut_point1:(G.mapping.num_ws - cut_point2), :].detach().cpu().numpy(),
            dtype=torch.float32, device=device, requires_grad=True
        )
        w_prior1 = torch.tensor(start_w[:, :cut_point1, :].detach().cpu().numpy(),
                                dtype=torch.float32, device=device, requires_grad=False)
        w_prior2 = torch.tensor(start_w[:, (G.mapping.num_ws - cut_point2):, :].detach().cpu().numpy(),
                                dtype=torch.float32, device=device, requires_grad=False)

        noise_bufs = init_noise_bufs(G)
        maybe_copy_noise_from(noise_bufs, noise_bufs_step1, G)

        optimizer = torch.optim.Adam([w_var] + list(noise_bufs.values()),
                                     betas=(0.9, 0.999), lr=initial_learning_rate)
        lpips_mea = MultiscaleLPIPS()

        for step in tqdm(range(num_steps)):
            t = step / num_steps
            w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
            lr = ramped_lr(t, initial_learning_rate, lr_rampdown_length, lr_rampup_length)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            ws = torch.cat((w_prior1, w_var, w_prior2), dim=1)
            ws = ws + torch.randn_like(ws) * w_noise_scale

            synth_images = G.synthesis(ws, noise_mode="const", force_fp32=True)

            if cs_gray_to_rgb:
                si = (
                    0.2989 * synth_images[:, 0, :, :] +
                    0.5870 * synth_images[:, 1, :, :] +
                    0.1140 * synth_images[:, 2, :, :]
                )
                synth_images = torch.cat([si] * 3, dim=0).unsqueeze(0)

            downsampled_image = F.interpolate(
                synth_images,
                size=(self.target.shape[2], self.target.shape[2]),
                mode="bilinear",
                align_corners=False,
            )

            mea_synth = self.sensor(downsampled_image)
            image_ATy_synth = normalize_per_channel(self.sensor.trans_torch_3(mea_synth))

            reg_loss = noise_regularizer(noise_bufs)
            loss_measure = F.mse_loss(mea_synth, self.mea_true, reduction="mean")
            loss_lpips = lpips_mea(image_ATy_synth, self.image_ATy)

            loss = (measure_loss_weight * loss_measure
                    + lpip_loss_weight * loss_lpips
                    + regularize_noise_weight * reg_loss)

            if step % 20 == 0:
                target_dist = F.mse_loss(self.target, torch.squeeze(downsampled_image))
                print(
                    "Phase2 -> loss_total:", float(loss.detach().cpu()),
                    "loss_mea:", float(measure_loss_weight * loss_measure.detach().cpu()),
                    "loss_lpips:", float(lpip_loss_weight * loss_lpips.detach().cpu()),
                    "loss_reg:", float(regularize_noise_weight * reg_loss.detach().cpu()),
                    "target_dist:", float(target_dist.detach().cpu()),
                    "lr:", lr,
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            normalize_all_noise(noise_bufs)

        del G
        return torch.cat((w_prior1, w_var, w_prior2), dim=1), noise_bufs

    def solver3(
        self,
        num_steps=cs_solver3_step,
        initial_w=None,
        noise_bufs_step1=None,
        lr=cs_solver3_lr,
        weight_measure=cs_solver_mea_weight,
        weight_lpips=cs_solver_lpip_weight,
        w_avg_samples=10000,
        device=cs_device,
    ):
        global cs_training_step

        use_ball_holder = True
        G = copy.deepcopy(self.G).requires_grad_(True).to(device).float()

        z = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
        with torch.no_grad():
            w_samples = self.G.mapping(torch.from_numpy(z).to(device), None)
        w_samples = w_samples[:, :1, :].detach().cpu().numpy().astype(np.float32)
        w_avg = np.mean(w_samples, axis=0, keepdims=True)

        start_w_np = initial_w if initial_w is not None else w_avg
        start_w = torch.tensor(start_w_np, dtype=torch.float32, device=device, requires_grad=False)
        if start_w.shape[1] == 1:
            start_w = torch.tile(start_w, (1, G.mapping.num_ws, 1))

        noise_bufs = init_noise_bufs(G)
        maybe_copy_noise_from(noise_bufs, noise_bufs_step1, G)

        optimizer = torch.optim.Adam(G.parameters(), lr=lr)
        lpips_mea = MultiscaleLPIPS()

        for step in tqdm(range(num_steps)):
            generated_images = G.synthesis(start_w, noise_mode="const", force_fp32=True)
            if cs_gray_to_rgb:
                gi = (
                    0.2989 * generated_images[:, 0, :, :] +
                    0.5870 * generated_images[:, 1, :, :] +
                    0.1140 * generated_images[:, 2, :, :]
                )
                generated_images = torch.cat([gi] * 3, dim=0).unsqueeze(0)

            downsampled_image = F.interpolate(
                generated_images,
                size=(self.target.shape[2], self.target.shape[2]),
                mode="bilinear",
                align_corners=False,
            )

            mea_synth = self.sensor(downsampled_image)
            image_ATy_synth = normalize_per_channel(self.sensor.trans_torch_3(mea_synth))

            loss_measure = F.mse_loss(mea_synth, self.mea_true, reduction="mean")
            loss_lpips = lpips_mea(image_ATy_synth, self.image_ATy)

            if use_ball_holder and cs_use_locality_regularization:
                ball_holder_loss_val = self.space_regulizer.space_regulizer_loss(
                    G, start_w, use_wandb=False
                )
            else:
                ball_holder_loss_val = 0

            loss = (weight_measure * loss_measure
                    + weight_lpips * loss_lpips
                    + ball_holder_loss_val)

            if step % 20 == 0:
                target_dist = F.mse_loss(self.target, torch.squeeze(downsampled_image))
                print(
                    "Phase3 -> loss_total:", float(loss.detach().cpu()),
                    "loss_mea:", float(weight_measure * loss_measure.detach().cpu()),
                    "loss_lpips:", float(weight_lpips * loss_lpips.detach().cpu()),
                    "target_dist:", float(target_dist.detach().cpu()),
                )
                if not cs_use_prior:
                    torch.save(G, f"{cs_checkpoints_dir}/model_prior_{cs_run_name}.pt")
                    torch.save(start_w, f"{cs_checkpoints_dir}/w_prior_{cs_run_name}.pt")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            use_ball_holder = (cs_training_step % cs_locality_regularization_interval == 0)
            cs_training_step += 1

    def solver4(self, initial_w=None, noise_bufs_step1=None):
        w1, n1 = self.solver1(initial_w=initial_w, noise_bufs_step1=noise_bufs_step1)
        w2, n2 = self.solver2(initial_w=w1, noise_bufs_step1=n1)
        self.solver3(initial_w=w2, noise_bufs_step1=n2)

    def solver_prior(self):
        w_prior = torch.load(cs_w_prior_path)
        noise_bufs = torch.load(cs_noise_path)
        w2, n2 = self.solver2(initial_w=w_prior, noise_bufs_step1=noise_bufs)
        self.solver3(initial_w=w2, noise_bufs_step1=n2)


def pre_process_image(img_path):
    image = np.array(Image.open(img_path))
    image = np.transpose(image, (2, 0, 1))
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    normalized_image = (image - mean[:, np.newaxis, np.newaxis]) / std[:, np.newaxis, np.newaxis]
    return normalized_image


def run_SOLVE():
    image = pre_process_image(cs_image_solve_path)
    model = CS_slover(cs_sample_num, image)
    if cs_use_prior:
        model.solver_prior()
    else:
        model.solver4()


if __name__ == '__main__':
    run_SOLVE()
