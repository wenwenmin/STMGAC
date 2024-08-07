

def main_repair(adata,df,device, save_name):
    import pandas as pd
    import numpy as np
    import math
    from inspect import isfunction
    from functools import partial

    from einops import rearrange, reduce
    from einops.layers.torch import Rearrange

    import torch
    from torch import nn, einsum
    import torch.nn.functional as F
    torch.set_num_threads(24)

    def exisit(x):
        return x is not None


    def default(val, d):
        if exisit(val):
            return val
        return d() if isfunction(d) else d


    class Residual(nn.Module):
        def __init__(self, fn):
            super(Residual, self).__init__()
            self.fn = fn

        def forward(self, x, *args, **kwargs):
            return self.fn(x, *args, **kwargs) + x


    def Upsample(dim, dim_out=None):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(dim, default(dim_out, dim), 3, 1, 1)
        )


    def Downsample(dim, dim_out=None):
        return nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
            nn.Conv2d(dim * 4, default(dim_out, dim), 1, 1, 0)
        )


    class SinusoidalPositionEmbeddings(nn.Module):
        def __init__(self, dim):
            super(SinusoidalPositionEmbeddings, self).__init__()
            self.dim = dim

        def forward(self, time):
            device = time.device
            half_dim = self.dim // 2
            embeddings = math.log(10000) / (half_dim - 1)
            embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
            embeddings = time[:, None] * embeddings[None, :]
            embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=1)
            return embeddings


    class WeightStandardizedConv2d(nn.Conv2d):
        def forward(self, x):
            eps = 1e-5 if x.dtype == torch.float32 else 1e-3

            weight = self.weight
            mean = reduce(weight, "o ... -> o 1 1 1", "mean")
            var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
            normalized_weight = (weight - mean) * (var + eps).rsqrt()

            return F.conv2d(
                x,
                normalized_weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups
            )


    class Block(nn.Module):
        def __init__(self, dim, dim_out, groups=8):
            super(Block, self).__init__()
            self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
            self.norm = nn.GroupNorm(groups, dim_out)
            self.act = nn.SiLU()

        def forward(self, x, scale_shift=None):
            x = self.proj(x)
            x = self.norm(x)

            if exisit(scale_shift):
                scale, shift = scale_shift
                x = x * (scale + 1) + shift

            x = self.act(x)

            return x


    class ResnetBlock(nn.Module):
        def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
            super(ResnetBlock, self).__init__()
            self.mlp = (
                nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if exisit(time_emb_dim) else None
            )
            self.block1 = Block(dim, dim_out, groups=groups)
            self.block2 = Block(dim_out, dim_out, groups=groups)
            self.res_conv = nn.Conv2d(dim, dim_out, 1, 1, 0) if dim != dim_out else nn.Identity()

        def forward(self, x, time_emb=None):
            scale_shift = None
            if exisit(self.mlp) and exisit(time_emb):
                time_emb = self.mlp(time_emb)
                time_emb = rearrange(time_emb, "b c -> b c 1 1")
                scale_shift = time_emb.chunk(2, dim=1)

            h = self.block1(x, scale_shift=scale_shift)
            h = self.block2(h)
            return h + self.res_conv(x)


    class Attention(nn.Module):
        def __init__(self, dim, heads=4, dim_head=32):
            super(Attention, self).__init__()
            self.scale = dim_head ** -0.5
            self.heads = heads
            hidden_dim = dim_head * heads
            self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, 1, 0, bias=False)
            self.to_out = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

        def forward(self, x):
            b, c, h, w = x.shape
            qkv = self.to_qkv(x).chunk(3, dim=1)
            q, k, v = map(
                lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
            )
            q = q * self.scale

            sim = einsum("b h d i, b h d j -> b h i j", q, k)
            sim = sim - sim.amax(dim=-1, keepdim=True).detach()
            attn = sim.softmax(dim=-1)

            out = einsum("b h i j, b h d j -> b h i d", attn, v)
            out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
            return self.to_out(out)


    class LinearAttention(nn.Module):
        def __init__(self, dim, heads=4, dim_head=32):
            super(LinearAttention, self).__init__()
            self.scale = dim_head ** -0.5
            self.heads = heads
            hidden_dim = dim_head * heads
            self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, 1, 0, bias=False)
            self.to_out = nn.Sequential(
                nn.Conv2d(hidden_dim, dim, 1, 1, 0),
                nn.GroupNorm(1, dim)
            )

        def forward(self, x):
            b, c, h, w = x.shape
            qkv = self.to_qkv(x).chunk(3, dim=1)
            q, k, v = map(
                lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
            )

            q = q.softmax(dim=-2)
            k = k.softmax(dim=-1)

            q = q * self.scale
            context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

            out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
            out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
            return self.to_out(out)


    class PreNorm(nn.Module):
        def __init__(self, dim, fn):
            super(PreNorm, self).__init__()
            self.fn = fn
            self.norm = nn.GroupNorm(1, dim)

        def forward(self, x):
            x = self.norm(x)
            return self.fn(x)


    class Unet(nn.Module):
        def __init__(self, dim, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), channels=3, self_condition=False,
                    resnet_block_groups=4):
            super(Unet, self).__init__()

            self.channels = channels
            self.self_condition = self_condition
            input_channels = channels * (2 if self_condition else 1)

            init_dim = default(init_dim, dim)
            self.init_conv = nn.Conv2d(input_channels, init_dim, 1, 1, 0)

            dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
            in_out = list(zip(dims[:-1], dims[1:]))

            block_klass = partial(ResnetBlock, groups=resnet_block_groups)

            time_dim = dim * 4

            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim)
            )

            self.downs = nn.ModuleList([])
            self.ups = nn.ModuleList([])
            num_resolutions = len(in_out)
            for ind, (dim_in, dim_out) in enumerate(in_out):
                is_last = ind >= (num_resolutions - 1)

                self.downs.append(
                    nn.ModuleList([
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, 1, 1)
                    ])
                )

            mid_dim = dims[-1]
            self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
            self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
            self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

            for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
                is_last = ind == (len(in_out) - 1)
                self.ups.append(
                    nn.ModuleList([
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, 1, 1)
                    ])
                )

            self.out_dim = default(out_dim, channels)

            self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
            self.final_conv = nn.Conv2d(dim, self.out_dim, 1, 1, 0)

        def forward(self, x, time, x_self_cond=None):
            if self.self_condition:
                x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
                x = torch.cat((x_self_cond, x), dim=1)

            x = self.init_conv(x)
            r = x.clone()

            t = self.time_mlp(time)

            h = []

            for block1, block2, attn, downsample in self.downs:
                x = block1(x, t)
                h.append(x)
                x = block2(x, t)
                x = attn(x)
                h.append(x)
                x = downsample(x)

            x = self.mid_block1(x, t)
            x = self.mid_attn(x)
            x = self.mid_block2(x, t)

            for block1, block2, attn, upsample in self.ups:
                x = torch.cat((x, h.pop()), dim=1)
                x = block1(x, t)
                x = torch.cat((x, h.pop()), dim=1)
                x = block2(x, t)
                x = attn(x)
                x = upsample(x)

            x = torch.cat((x, r), dim=1)
            x = self.final_res_block(x, t)
            return self.final_conv(x)




    import os
    import torch
    from torch.utils.data import DataLoader
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    from tqdm.auto import tqdm
    import numpy as np
    from torchvision import transforms, datasets
    # DDPM模型

    # 定义4种生成β的方法，均需传入总步长T，返回β序列
    def cosine_beta_schedule(timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)


    def linear_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)


    def quadratic_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


    def sigmoid_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


    # 从序列a中取t时刻的值a[t](batch_size个)，维度与x_shape相同，第一维为batch_size
    def extract(a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


    # 扩散过程采样，即通过x0和t计算xt
    def q_sample(x_start, t, noise=None):
        std_dev = 0.0001
        if noise is None:
            noise = torch.randn_like(x_start) * std_dev
        sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumpord_t = extract(sqrt_one_minus_alphas_cumpord, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumpord_t * noise


    # 损失函数loss，共3种计算方式，原文使用l2
    def p_losses(denoise_model, x_start, t, noise=None, loss_type="l2"):
        
        std_dev = 0.0001
        if noise is None:
            noise = torch.randn_like(x_start) * std_dev

        x_noisy = q_sample(x_start, t, noise)
        predicted_noise = denoise_model(x_noisy, t)

        if loss_type == "l1":
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == "l2":
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss


    # 逆扩散过程采样，即通过xt和t计算xt-1，此过程需要通过网络
    @torch.no_grad()
    def p_sample(model, x, t, t_index):
        betas_t = extract(betas, t, x.shape)
        sqrt_one_minus_alphas_cumpord_t = extract(sqrt_one_minus_alphas_cumpord, t, x.shape)
        sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

        model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumpord_t)
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(posterior_variance, t, x.shape)
            std_dev = 0.0001  # Define the standard deviation here
            noise = torch.randn_like(x) * std_dev  # Fix the typo here
            return model_mean + torch.sqrt(posterior_variance_t) * noise


    # 逆扩散过程T次采样，即通过xT和T计算xi，获得每一个时刻的图像列表[xi]，此过程需要通过网络
    @torch.no_grad()
    def p_sample_loop(model, shape):
        device = next(model.parameters()).device
        b = shape[0]
        img = torch.randn(shape, device=device)
        imgs = []
        for i in tqdm(reversed(range(0, timesteps)), desc="sampling loop time step", total=timesteps):
            img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
            imgs.append(img.cpu())
        return imgs


    # 逆扩散过程T次采样，允许传入batch_size指定生成图片的个数，用于生成结果的可视化
    @torch.no_grad()
    def sample(model, image_size, batch_size=16, channels=1):
        return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))












    
    # 读取CSV文件

    # 将DataFrame转换为numpy数组
    data_matrix = df.values
    # 数据预处理：将每个数值减去该行的最小值，然后除以该行的最大值减去最小值
    # row_min = np.min(data_matrix, axis=1, keepdims=True)
    # row_max = np.max(data_matrix, axis=1, keepdims=True)
    # scaled_matrix = (data_matrix - row_min) / (row_max - row_min) #* 255

    scaled_matrix = data_matrix #* 255
    # 将数据重新排列为4015个64x64的矩阵
    n_samples = scaled_matrix.shape[0]
    image_size = int(np.sqrt(scaled_matrix.shape[1]))
    reshaped_matrix = scaled_matrix.reshape(n_samples, image_size, image_size)
    image_tensors = torch.tensor(reshaped_matrix, dtype=torch.float32).unsqueeze(1) 
    timesteps = 2  # 总步长T
    # 以下参数均为序列(List)，需要传入t获得对应t时刻的值 xt = X[t]
    betas = linear_beta_schedule(timesteps=timesteps)  # 选择一种方式，生成β(t)
    alphas = 1. - betas  # α(t)
    alphas_cumprod = torch.cumprod(alphas, axis=0)  # α的连乘序列，对应α_bar(t)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)  # 将α_bar的最后一个值删除，在最开始添加1，对应前一个时刻的α_bar，即α_bar(t-1)
    sqrt_recip_alphas = torch.sqrt(1. / alphas)  # 1/根号下α(t)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)  # 根号下α_bar(t)
    sqrt_one_minus_alphas_cumpord = torch.sqrt(1. - alphas_cumprod)  # 根号下(1-α_bar(t))
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)  # β(t)x(1-α_bar(t-1))/(1-α_bar(t))，即β^~(t)
    total_epochs = 50
    channels = 1
    batch_size = 256
    lr = 1e-4

    # 创建一个空字典用于存储结果
    result_dict = {}

    for i in tqdm(range(len(image_tensors)), desc='Processing'):
        spot_dict = {}
        image = image_tensors[i].to(device)
        image = image.unsqueeze(0)
        
        model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(total_epochs):
                optimizer.zero_grad()

                # Randomly sample a time step
                t = torch.randint(0, timesteps, (1,), device=device).long()

                # Add noise to the image
                noise = torch.randn_like(image)
                x_noisy = q_sample(image, t, noise)

                # Denoise the image
                predicted_noise = model(x_noisy, t)
                loss = F.mse_loss(noise, predicted_noise)
                #loss = F.smooth_l1_loss(noise, predicted_noise)
                loss.backward()
                optimizer.step()

                #print(f"Epoch [{epoch + 1}/{total_epochs}], Loss: {loss.item()}")
                #print(np.max((x_noisy - predicted_noise).squeeze().detach().cpu().numpy()))
                #print(np.min((x_noisy - predicted_noise).squeeze().detach().cpu().numpy()))
                spot_dict[epoch] = (x_noisy - predicted_noise).squeeze().detach().cpu().numpy()

        
        print(len(spot_dict))
        # 将结果存储到字典中
        result_dict[i] = spot_dict

    def mse_similarity(A, B):
        """
        计算两个矩阵的MSE相似性

        Args:
            A: 第一个矩阵
            B: 第二个矩阵

        Returns:
            MSE相似性
        """

        # 计算差值的平方和
        diff = A - B
        sq_diff = np.square(diff)

        # 计算MSE相似性
        return np.mean(sq_diff)

    NumDiff=len(result_dict[0])

    average_result = {}

    # 使用循环打印字典数据
    for ij in range(len(result_dict)):
        
        original_matrix = image_tensors[ij].squeeze().cpu().numpy()
        # 计算所有矩阵与原始数据方阵的MSE相似性
        similarities = {}
        for i in range(NumDiff):
            similarities[i] = mse_similarity(original_matrix, result_dict[ij][i])

        # 选择最相近的4个方阵
        most_similar_matrices = []
        for i in range(2):
            min_similarity = float("inf")
            min_idx = None
            for j, similarity in similarities.items():
                if similarity < min_similarity:
                    min_similarity = similarity
                    min_idx = j

        most_similar_matrices.append(min_idx)
        #print(most_similar_matrices)
        # 计算最相近的5个方阵相加取平均
        average_matrix = np.zeros((64, 64))
        for i in most_similar_matrices:
            average_matrix += result_dict[ij][i]

        average_matrix /= 2
        
        # average_matrix=(average_matrix+ original_matrix)/2
        average_result[ij] = average_matrix
        # 打印结果
        print(average_matrix)
        
        print(ij, ':', average_result[ij])





    revector_result = {}

    for ij in range(len(result_dict)):
        revector_result[ij] = average_result[ij].flatten()




    # 将修复的每个图像存储为一维向量并合并为一个大矩阵
    repaired_images_matrix = np.vstack([revector_result[i].flatten() for i in range(len(revector_result))])
    # 输出结果
    # print("合并后的矩阵：")
    # print(repaired_images_matrix)

    # 输出结果
    # print("拼接后的矩阵：")
    # print(matrix)
    np.savetxt(save_name +'_example.csv', repaired_images_matrix, delimiter=',')










