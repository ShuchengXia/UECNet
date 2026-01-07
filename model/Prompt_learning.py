import torch, torchvision
import torch.nn as nn
from torch.nn import functional as F
import torch.optim
import argparse
import numpy as np
import os, random
from torch.utils.data import DataLoader
from data.Prompt_data import ExposureDataset
import clip

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = "cuda" if torch.cuda.is_available() else "cpu"

def manual_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CLIPTextOnly(nn.Module):
    def __init__(self, model_name="ViT-B/32", device="cuda"):
        super().__init__()
        full_model, _ = clip.load(model_name, device=device)

        # 保留文本编码器部分
        self.token_embedding = full_model.token_embedding
        self.positional_embedding = full_model.positional_embedding
        self.transformer = full_model.transformer
        self.ln_final = full_model.ln_final
        self.text_projection = full_model.text_projection
        self.encode_text = full_model.encode_text

        # 删除视觉编码器
        del full_model.visual

        # 冻结权重
        for p in self.parameters():
            p.requires_grad = False

        self.float()


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = torch.float32

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class Backbone(nn.Module):
    def __init__(self, backbone='resnet18'):
        super(Backbone, self).__init__()

        if backbone == 'resnet18':
            resnet = torchvision.models.resnet.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        elif backbone == 'resnet50':
            resnet = torchvision.models.resnet.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        elif backbone == 'resnet101':
            resnet = torchvision.models.resnet.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1)

        self.block0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        )
        self.block1 = resnet.layer1
        self.block2 = resnet.layer2
        self.block3 = resnet.layer3
        self.block4 = resnet.layer4

    def forward(self, x, returned=[3]):
        blocks = [self.block0(x)]

        blocks.append(self.block1(blocks[-1]))
        blocks.append(self.block2(blocks[-1]))
        blocks.append(self.block3(blocks[-1]))
        blocks.append(self.block4(blocks[-1]))

        out = [blocks[i] for i in returned]
        return out


class Prompts(nn.Module):
    def __init__(self, clip_model, initials=None, train_backbone=False):
        super(Prompts, self).__init__()
        print(f"Type of initials: {type(initials)}")
        print("The initial prompts are:", initials)
        self.train_backbone = train_backbone
        self.model = clip_model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.text_encoder = TextEncoder(self.model).to(torch.float32)
        self.prompt_dim = self.model.token_embedding.embedding_dim
        self.dtype = torch.float32
        # ----------- 图像编码器 (ResNet18 替代 CLIP 图像编码器) -----------
        print("Initializing image_backbone...")
        self.image_backbone = Backbone('resnet18')
        self.img_embed = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.img_fc = nn.Linear(256, 128)  # 映射到与 text feature 相同维度

        # ----------- 文本 Prompt 初始化 -----------
        self.text = clip.tokenize(initials).cuda()
        self.embedding_prompt = nn.Parameter(
            self.model.token_embedding(self.text).to(self.dtype).requires_grad_()).cuda()  # 只有文本token
        self.text_feat_compressor = nn.Sequential(
                                                nn.Linear(512, 256),
                                                nn.ReLU(),
                                                nn.Linear(256, 128),
                                                nn.ReLU()
                                                )

    def forward(self, image, flag=1):
        # 处理文本 Prompt（text encoder）
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        tokenized_prompts = self.text
        text_features = self.text_encoder(self.embedding_prompt, tokenized_prompts).to(torch.float32)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # [num_prompts, 512]
        text_features = self.text_feat_compressor(text_features) # [num_prompts, 128]

        # 处理图像（image encoder）
        if self.train_backbone:
            feat = self.image_backbone(image)[0]  # 可训练
        else:
            with torch.no_grad():  # 不训练时省显存
                for param in self.image_backbone.parameters():
                    param.requires_grad = False
                    feat = self.image_backbone(image)[0]
        feat = self.img_embed(feat)              # [B, 1024, 1, 1]
        feat = feat.view(feat.size(0), -1)       # [B, 1024]
        image_features = self.img_fc(feat)       # [B, 512]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # 相似度匹配
        logits = 100.0 * image_features @ text_features.T  # [B, num_prompts]
        if flag == 0:
            return logits
        else:
            return logits.softmax(dim=-1)


def train(config):
    clip_model = CLIPTextOnly("ViT-B/32", device=device)
    for para in clip_model.parameters():
        para.requires_grad = False

    prompt_model = Prompts(
        clip_model=clip_model,
        train_backbone=config.train_backbone,
        initials=[
            "Natural lighting, balanced contrast region.",
            "White glare and intense highlight region.",
            "Slightly overexposed region.",
            "Underexposed and dim region.",
            "Completely dark and black region."
        ]
    ).to(device)

    # 加载训练/验证数据
    prompt_train_dataset = ExposureDataset(config.train_path)
    prompt_train_loader = DataLoader(prompt_train_dataset, batch_size=config.prompt_batch_size, shuffle=True,
                                     num_workers=config.num_workers, pin_memory=True)

    prompt_val_dataset = ExposureDataset(config.val_path)
    prompt_val_loader = DataLoader(prompt_val_dataset, batch_size=config.prompt_batch_size, shuffle=False,
                                   num_workers=config.num_workers, pin_memory=True)

    prompt_optimizer = torch.optim.Adam(prompt_model.parameters(), lr=config.prompt_lr, weight_decay=config.weight_decay)

    best_val_acc = 0.0

    for epoch in range(config.num_epochs):
        prompt_model.train()
        epoch_loss = 0.0
        train_correct = 0
        train_total = 0
        n_steps = 0

        for iteration, item in enumerate(prompt_train_loader):
            img_lowlight, label, _ = item
            img_lowlight = img_lowlight.cuda()
            label = label.cuda()
            output = prompt_model(img_lowlight, 0)
            loss = F.cross_entropy(output, label)

            prompt_optimizer.zero_grad()
            loss.backward()
            prompt_optimizer.step()

            n_steps += 1
            epoch_loss += loss.item()

            preds = output.argmax(dim=1)
            train_correct += (preds == label).sum().item()
            train_total += label.size(0)

            if ((iteration + 1) % config.prompt_display_iter) == 0:
                print(f'Epoch:{epoch + 1}|Iter:{iteration + 1}/{len(prompt_train_loader)}|lr:{prompt_optimizer.param_groups[0]["lr"]:.6f},'
                      f' Loss: {loss.item():.3f}')

        train_acc = train_correct / train_total

        # ---------- 验证阶段 ----------
        prompt_model.eval()
        val_correct = 0
        val_total = 0
        outputs_with_filenames = []  # <-- 新增：用于收集输出与文件名

        with torch.no_grad():
            for img_val, label_val, filename in prompt_val_loader:
                img_val = img_val.cuda()
                label_val = label_val.cuda()
                output = prompt_model(img_val, 0)
                preds = output.argmax(dim=1)

                val_correct += (preds == label_val).sum().item()
                val_total += label_val.size(0)

                for i in range(len(filename)):
                    outputs_with_filenames.append({
                        "filename": filename[i],
                        "logits": output[i].cpu().numpy(),  # numpy格式
                        "pred": preds[i].item(),
                        "label": label_val[i].item()
                    })

        val_acc = val_correct / val_total

        # ---------- 日志输出 ----------
        print(f"[Epoch {epoch + 1}] "
              f"Train Loss: {epoch_loss / n_steps:.4f}, "
              f"Train Acc: {train_acc * 100:.2f}%, "
              f"Val Acc: {val_acc * 100:.2f}%")

        # ---------- 模型保存 ----------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(prompt_model.state_dict(), config.result_folder + "best_valacc_class5_06271723.pth")
            print(f"[INFO] Best model saved at Epoch {epoch + 1} with Val Acc: {val_acc * 100:.2f}%")

            # 打印所有输出
            print("[INFO] Output logits of best model epoch:")
            for item in outputs_with_filenames:
                print(f"Filename: {item['filename']}, "
                    f"Label: {item['label']}, "
                    f"Pred: {item['pred']}, "
                    f"Logits: {np.round(item['logits'], 4)}")

            # （可选）写入文件保存
            output_path = os.path.join(config.result_folder, f"val_outputs_epoch{epoch + 1}.txt")
            with open(output_path, "w") as f:
                for item in outputs_with_filenames:
                    logit_str = ", ".join([f"{x:.4f}" for x in item['logits']])
                    f.write(f"{item['filename']}, label={item['label']}, pred={item['pred']}, logits=[{logit_str}]\n")
            print(f"[INFO] Saved validation outputs to {output_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('-b', '--train_path', type=str, default="/ext_ssd/xsc_datasets/prompt_learning/train/")
    parser.add_argument('--val_path', type=str, default="/ext_ssd/xsc_datasets/prompt_learning/test/")
    parser.add_argument('--length_prompt', type=int, default=16)
    parser.add_argument('--prompt_lr', type=float, default=0.00008)  # 0.00001#0.00008
    parser.add_argument('--weight_decay', type=float, default=0.001)  # 0.0001
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--prompt_batch_size', type=int, default=64)  # 32
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--result_folder', type=str, default='/home/xsc/LightRestore/result/Prompt/')
    parser.add_argument('--prompt_display_iter', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--train_backbone', default=True, help='Enable training of ResNet backbone')

    config = parser.parse_args()
    manual_seed(config.seed)

    train(config)