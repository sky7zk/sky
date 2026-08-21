# DoseRAD2026 Photon-on-CT 提交操作指南

本文档描述你需要亲自完成的网页操作，以及提交前后如何检查结果。目标阶段是：

- Challenge：DoseRAD2026
- Task：Photon dose calculation on CT
- Phase：Preliminary test - Photon dose on CT
- 截止日期：2026-08-31
- Preliminary 每个任务最多提交10次

## 1. 账号准备

1. 登录 <https://grand-challenge.org/accounts/login/>。
2. 确认账号已经通过邮箱/身份验证。未验证账号不能提交。
3. 打开 <https://doserad2026.grand-challenge.org/participants/registration/create/>。
4. 点击加入 DoseRAD2026。每个团队只能使用一个账号，团队最多5人。

不要把 Grand Challenge 或 GitHub 密码、验证码、Cookie 发给任何人。

## 2. 提交文件说明

本仓库用于构建算法容器。权重不提交到 GitHub。

- GitHub仓库：<https://github.com/sky7zk/sky>
- 容器代码：本仓库根目录
- 模型文件：`model.tar.gz`
- 使用的checkpoint：Epoch 150 `best_full_ct_beam_mae.pth`
- 归一化配置：`normalization.json`

在训练服务器生成模型包：

```bash
cd /home/a123456/zk/DoseRAD_v2_transfer_20260804/sky
./prepare_model.sh
ls -lh model.tar.gz
```

`model.tar.gz` 不会被 Git 提交，需要在 Grand Challenge 的 Algorithm → Models 页面单独上传。

## 3. 推荐方式：GitHub自动构建容器

### 3.0 当前服务器的一次性准备

当前服务器是 Ubuntu 24.04，尚未安装Docker和GitHub CLI。请在服务器终端执行：

```bash
sudo apt-get update
sudo apt-get install -y docker.io gh
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
```

安装NVIDIA Container Toolkit（Docker使用GPU所需）：

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

`usermod` 后退出当前登录会话并重新登录，或者临时执行：

```bash
newgrp docker
```

验证Docker与GPU：

```bash
docker --version
docker run --rm --gpus all ubuntu:24.04 nvidia-smi
```

登录GitHub CLI：

```bash
gh auth login
gh auth status
```

在交互选项中建议选择：

```text
GitHub.com
HTTPS
Login with a web browser
```

浏览器授权完成后，不要把验证码或token粘贴进聊天。

### 3.1 推送代码

将本仓库代码提交并推送到 `sky7zk/sky` 的 `main` 分支。推送前确认：

```bash
git status
git diff --check
git add .
git commit -m "Add DoseRAD2026 Photon CT submission container"
git push origin main
```

确认 GitHub网页中看不到 `.pth`、训练数据、患者数据或 `model.tar.gz`。

### 3.2 创建 Algorithm

打开：

<https://doserad2026.grand-challenge.org/evaluation/photon-dose-preliminary-testing/algorithms/create/>

填写：

- Title：建议 `DoseRAD CNN-ConvLSTM Photon CT`
- GPU：`NVIDIA A10G Tensor Core GPU (24GiB VRAM)`
- Memory：若页面允许选择，优先32GiB
- Algorithm类型/接口：使用该 preliminary phase 自动提供的接口

保存后进入 Algorithm 管理页。

### 3.3 连接GitHub

在 Algorithm 的 Containers/Manage Containers 页面：

1. 选择连接 GitHub repository；
2. 授权 Grand Challenge 访问 `sky7zk/sky`；
3. 选择仓库 `sky7zk/sky`；
4. 选择 `main` 或创建的 release/tag；
5. 启动构建；
6. 等待状态变为 `Active`。

构建可能需要较长时间。构建失败时点击信息按钮查看日志。构建和上传容器不消耗10次submission额度。

### 3.4 上传模型

进入 Algorithm → Models：

1. 上传本地生成的 `model.tar.gz`；
2. 等待解包/验证完成；
3. 确认它是该 Algorithm 当前启用的模型。

运行时文件会解压到 `/opt/ml/model`，程序会自动寻找：

```text
best_full_ct_beam_mae.pth
normalization.json
```

## 4. 本地Docker测试（强烈建议）

当前训练服务器尚未安装 Docker。若在有 Docker 和 NVIDIA Container Toolkit 的机器测试：

```bash
cd sky
python prepare_local_test_input.py \
  --patient-dir /path/to/photon/training/1ABB039 \
  --control-points 4
./do_test_run.sh
```

脚本会构建镜像、挂载模型、使用 `--network=none` 启动容器、等待
`/health=200`、调用 `/invoke` 并确认返回201。输出必须位于：

```text
/output/images/stacked-radiation-dose-map-1/output.mha
...
/output/images/stacked-radiation-dose-map-10/output.mha
```

## 5. 第一次Preliminary提交

当容器和模型均为 Active 后，打开：

<https://doserad2026.grand-challenge.org/evaluation/photon-dose-preliminary-testing/submissions/create/>

1. 顶部确认阶段是 `Preliminary test - Photon dose on CT`；
2. Algorithm选择刚创建的算法；
3. Comment建议填写：`Epoch 150 baseline, batch size 2, first interface test`；
4. 点击 Save；
5. 记录这是第几次submission。

不要在第一次提交前切到 Final test。Preliminary有日志，Final没有日志且只有2次机会。

## 6. 提交后的检查顺序

先看状态和日志，再看分数：

1. 容器是否成功启动；
2. `/health` 是否返回200；
3. `/invoke` 是否返回201；
4. 是否写出10个输出槽；
5. 是否有 `Implementation Errors Count`；
6. GPU显存峰值是否小于24GiB；
7. Runtime是否低于Photon硬限制181秒；
8. 最后查看 Beam MAE、IDD、Plan MAE、1%/1mm local gamma和DVH。

第一次提交的首要目标是验证接口、显存和runtime，不要只看最终名次。

## 7. 常见错误处理

### CUDA out of memory

修改Dockerfile中的：

```dockerfile
ENV DOSERAD_BATCH_SIZE=1
```

重新构建容器。不要修改模型或输出格式。

### 模型文件找不到

检查 Models 页面是否上传并启用了 `model.tar.gz`。压缩包内必须直接或在子目录中包含两个唯一文件：

```text
best_full_ct_beam_mae.pth
normalization.json
```

### 输出格式错误

检查每个输出目录只有一个 `.mha`，所有真实和占位输出均为4D。真实输出必须按 `idx_in_output` 堆叠。

### Implementation Errors Count大于0

重点检查 `minimum_cutoff`：每个dosemap中小于或等于其cutoff的正值必须为0。若平台日志显示cutoff单位与训练数据剂量标度不一致，暂停重复提交，先在比赛论坛询问组织者。

### Runtime超过181秒

依次尝试：

1. batch size从2调到4（前提是显存安全）；
2. 减少Python日志；
3. 避免MHA压缩；
4. 保持模型在`/health`阶段加载；
5. 分析CT重采样和BEV回采样耗时。

## 8. 官方链接

- 提交说明：<https://doserad2026.grand-challenge.org/submission-instructions/>
- 指标与排名：<https://doserad2026.grand-challenge.org/metrics-and-ranking/>
- 官方示例：<https://github.com/DoseRAD2026/example-submission>
- Grand Challenge容器说明：<https://grand-challenge.org/documentation/building-and-testing-the-container/>
- Preliminary leaderboard：<https://doserad2026.grand-challenge.org/evaluation/photon-dose-preliminary-testing/leaderboard/>
- 比赛论坛：<https://doserad2026.grand-challenge.org/forum/topics/>
