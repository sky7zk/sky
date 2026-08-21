# DoseRAD2026 提交：你需要做的事情（无需安装 Docker）

本流程使用 **GitHub + Grand Challenge 自动构建**，你的电脑和训练服务器都不需要安装 Docker。

目标：将 Epoch 150 模型提交到：

- 比赛：DoseRAD2026
- 任务：Photon dose calculation on CT
- 阶段：Preliminary test - Photon dose on CT
- GitHub仓库：<https://github.com/sky7zk/sky>

## 当前进度（2026-08-21）

- GitHub CLI 已登录 `sky7zk`；
- 提交代码已推送到 `codex/doserad-photon-ct-submission` 分支；
- 草稿 PR：<https://github.com/sky7zk/sky/pull/1>；
- 本地单元测试：`3 passed`；
- `model.tar.gz` 已准备好，没有上传到公开 GitHub 仓库。

你现在应从下面的“第三步”开始；第一、二步已完成。

## 第一步：登录 GitHub（已完成）

打开服务器终端，检查是否已有 GitHub CLI：

```bash
gh --version
```

如果提示找不到 `gh`，只安装 GitHub CLI，不需要安装 Docker：

```bash
sudo apt-get update
sudo apt-get install -y gh
```

登录：

```bash
gh auth login
```

依次选择：

```text
GitHub.com
HTTPS
Login with a web browser
```

终端会显示一个临时验证码和网页地址。你需要亲自在浏览器打开网页，登录 GitHub 并确认授权。

完成后检查：

```bash
gh auth status
```

看到已经登录 `sky7zk` 后，告诉我：

```text
GitHub登录好了
```

不要把 GitHub 密码、Token、Cookie或验证码发给我。

## 第二步：推送代码（已完成）

已完成代码检查、测试和推送。模型权重、患者数据和本地测试数据均未上传。

打开 <https://github.com/sky7zk/sky/pull/1>，检查后依次点击：

```text
Ready for review
Merge pull request
Confirm merge
```

合并后确认 <https://github.com/sky7zk/sky> 的 `main` 分支上能看到 `Dockerfile`。

## 第三步：加入 DoseRAD2026

1. 登录 <https://grand-challenge.org/accounts/login/>。
2. 确认账号已经完成验证。
3. 打开：<https://doserad2026.grand-challenge.org/participants/registration/create/>
4. 点击加入 DoseRAD2026。

注意：每个团队只能使用一个账号参赛，团队最多5人。

## 第四步：在 Grand Challenge 创建 Algorithm

代码推送到 GitHub 后，打开：

<https://doserad2026.grand-challenge.org/evaluation/photon-dose-preliminary-testing/algorithms/create/>

建议填写：

```text
Title: DoseRAD CNN-ConvLSTM Photon CT
GPU: NVIDIA A10G Tensor Core GPU (24GiB VRAM)
Memory: 32 GiB（如果页面允许选择）
```

确认当前阶段是：

```text
Preliminary test - Photon dose on CT
```

不要误选 MRI、Proton 或 Final test。

## 第五步：连接 GitHub 仓库

创建 Algorithm 后，进入它的 `Containers` 或 `Manage Containers` 页面：

1. 选择连接 GitHub repository；
2. 如果出现 GitHub授权页面，允许 Grand Challenge 访问仓库；
3. 选择仓库 `sky7zk/sky`；
4. 选择分支 `main`；
5. 启动构建；
6. 等待容器状态变为 `Active`。

构建容器可能需要一段时间。容器构建失败时，点击信息按钮查看日志，并把错误截图或日志发给我。

构建或更新容器不会消耗比赛的10次 preliminary submission机会。

## 第六步：上传模型文件

模型权重没有上传到公开 GitHub，需要在 Grand Challenge 单独上传。

模型包位于：

```text
/home/a123456/zk/DoseRAD_v2_transfer_20260804/sky/model.tar.gz
```

文件大小约为6.4 MB。

在 Algorithm 页面找到 `Models`：

1. 点击上传模型；
2. 选择 `model.tar.gz`；
3. 等待上传、解包和验证完成；
4. 确认该模型已经与当前 Algorithm关联或启用。

模型包内包含：

```text
best_full_ct_beam_mae.pth
normalization.json
```

## 第七步：创建第一次 Preliminary Submission

必须等到下面两项都准备好：

```text
Container: Active
Model: 已上传并启用
```

然后打开：

<https://doserad2026.grand-challenge.org/evaluation/photon-dose-preliminary-testing/submissions/create/>

填写：

```text
Algorithm: 选择刚创建的 DoseRAD CNN-ConvLSTM Photon CT
Comment: Epoch 150 baseline, batch size 2, first interface test
```

点击 `Save` 创建评估。

第一次提交的目标首先是确认容器、输入输出和运行时间正确，不要只关注排行榜名次。

## 第八步：提交后需要查看什么

Preliminary阶段允许查看日志。按以下顺序检查：

1. 容器是否成功启动；
2. `/health` 是否成功；
3. `/invoke` 是否成功；
4. 是否生成10个输出槽；
5. `Implementation Errors Count` 是否为0；
6. GPU是否发生显存不足；
7. Photon runtime是否低于181秒；
8. Beam MAE；
9. IDD Curve Distance；
10. Stratified Plan-Level MAE；
11. Local Gamma Index（官方是完整计划1%/1 mm）；
12. DVH-Based Clinical Score。

评估结束后，把提交详情页截图、全部分数和日志发给我，我会分析下一次应该修改什么。

## 常见问题

### GitHub容器构建失败

把构建日志发给我。不要立即创建submission；容器构建不占submission次数，可以反复修复。

### 找不到模型

检查 `model.tar.gz` 是否上传到同一个 Algorithm 的 `Models` 页面，并确认已经启用。

### CUDA out of memory

先不要重复提交。告诉我日志内容，我会把推理batch size从2改为1，再让Grand Challenge重新构建。

### Implementation Errors Count不为0

把日志和错误数发给我。重点检查比赛的 `minimum_cutoff` 单位和输出剂量标度，不要连续浪费submission机会。

### Runtime超过181秒

把平台报告的runtime和显存峰值发给我，再决定是否将batch size调到4或优化重采样。

## 你现在只需要做什么

按照以下顺序操作：

1. 打开 <https://github.com/sky7zk/sky/pull/1> 并合并到 `main`；
2. 从本文档第三步开始，加入 DoseRAD2026；
3. 创建 Photon CT Preliminary Algorithm；
4. 连接 `sky7zk/sky` 的 `main` 分支并等待 Container 变为 `Active`；
5. 上传服务器上的 `model.tar.gz`；
6. 容器和模型都就绪后，创建第一次 Preliminary Submission。

如果你不想自己点击合并，直接告诉我：

```text
帮我合并PR
```
