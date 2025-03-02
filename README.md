# 2025 MCM/ICM Team 2503720 项目文档

## 操作流程

### 1. 在git的官网下载git
网址：https://git-scm.com/ 下载安装即可。

### 2. 配置SSH（可选）
可以看这个教程：https://blog.csdn.net/weixin_42310154/article/details/118340458

### 3. 将项目克隆到本地
先cd到希望创建目录的地方。
1. 如果配置了SSH就在git bash中输入以下命令：
```shell
git clone git@github.com:BlackMaple1203/MCM.git
```

2. 如果没有配置SSH就在git bash中输入以下命令：
```shell
git clone https://github.com/BlackMaple1203/MCM.git
```
第二种方法可能连接不稳定。

### 4. 创建并切换到自己的分支
```shell
git checkout -b your_branch_name
```

### 5. 在自己的分支上进行操作、修改、提交

```shell
git add .
git commit -m "your commit message"
git push origin your_branch_name
```

### 6. 在github上提交pull request

## 常见问题

### 1. 在其他人更新仓库后，本地进度落后了怎么办

首先打开命令行工具，cd到项目目录下，然后输入以下命令：

将自己已经对本地文件做的修改commit到本地仓库：
```shell
git add .
git commit -m "your commit message"
```

然后将远程仓库的更新拉取到本地：
```shell
git pull origin main
```

最后选择要不要将自己的修改合并到远程仓库，如果需要：
```shell
git push origin your_branch_name
```

### 2. 编译tex文件时遇到报错`You must install Pygments`

由于在tex文件中我使用了`minted`环境来展示代码，因此电脑需要安装好python 2.7以上的环境以及`Pygments`包。

安装方法：
```shell
pip3 install Pygments
```