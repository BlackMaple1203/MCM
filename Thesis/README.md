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
