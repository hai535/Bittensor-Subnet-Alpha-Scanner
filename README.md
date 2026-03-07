# Jojo - AI Chat Application

Jojo 是一个基于 Claude CLI 的 Web 聊天应用，提供类似微信风格的对话界面，支持多会话管理、流式响应、图片上传和多用户权限控制。

## 功能特性

- **实时流式对话** — 基于 Claude CLI，支持 SSE 流式输出，打字机效果逐字显示
- **多会话管理** — 创建、切换、删除多个独立对话，自动生成会话标题
- **图片上传** — 支持拖拽或点击上传图片（PNG/JPG/GIF/WebP/BMP）
- **用户认证** — Token 认证机制，管理员可增删改用户
- **定时任务面板** — 可视化查看服务器 Cron 和 Systemd 定时任务
- **移动端适配** — 响应式设计，支持手机和平板访问
- **SQLite 存储** — 轻量级本地数据库，会话和消息持久化

## 技术栈

| 组件 | 技术 |
|------|------|
| 后端 | Python Flask |
| 前端 | 原生 HTML/CSS/JS（单文件） |
| AI | Claude CLI（命令行调用） |
| 数据库 | SQLite |
| 进程管理 | PM2（可选） |

## 项目结构

```
jojo/
├── app.py              # Flask 主应用，API 路由
├── chat_store.py        # SQLite 数据库操作层
├── index.html           # 前端页面（单文件 SPA）
├── ecosystem.config.js  # PM2 进程管理配置
├── send_mail.py         # 邮件发送工具
└── .gitignore           # Git 忽略规则
```

## 部署教程

### 前置要求

- Python 3.8+
- [Claude CLI](https://docs.anthropic.com/en/docs/claude-cli) 已安装并配置
- pip

### 1. 克隆仓库

```bash
git clone https://github.com/hai535/jojo-.git
cd jojo-
```

### 2. 安装依赖

```bash
pip install flask
```

### 3. 配置用户

首次启动会自动创建 `users.json`，默认管理员 Token 为 `shamless`。你可以在启动后通过管理面板添加其他用户。

如需修改管理员 Token，编辑 `app.py` 中的 `ADMIN_USER` 变量。

### 4. 启动应用

```bash
python app.py
```

应用默认运行在 `http://0.0.0.0:7682`。

### 5. 访问使用

浏览器打开 `http://你的服务器IP:7682`，输入 Token 登录即可开始对话。

### 使用 PM2 后台运行（推荐）

```bash
# 安装 PM2
npm install -g pm2

# 启动应用
pm2 start app.py --name jojo --interpreter python3

# 开机自启
pm2 startup
pm2 save
```

### 使用 Nginx 反向代理（可选）

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:7682;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_buffering off;           # SSE 流式响应需要
        proxy_cache off;
    }
}
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/auth` | 用户登录认证 |
| POST | `/api/chat` | 发送消息（SSE 流式） |
| GET | `/api/sessions` | 获取会话列表 |
| GET | `/api/sessions/:id/messages` | 获取会话消息 |
| DELETE | `/api/sessions/:id` | 删除会话 |
| POST | `/api/upload` | 上传图片 |
| GET | `/api/crontab` | 查看定时任务 |
| GET/POST/PUT/DELETE | `/api/admin/users` | 用户管理（管理员） |

## License

MIT
