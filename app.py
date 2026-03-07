import os
import json
import subprocess
import threading
import uuid
from flask import Flask, request, jsonify, send_file, Response, stream_with_context
import chat_store

app = Flask(__name__)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")
ADMIN_USER = "shamless"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return set(json.load(f))
    return {ADMIN_USER}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(sorted(list(users)), f)

chat_store.init_db()

def get_user():
    """Extract and validate token from request. Returns user token or None."""
    token = request.headers.get("X-Auth-Token", "")
    if token in load_users():
        return token
    return None

def is_admin():
    return get_user() == ADMIN_USER

@app.route("/")
def index():
    return send_file("index.html")

@app.route("/api/auth", methods=["POST"])
def auth():
    data = request.json or {}
    token = data.get("token", "")
    users = load_users()
    if token in users:
        # Non-admin users: clear all their sessions (no memory)
        if token != ADMIN_USER:
            chat_store.clear_user_sessions(token)
        return jsonify({"ok": True, "user": token, "is_admin": token == ADMIN_USER})
    return jsonify({"ok": False, "error": "Invalid token"}), 401

@app.route("/api/chat", methods=["POST"])
def chat():
    user = get_user()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.json
    session_id = data.get("session_id", "default")
    message = data.get("message", "")

    chat_store.add_message(session_id, "user", message, user=user)

    # Build conversation context for claude CLI
    # Use the last few messages as context
    history = chat_store.get_messages(session_id)
    prompt_parts = []
    for msg in history[-10:]:  # last 10 messages for context
        if msg["role"] == "user":
            prompt_parts.append(f"User: {msg['content']}")
        else:
            prompt_parts.append(f"Assistant: {msg['content']}")

    full_prompt = "\n".join(prompt_parts)
    if len(history) > 1:
        full_prompt = "Previous conversation:\n" + "\n".join(prompt_parts[:-1]) + "\n\nNow respond to the latest message: " + message
    else:
        full_prompt = message

    def generate():
        try:
            env = os.environ.copy()
            env.pop("CLAUDECODE", None)
            proc = subprocess.Popen(
                ["claude", "-p", full_prompt],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                bufsize=1
            )

            full_response = ""
            while True:
                chunk = proc.stdout.read(20)
                if not chunk:
                    break
                full_response += chunk
                yield f"data: {json.dumps({'text': chunk})}\n\n"

            proc.wait()
            if proc.returncode != 0:
                err = proc.stderr.read()
                if err and not full_response:
                    yield f"data: {json.dumps({'error': err})}\n\n"

            if full_response:
                chat_store.add_message(session_id, "assistant", full_response, user=user)

            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")

@app.route("/api/clear", methods=["POST"])
def clear():
    user = get_user()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.json
    session_id = data.get("session_id", "default")
    chat_store.delete_session(session_id)
    return jsonify({"ok": True})

@app.route("/api/sessions", methods=["GET"])
def sessions():
    user = get_user()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify(chat_store.list_sessions(user=user))

@app.route("/api/sessions/<session_id>/messages", methods=["GET"])
def session_messages(session_id):
    user = get_user()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify(chat_store.get_messages(session_id))

@app.route("/api/sessions/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    user = get_user()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    chat_store.delete_session(session_id)
    return jsonify({"ok": True})

def describe_cron(raw, task_type="cron", unit="", source=""):
    """Generate a short Chinese description for a cron job (max 50 chars)."""
    text = raw.lower()
    cmd = raw

    # For cron/cron.d, extract the command part (skip schedule fields)
    if task_type in ("cron", "cron.d"):
        parts = raw.split()
        # Standard cron: 5 schedule fields + command
        if len(parts) > 5:
            cmd = " ".join(parts[5:])
        # cron.d has 6th field as user
        if task_type == "cron.d" and len(parts) > 6:
            cmd = " ".join(parts[6:])
        text = cmd.lower()

    # Match common patterns
    if "certbot" in text or "letsencrypt" in text or "ssl" in text:
        return "自动续期SSL证书，保持HTTPS安全连接"
    if "backup" in text or "rsync" in text or "mysqldump" in text or "pg_dump" in text:
        return "定时备份数据，防止数据丢失"
    if "logrotate" in text or "log" in text and "rotate" in text:
        return "日志轮转清理，防止磁盘空间不足"
    if "apt" in text and ("update" in text or "upgrade" in text):
        return "自动更新系统软件包"
    if "python" in text or ".py" in text:
        script = ""
        for p in cmd.split():
            if p.endswith(".py"):
                script = os.path.basename(p)
                break
        if script:
            return f"定时运行Python脚本 {script}"
        return "定时运行Python脚本任务"
    if "node" in text or ".js" in text:
        return "定时运行Node.js脚本任务"
    if "curl" in text or "wget" in text:
        return "定时发送HTTP请求或下载文件"
    if "rm " in text or "find" in text and "delete" in text:
        return "定时清理临时文件或过期数据"
    if "mail" in text or "sendmail" in text:
        return "定时发送邮件通知"
    if "docker" in text:
        return "Docker容器相关定时任务"
    if "nginx" in text:
        return "Nginx服务相关定时操作"
    if "reboot" in text or "shutdown" in text:
        return "定时重启或关机任务"
    if "monitor" in text or "check" in text or "health" in text:
        return "定时健康检查或监控任务"
    if "sync" in text:
        return "定时数据同步任务"
    if "cron" in text and "clean" in text:
        return "清理过期的定时任务会话"

    # Systemd timer descriptions
    if task_type == "systemd":
        name = unit.replace(".timer", "").lower()
        if "apt" in name:
            return "APT软件包自动更新检查"
        if "fstrim" in name:
            return "SSD磁盘TRIM优化，延长硬盘寿命"
        if "logrotate" in name:
            return "系统日志自动轮转清理"
        if "man-db" in name:
            return "更新man手册页数据库"
        if "systemd-tmpfiles" in name:
            return "清理系统临时文件目录"
        if "motd" in name:
            return "更新登录欢迎信息"
        if "dpkg" in name:
            return "dpkg数据库维护"
        if "update-notifier" in name:
            return "检查可用的系统更新"
        if "snapd" in name:
            return "Snap应用自动更新管理"
        return f"系统定时服务：{unit.replace('.timer', '')}"

    # For cron.d, mention source
    if task_type == "cron.d" and source:
        return f"来自 {source} 的系统定时任务"

    return "服务器定时执行的计划任务"


@app.route("/api/crontab", methods=["GET"])
def crontab():
    tasks = []

    # User crontab
    try:
        result = subprocess.run(["crontab", "-l"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    desc = describe_cron(line, "cron")
                    tasks.append({"type": "cron", "raw": line, "desc": desc})
    except Exception:
        pass

    # /etc/cron.d/
    try:
        for fname in os.listdir("/etc/cron.d"):
            fpath = os.path.join("/etc/cron.d", fname)
            if os.path.isfile(fpath):
                with open(fpath) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and not line.startswith("SHELL") and not line.startswith("PATH") and not line.startswith("MAILTO"):
                            desc = describe_cron(line, "cron.d", source=fname)
                            tasks.append({"type": "cron.d", "source": fname, "raw": line, "desc": desc})
    except Exception:
        pass

    # Systemd timers
    try:
        result = subprocess.run(
            ["systemctl", "list-timers", "--no-pager", "--plain"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            for line in lines[1:]:  # skip header
                parts = line.split()
                if len(parts) >= 5 and not line.startswith(" "):
                    timer_name = ""
                    activates = ""
                    for p in parts:
                        if p.endswith(".timer"):
                            timer_name = p
                        elif p.endswith(".service"):
                            activates = p
                    if timer_name:
                        desc = describe_cron(line, "systemd", unit=timer_name)
                        tasks.append({
                            "type": "systemd",
                            "unit": timer_name,
                            "activates": activates,
                            "raw": line,
                            "desc": desc
                        })
    except Exception:
        pass

    return jsonify(tasks)


@app.route("/api/upload", methods=["POST"])
def upload():
    user = get_user()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": "Unsupported file type"}), 400
    fname = uuid.uuid4().hex + ext
    fpath = os.path.join(UPLOAD_DIR, fname)
    f.save(fpath)
    return jsonify({"ok": True, "url": f"/uploads/{fname}", "filename": f.filename})

@app.route("/uploads/<filename>")
def serve_upload(filename):
    fpath = os.path.join(UPLOAD_DIR, filename)
    if not os.path.isfile(fpath):
        return "Not found", 404
    return send_file(fpath)

@app.route("/api/admin/users", methods=["GET"])
def admin_list_users():
    if not is_admin():
        return jsonify({"error": "Forbidden"}), 403
    users = sorted(list(load_users()))
    return jsonify(users)

@app.route("/api/admin/users", methods=["POST"])
def admin_add_user():
    if not is_admin():
        return jsonify({"error": "Forbidden"}), 403
    data = request.json or {}
    token = data.get("token", "").strip()
    if not token:
        return jsonify({"error": "Token cannot be empty"}), 400
    users = load_users()
    if token in users:
        return jsonify({"error": "User already exists"}), 400
    users.add(token)
    save_users(users)
    return jsonify({"ok": True})

@app.route("/api/admin/users/<token>", methods=["PUT"])
def admin_edit_user(token):
    if not is_admin():
        return jsonify({"error": "Forbidden"}), 403
    if token == ADMIN_USER:
        return jsonify({"error": "Cannot modify admin user"}), 400
    data = request.json or {}
    new_token = data.get("token", "").strip()
    if not new_token:
        return jsonify({"error": "Token cannot be empty"}), 400
    users = load_users()
    if token not in users:
        return jsonify({"error": "User not found"}), 404
    if new_token != token and new_token in users:
        return jsonify({"error": "New token already exists"}), 400
    users.discard(token)
    users.add(new_token)
    save_users(users)
    # Update sessions user field
    chat_store.rename_user(token, new_token)
    return jsonify({"ok": True})

@app.route("/api/admin/users/<token>", methods=["DELETE"])
def admin_delete_user(token):
    if not is_admin():
        return jsonify({"error": "Forbidden"}), 403
    if token == ADMIN_USER:
        return jsonify({"error": "Cannot delete admin user"}), 400
    users = load_users()
    if token not in users:
        return jsonify({"error": "User not found"}), 404
    users.discard(token)
    save_users(users)
    # Delete all their sessions
    chat_store.clear_user_sessions(token)
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7682, debug=False)
