"""
任务进度上报模块 - 供长时间运行的脚本使用

使用方法:
    from task_progress import TaskProgress

    tp = TaskProgress("获取内置钱包")
    tp.update("登录钱包1", current=1, total=3)
    tp.update("获取deposit地址", current=2, total=3)
    tp.done()  # 清理进度文件
"""
import json
import os

PROGRESS_FILE = "/tmp/jojo_task_progress.json"


class TaskProgress:
    def __init__(self, task_name=""):
        self.task_name = task_name
        self.total = 0
        self.current = 0

    def update(self, step, current=None, total=None, percent=None):
        """更新进度"""
        if current is not None:
            self.current = current
        if total is not None:
            self.total = total

        data = {
            "task": self.task_name,
            "step": step,
        }
        if self.current and self.total:
            data["current"] = self.current
            data["total"] = self.total
            if percent is None:
                data["percent"] = round((self.current / self.total) * 100)
        if percent is not None:
            data["percent"] = percent

        try:
            with open(PROGRESS_FILE, 'w') as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception:
            pass

    def done(self):
        """任务完成，清理进度文件"""
        try:
            if os.path.exists(PROGRESS_FILE):
                os.remove(PROGRESS_FILE)
        except Exception:
            pass


def update_progress(task, step, current=None, total=None, percent=None):
    """快捷函数 - 一行代码上报进度"""
    data = {"task": task, "step": step}
    if current is not None:
        data["current"] = current
    if total is not None:
        data["total"] = total
    if percent is not None:
        data["percent"] = percent
    elif current and total:
        data["percent"] = round((current / total) * 100)
    try:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception:
        pass


def clear_progress():
    """清理进度文件"""
    try:
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
    except Exception:
        pass
