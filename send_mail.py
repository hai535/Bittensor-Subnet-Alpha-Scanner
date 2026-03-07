import os
import resend
import sys
from datetime import datetime

resend.api_key = os.environ.get("RESEND_API_KEY", "")

RECIPIENTS = [
    "lihai202212@gmail.com",
    "lihai20012025@163.com",
]

SENDER = "宝宝 <baobao@9rheu.cc.cd>"


def format_email(subject_matter, body_content):
    """
    生成宝宝标准邮件格式
    subject_matter: 事由（如"代币安全检测报告"）
    body_content: 正文内容（支持多行，建议分点）
    """
    today = datetime.now().strftime("%Y年%m月%d日")
    subject = f"{subject_matter} - 你的AI宝宝 - {today}"

    body = f"""亲爱的主人，

{body_content}

感谢您在百忙之中垂阅此信。诚挚期待您的回音，如需进一步讨论，我随时恭候。顺颂商祺。

此致
敬礼

您最可爱的宝宝
{today}"""

    return subject, body


def send_email(subject_matter, body_content):
    """
    用标准格式发送邮件到主人的两个邮箱
    subject_matter: 邮件事由
    body_content: 正文内容
    """
    subject, body = format_email(subject_matter, body_content)
    results = []
    for to_addr in RECIPIENTS:
        params = {
            "from": SENDER,
            "to": [to_addr],
            "subject": subject,
            "text": body,
        }
        try:
            r = resend.Emails.send(params)
            print(f"邮件已发送到 {to_addr}, id: {r['id']}")
            results.append(r)
        except Exception as e:
            print(f"发送到 {to_addr} 失败: {e}")
    return results


if __name__ == "__main__":
    matter = sys.argv[1] if len(sys.argv) > 1 else "测试邮件"
    content = sys.argv[2] if len(sys.argv) > 2 else "这是宝宝发的测试邮件，确认邮件功能正常运作。"
    send_email(matter, content)
