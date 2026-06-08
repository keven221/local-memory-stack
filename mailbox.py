#!/usr/bin/env python3
"""文件邮箱 — Agent 间零依赖消息传递。

每个消息是一个 JSON 文件，写入邮箱目录即投递。
无守护进程，无网络依赖，纯文件系统操作。

用法：
  # 命令行
  python3 mailbox.py send --to cron-job-research --subject "新发现" --body "..."
  python3 mailbox.py list --box cron-job-research
  python3 mailbox.py read --box cron-job-research
  python3 mailbox.py read --box cron-job-research --id msg_xxx
  python3 mailbox.py ack --box cron-job-research --id msg_xxx
  python3 mailbox.py stats

  # Python
  from mailbox import Mailbox
  mb = Mailbox()
  mb.send("cron-research", subject="发现", body="内容")
  messages = mb.list("cron-research")
  mb.ack("cron-research", messages[0]["id"])

消息格式：
  {
    "id": "msg_<hex8>",
    "from": "<sender>",
    "to": "<mailbox>",
    "subject": "...",
    "body": "...",
    "timestamp": "ISO8601",
    "status": "unread|read|acked",
    "ttl_hours": 72,        # 过期自动清理
    "priority": "normal",   # low|normal|high
    "ref": null             # 可选：引用外部文件路径
  }

邮箱结构：
  ~/.hermes/mailbox/
    ├── _global/              # 广播邮箱
    ├── cron-research/        # 自学 cron
    │   ├── msg_a1b2c3d4.json
    │   └── msg_e5f6g7h8.json
    └── ...
"""

import json
import os
import sys
import time
import uuid
import glob
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

# 默认邮箱根目录
MAILBOX_ROOT = os.path.expanduser("~/.hermes/mailbox")
DEFAULT_TTL_HOURS = 72  # 3 天过期


class Mailbox:
    """文件邮箱 — send/list/read/ack/cleanup。"""

    def __init__(self, root: str = MAILBOX_ROOT):
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _box_dir(self, name: str) -> Path:
        """获取邮箱目录（自动创建）。"""
        d = self._root / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def send(self, to: str, subject: str, body: str,
             sender: str = "agent",
             priority: str = "normal",
             ttl_hours: int = DEFAULT_TTL_HOURS,
             ref: Optional[str] = None) -> Dict[str, Any]:
        """发送消息到指定邮箱。

        Args:
            to: 邮箱名（如 "cron-research", "_global"）
            subject: 主题
            body: 正文
            sender: 发送者标识
            priority: low|normal|high
            ttl_hours: 过期时间（小时）
            ref: 可选引用路径（如大文件的路径）

        Returns:
            消息 dict
        """
        msg_id = f"msg_{uuid.uuid4().hex[:8]}"
        msg = {
            "id": msg_id,
            "from": sender,
            "to": to,
            "subject": subject,
            "body": body,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "unread",
            "ttl_hours": ttl_hours,
            "priority": priority,
            "ref": ref,
        }

        box_dir = self._box_dir(to)
        filepath = box_dir / f"{msg_id}.json"
        filepath.write_text(json.dumps(msg, ensure_ascii=False, indent=2))
        return msg

    def list(self, box: str, status: Optional[str] = None,
             include_expired: bool = False) -> List[Dict[str, Any]]:
        """列出邮箱中的消息。

        Args:
            box: 邮箱名
            status: 过滤状态（unread/read/acked/None=全部）
            include_expired: 是否包含过期消息

        Returns:
            消息列表（按时间排序）
        """
        box_dir = self._box_dir(box)
        messages = []
        now = datetime.now(timezone.utc)

        for f in sorted(box_dir.glob("msg_*.json")):
            try:
                msg = json.loads(f.read_text())
            except Exception:
                continue

            # 过期检查
            if not include_expired:
                ts = datetime.fromisoformat(msg.get("timestamp", "").replace("Z", "+00:00"))
                ttl = msg.get("ttl_hours", DEFAULT_TTL_HOURS)
                if (now - ts).total_seconds() > ttl * 3600:
                    continue

            # 状态过滤
            if status and msg.get("status") != status:
                continue

            messages.append(msg)

        return sorted(messages, key=lambda m: m.get("timestamp", ""))

    def read(self, box: str, msg_id: Optional[str] = None) -> Any:
        """读取消息（自动标记为 read）。

        Args:
            box: 邮箱名
            msg_id: 消息 ID。None 时读取最早的一条未读消息。

        Returns:
            消息 dict，或 None
        """
        if msg_id:
            filepath = self._box_dir(box) / f"{msg_id}.json"
            if not filepath.exists():
                return None
            msg = json.loads(filepath.read_text())
        else:
            unread = self.list(box, status="unread")
            if not unread:
                return None
            msg = unread[0]
            filepath = self._box_dir(box) / f"{msg['id']}.json"

        # 标记为已读
        if msg.get("status") == "unread":
            msg["status"] = "read"
            filepath.write_text(json.dumps(msg, ensure_ascii=False, indent=2))

        return msg

    def ack(self, box: str, msg_id: str) -> bool:
        """确认消息（标记为 acked）。"""
        filepath = self._box_dir(box) / f"{msg_id}.json"
        if not filepath.exists():
            return False
        msg = json.loads(filepath.read_text())
        msg["status"] = "acked"
        filepath.write_text(json.dumps(msg, ensure_ascii=False, indent=2))
        return True

    def delete(self, box: str, msg_id: str) -> bool:
        """删除消息。"""
        filepath = self._box_dir(box) / f"{msg_id}.json"
        if filepath.exists():
            filepath.unlink()
            return True
        return False

    def cleanup(self, box: Optional[str] = None) -> Dict[str, int]:
        """清理过期消息。

        Args:
            box: 指定邮箱。None 时清理所有邮箱。

        Returns:
            {"cleaned": N, "remaining": M}
        """
        now = datetime.now(timezone.utc)
        cleaned = 0
        remaining = 0

        boxes = [box] if box else [d.name for d in self._root.iterdir() if d.is_dir()]

        for b in boxes:
            box_dir = self._root / b
            if not box_dir.is_dir():
                continue
            for f in box_dir.glob("msg_*.json"):
                try:
                    msg = json.loads(f.read_text())
                    ts = datetime.fromisoformat(msg.get("timestamp", "").replace("Z", "+00:00"))
                    ttl = msg.get("ttl_hours", DEFAULT_TTL_HOURS)
                    if (now - ts).total_seconds() > ttl * 3600:
                        f.unlink()
                        cleaned += 1
                    else:
                        remaining += 1
                except Exception:
                    remaining += 1

        return {"cleaned": cleaned, "remaining": remaining}

    def stats(self) -> Dict[str, Any]:
        """全局统计。"""
        now = datetime.now(timezone.utc)
        result = {"boxes": {}, "total_messages": 0, "total_unread": 0}

        for d in sorted(self._root.iterdir()):
            if not d.is_dir():
                continue
            box_name = d.name
            msgs = []
            unread = 0
            for f in d.glob("msg_*.json"):
                try:
                    msg = json.loads(f.read_text())
                    msgs.append(msg)
                    if msg.get("status") == "unread":
                        unread += 1
                except Exception:
                    pass
            result["boxes"][box_name] = {"total": len(msgs), "unread": unread}
            result["total_messages"] += len(msgs)
            result["total_unread"] += unread

        return result

    def broadcast(self, subject: str, body: str, **kwargs) -> Dict[str, Any]:
        """广播到 _global 邮箱。"""
        return self.send("_global", subject=subject, body=body, **kwargs)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="文件邮箱 — Agent 间消息传递")
    sub = parser.add_subparsers(dest="command")

    # send
    p_send = sub.add_parser("send", help="发送消息")
    p_send.add_argument("--to", required=True, help="目标邮箱")
    p_send.add_argument("--subject", required=True, help="主题")
    p_send.add_argument("--body", required=True, help="正文")
    p_send.add_argument("--sender", default="agent", help="发送者")
    p_send.add_argument("--priority", default="normal", choices=["low", "normal", "high"])
    p_send.add_argument("--ref", default=None, help="引用文件路径")

    # list
    p_list = sub.add_parser("list", help="列出消息")
    p_list.add_argument("--box", required=True, help="邮箱名")
    p_list.add_argument("--status", default=None, help="过滤状态")
    p_list.add_argument("--all", action="store_true", help="包含过期")

    # read
    p_read = sub.add_parser("read", help="读取消息")
    p_read.add_argument("--box", required=True, help="邮箱名")
    p_read.add_argument("--id", default=None, help="消息 ID（默认最早未读）")

    # ack
    p_ack = sub.add_parser("ack", help="确认消息")
    p_ack.add_argument("--box", required=True, help="邮箱名")
    p_ack.add_argument("--id", required=True, help="消息 ID")

    # cleanup
    p_clean = sub.add_parser("cleanup", help="清理过期消息")
    p_clean.add_argument("--box", default=None, help="指定邮箱")

    # stats
    sub.add_parser("stats", help="全局统计")

    args = parser.parse_args()
    mb = Mailbox()

    if args.command == "send":
        msg = mb.send(args.to, args.subject, args.body,
                      sender=args.sender, priority=args.priority, ref=args.ref)
        print(f"✅ 已发送 {msg['id']} → {args.to}")

    elif args.command == "list":
        messages = mb.list(args.box, status=args.status,
                          include_expired=args.all)
        if not messages:
            print(f"📭 {args.box}: 空")
        else:
            for m in messages:
                status_icon = {"unread": "🔵", "read": "📖", "acked": "✅"}.get(m["status"], "?")
                print(f"  {status_icon} [{m['id']}] {m['subject']} ({m['from']}) {m['timestamp'][:16]}")

    elif args.command == "read":
        msg = mb.read(args.box, msg_id=args.id)
        if msg:
            print(f"📬 [{msg['id']}] From: {msg['from']}")
            print(f"   主题: {msg['subject']}")
            print(f"   时间: {msg['timestamp']}")
            print(f"   状态: {msg['status']}")
            if msg.get("ref"):
                print(f"   引用: {msg['ref']}")
            print(f"\n{msg['body']}")
        else:
            print("📭 无消息")

    elif args.command == "ack":
        if mb.ack(args.box, args.id):
            print(f"✅ 已确认 {args.id}")
        else:
            print(f"❌ 未找到 {args.id}")

    elif args.command == "cleanup":
        result = mb.cleanup(box=args.box)
        print(f"🧹 清理: {result['cleaned']} 条过期，{result['remaining']} 条保留")

    elif args.command == "stats":
        stats = mb.stats()
        print(f"📊 邮箱统计:")
        for box, info in stats["boxes"].items():
            print(f"   {box}: {info['total']} 条（{info['unread']} 未读）")
        print(f"   总计: {stats['total_messages']} 条，{stats['total_unread']} 未读")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
