# -*- coding: utf-8 -*-
"""轻量级本地密钥加密工具

用于对 llm_config.json 里的 API Key 之类的敏感凭据做加密落盘，
避免明文出现在项目目录/git 提交里。

实现说明：
- 机器密钥存于 BASE_DIR/.secret_key（被 .gitignore 排除、权限 600）
- 首次调用时自动生成 32 字节随机 key
- 加密算法：HKDF(HMAC-SHA256) 派生流密钥 → XOR → base64
  * 纯标准库（避免引入 cryptography 依赖）
  * 对抗"明文泄露/截图/误提交"等主要威胁，
    不声称抵御具备本地文件读取权限的高级攻击者
- 所有密文统一前缀 "enc:v1:"，解密时按前缀路由
"""

import os
import hmac
import hashlib
import base64
import secrets
import stat

from app.config import BASE_DIR

KEY_PATH = os.path.join(BASE_DIR, ".secret_key")
CIPHER_PREFIX = "enc:v1:"


def _load_or_create_key():
    """读取或首次生成机器密钥。"""
    if os.path.exists(KEY_PATH):
        with open(KEY_PATH, "rb") as fp:
            key = fp.read()
        if len(key) >= 32:
            return key[:32]

    key = secrets.token_bytes(32)
    with open(KEY_PATH, "wb") as fp:
        fp.write(key)
    # 尽力收紧权限（Windows 上此调用会静默跳过不支持的位）
    try:
        os.chmod(KEY_PATH, stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        pass
    return key


def _keystream(master_key, nonce, length):
    """基于 HMAC-SHA256 的简单 KDF/流密钥。"""
    out = bytearray()
    counter = 0
    while len(out) < length:
        block = hmac.new(
            master_key,
            nonce + counter.to_bytes(4, "big"),
            hashlib.sha256,
        ).digest()
        out.extend(block)
        counter += 1
    return bytes(out[:length])


def encrypt(plaintext):
    """加密一段字符串，返回带前缀的密文字符串。

    空字符串/None 原样返回（避免把空值加密成无意义密文）。
    """
    if not plaintext:
        return plaintext
    if isinstance(plaintext, str) and plaintext.startswith(CIPHER_PREFIX):
        return plaintext  # 已加密，幂等

    key = _load_or_create_key()
    nonce = secrets.token_bytes(12)
    data = plaintext.encode("utf-8") if isinstance(plaintext, str) else bytes(plaintext)
    ks = _keystream(key, nonce, len(data))
    ct = bytes(a ^ b for a, b in zip(data, ks))
    # 附带一个 HMAC 防篡改
    tag = hmac.new(key, nonce + ct, hashlib.sha256).digest()[:16]
    blob = base64.urlsafe_b64encode(nonce + tag + ct).decode("ascii")
    return CIPHER_PREFIX + blob


def decrypt(ciphertext):
    """解密；若输入不是密文（向后兼容明文旧配置）直接原样返回。"""
    if not ciphertext or not isinstance(ciphertext, str):
        return ciphertext
    if not ciphertext.startswith(CIPHER_PREFIX):
        return ciphertext  # 明文旧数据，原样返回

    try:
        key = _load_or_create_key()
        blob = base64.urlsafe_b64decode(ciphertext[len(CIPHER_PREFIX):].encode("ascii"))
        if len(blob) < 12 + 16:
            return ""
        nonce = blob[:12]
        tag = blob[12:28]
        ct = blob[28:]
        expected = hmac.new(key, nonce + ct, hashlib.sha256).digest()[:16]
        if not hmac.compare_digest(tag, expected):
            # 密文被篡改或 .secret_key 被轮换过，拒绝解密
            return ""
        ks = _keystream(key, nonce, len(ct))
        pt = bytes(a ^ b for a, b in zip(ct, ks))
        return pt.decode("utf-8", errors="replace")
    except Exception:
        return ""


def is_encrypted(value):
    return isinstance(value, str) and value.startswith(CIPHER_PREFIX)
