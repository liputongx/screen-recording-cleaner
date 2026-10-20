# -*- coding: utf-8 -*-
"""
录屏文档清洗工具 —— 后端
功能：上传视频 -> 抽帧(可调时间倍率) -> 视觉大模型识别 -> 内容清洗 -> 导出 Markdown
"""
import os
import re
import json
import time
import base64
import subprocess
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import imageio_ffmpeg
import requests
from flask import Flask, request, jsonify, send_file, send_from_directory

# ---------------------------------------------------------------------------
# 基础配置
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
FRAME_DIR = os.path.join(BASE_DIR, "frames")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
STATIC_DIR = os.path.join(BASE_DIR, "static")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

for d in (UPLOAD_DIR, FRAME_DIR, OUTPUT_DIR, STATIC_DIR):
    os.makedirs(d, exist_ok=True)

app = Flask(__name__, static_folder=STATIC_DIR)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2GB

# 多厂商视觉模型预设（均走 OpenAI 兼容 /chat/completions 接口）
# "local" 为本地开源 OCR（RapidOCR），免费离线、无需 Key
PROVIDERS = {
    "local": {
        "name": "本地开源 OCR (RapidOCR)",
        "base_url": "",
        "model": "",
    },
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
    },
    "qwen": {
        "name": "阿里通义千问",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-vl-max",
    },
    "zhipu": {
        "name": "智谱 AI",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4v-flash",
    },
    "hunyuan": {
        "name": "腾讯混元",
        "base_url": "https://api.hunyuan.cloud.tencent.com/v1",
        "model": "hunyuan-vision",
    },
    "custom": {
        "name": "自定义 OpenAI 兼容",
        "base_url": "",
        "model": "",
    },
}

# 识别提示词
DEFAULT_RECOGNIZE_PROMPT = (
    "你是专业的文档内容转录助手。请仔细识别这张图片（来自录屏视频的一帧）中的全部文字内容，"
    "并忠实、完整地以 Markdown 格式转录。要求：\n"
    "1) 保持原有的标题层级和段落结构；\n"
    "2) 如果画面中有表格，请用 Markdown 表格输出；\n"
    "3) 不要添加任何解释、补充、总结或评价；\n"
    "4) 如果画面没有文字或无法识别，只回复 [无内容]。"
)

DEFAULT_CLEAN_PROMPT = (
    "以下是从一段录屏视频中逐帧识别出的文字内容，按时间顺序排列，可能包含重复和界面噪声。\n"
    "请你：\n"
    "1) 去除重复内容；\n"
    "2) 去除录屏界面噪声（时间戳、窗口标题、工具栏、进度条等无关文字）；\n"
    "3) 合并整理为一份连贯、结构清晰的 Markdown 文档；\n"
    "4) 保留原文的核心信息和先后顺序，不要编造内容；\n"
    "5) 直接输出整理后的 Markdown，不要任何解释。\n\n"
    "=== 识别内容开始 ===\n{content}\n=== 识别内容结束 ==="
)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def run(cmd):
    """执行命令并返回 (returncode, stdout, stderr)。"""
    p = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    return p.returncode, p.stdout.decode("utf-8", "ignore"), p.stderr.decode("utf-8", "ignore")


def probe_video(path):
    """用 ffmpeg 解析视频时长/分辨率/帧率。"""
    code, _, err = run([FFMPEG, "-i", path])
    info = {"duration": 0.0, "width": 0, "height": 0, "fps": 0.0}
    # 时长
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", err)
    if m:
        h, mi, s = m.groups()
        info["duration"] = int(h) * 3600 + int(mi) * 60 + float(s)
    # 分辨率与帧率
    m = re.search(r"(\d{2,5})x(\d{2,5})", err)
    if m:
        info["width"], info["height"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"(\d+(?:\.\d+)?)\s*fps", err)
    if m:
        info["fps"] = float(m.group(1))
    return info


def frame_path(video_id, idx):
    return os.path.join(FRAME_DIR, video_id, f"frame_{idx:06d}.jpg")


# ---------------------------------------------------------------------------
# 本地 OCR（RapidOCR，懒加载单例）
# ---------------------------------------------------------------------------
_ocr_engine = None
_ocr_lock = threading.Lock()


def get_ocr_engine():
    """懒加载 RapidOCR 引擎（首次调用约 1~2 秒）。"""
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr_engine = RapidOCR()
    return _ocr_engine


def ocr_image(path):
    """本地识别单张图片，返回多行文本。"""
    engine = get_ocr_engine()
    with _ocr_lock:
        result, _ = engine(path)
    if not result:
        return ""
    lines = [item[1] for item in result if item and str(item[1]).strip()]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 页面
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


# ---------------------------------------------------------------------------
# 上传
# ---------------------------------------------------------------------------
@app.route("/api/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "未收到文件"}), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in (".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".m4v"):
        return jsonify({"error": f"不支持的格式 {ext}"}), 400
    video_id = uuid.uuid4().hex[:12]
    save_path = os.path.join(UPLOAD_DIR, video_id + ext)
    f.save(save_path)
    info = probe_video(save_path)
    if info["duration"] <= 0:
        return jsonify({"error": "无法解析视频，请确认文件有效"}), 400
    return jsonify({
        "video_id": video_id,
        "filename": f.filename,
        "duration": round(info["duration"], 2),
        "width": info["width"],
        "height": info["height"],
        "fps": round(info["fps"], 2),
    })


# ---------------------------------------------------------------------------
# 抽帧（核心：可调时间倍率）
# ---------------------------------------------------------------------------
@app.route("/api/extract", methods=["POST"])
def extract():
    data = request.get_json(force=True)
    video_id = data.get("video_id")
    interval = float(data.get("interval", 1.0))   # 抽帧间隔（秒），倍率体现在这里
    start = float(data.get("start", 0.0))          # 起始时间（秒）
    end = float(data.get("end", 0.0))              # 结束时间（秒，0 表示到结尾）
    max_width = int(data.get("max_width", 1600))   # 输出帧最大宽度，控制体积

    if not video_id or interval <= 0:
        return jsonify({"error": "参数错误：缺少 video_id 或 interval 非法"}), 400

    src = None
    for fname in os.listdir(UPLOAD_DIR):
        if fname.startswith(video_id):
            src = os.path.join(UPLOAD_DIR, fname)
            break
    if not src:
        return jsonify({"error": "视频不存在，请重新上传"}), 404

    # 每次抽帧写入独立时间戳子目录，避免删除旧帧（兼容各类沙箱环境）
    subdir = str(int(time.time() * 1000))
    out_dir = os.path.join(FRAME_DIR, video_id, subdir)
    os.makedirs(out_dir, exist_ok=True)

    # 用 fps 滤镜按固定间隔抽帧，同时缩放控制体积
    fps = 1.0 / interval
    vf = f"fps={fps:.6f},scale='min({max_width},iw)':-2"
    cmd = [FFMPEG, "-y"]
    if start > 0:
        cmd += ["-ss", str(start)]
    cmd += ["-i", src]
    if end > 0:
        cmd += ["-t", str(end - start)]
    cmd += ["-vf", vf, "-q:v", "2", os.path.join(out_dir, "frame_%06d.jpg")]

    code, _, err = run(cmd)
    if code != 0:
        return jsonify({"error": "抽帧失败：" + err[-500:]}), 500

    frames = sorted(os.listdir(out_dir))
    total = len(frames)
    return jsonify({
        "total": total,
        "interval": interval,
        "subdir": subdir,
        "url_prefix": f"/api/frame/{video_id}/{subdir}/",
        "names": frames,
    })


@app.route("/api/frame/<video_id>/<subdir>/<name>")
def get_frame(video_id, subdir, name):
    return send_from_directory(os.path.join(FRAME_DIR, video_id, subdir), name)


# ---------------------------------------------------------------------------
# 识别（多厂商视觉大模型）
# ---------------------------------------------------------------------------
def call_vision(provider_cfg, api_key, prompt, image_path):
    """调用 OpenAI 兼容视觉接口识别单张图片。"""
    with open(image_path, "rb") as fp:
        b64 = base64.b64encode(fp.read()).decode()
    ext = os.path.splitext(image_path)[1].lstrip(".") or "jpg"
    data_url = f"data:image/{ext};base64,{b64}"

    body = {
        "model": provider_cfg["model"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "temperature": 0.1,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = provider_cfg["base_url"].rstrip("/") + "/chat/completions"
    r = requests.post(url, json=body, headers=headers, timeout=120)
    if r.status_code != 200:
        msg = f"API {r.status_code}: {r.text[:300]}"
        # 识别模型不支持图片输入时的友好提示
        low = r.text.lower()
        if any(k in low for k in ("image", "vision", "multimodal", "content_type", "not support", "不支持")):
            msg += " ｜ 该模型可能不支持图片输入，请切换供应商为通义千问/智谱/混元等视觉模型。"
        raise RuntimeError(msg)
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise RuntimeError(f"API 返回格式异常: {json.dumps(data, ensure_ascii=False)[:300]}")


@app.route("/api/recognize", methods=["POST"])
def recognize():
    data = request.get_json(force=True)
    video_id = data.get("video_id")
    subdir = data.get("subdir", "")
    provider_key = data.get("provider", "local")
    api_key = data.get("api_key", "").strip()
    base_url = data.get("base_url", "").strip()
    model = data.get("model", "").strip()
    prompt = data.get("prompt", "") or DEFAULT_RECOGNIZE_PROMPT
    concurrency = int(data.get("concurrency", 3))
    indices = data.get("indices")  # 可选：只识别指定帧序号（1-based），为空则全部

    if not video_id:
        return jsonify({"error": "缺少 video_id"}), 400

    is_local = provider_key == "local"

    if not is_local:
        if provider_key not in PROVIDERS:
            return jsonify({"error": "未知供应商"}), 400
        if not api_key:
            return jsonify({"error": "请填写 API Key"}), 400
        provider_cfg = {
            "base_url": base_url or PROVIDERS[provider_key]["base_url"],
            "model": model or PROVIDERS[provider_key]["model"],
        }
        if not provider_cfg["base_url"] or not provider_cfg["model"]:
            return jsonify({"error": "请填写 Base URL 和模型名"}), 400

    fdir = os.path.join(FRAME_DIR, video_id, subdir)
    if not os.path.isdir(fdir):
        return jsonify({"error": "请先抽帧"}), 400
    names = sorted(os.listdir(fdir))
    if indices:
        names = [n for n in names if int(n.split("_")[1].split(".")[0]) in set(indices)]

    def worker(name):
        idx = int(name.split("_")[1].split(".")[0])
        path = os.path.join(fdir, name)
        if is_local:
            text = ocr_image(path)
        else:
            text = call_vision(provider_cfg, api_key, prompt, path)
        return idx, text

    results = {}
    errors = []
    if is_local:
        # 本地 OCR 串行处理，避免多线程争用引擎
        for name in names:
            try:
                idx, text = worker(name)
                results[idx] = text
            except Exception as e:
                errors.append(f"帧 {name}: {str(e)}")
    else:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = {pool.submit(worker, n): n for n in names}
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    idx, text = fut.result()
                    results[idx] = text
                except Exception as e:
                    errors.append(f"帧 {name}: {str(e)}")

    # 按帧序号排序输出
    ordered = [{"index": k, "text": results[k]} for k in sorted(results)]
    return jsonify({
        "total": len(names),
        "success": len(results),
        "errors": errors,
        "results": ordered,
    })


# ---------------------------------------------------------------------------
# 内容清洗（内置规则 + 可选 LLM 整合）
# ---------------------------------------------------------------------------
def builtin_clean(results):
    """内置清洗：去空、去连续重复、去常见界面噪声。"""
    lines = []
    prev = None
    noise_patterns = [
        r"^\d{1,2}:\d{2}(:\d{2})?\s*$",       # 时间戳
        r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}",        # 日期行
    ]
    for item in results:
        text = (item.get("text") or "").strip()
        if not text or text == "[无内容]":
            continue
        # 去连续重复（相邻帧内容一致）
        if text == prev:
            continue
        prev = text
        for ln in text.splitlines():
            s = ln.strip()
            if not s:
                continue
            if any(re.match(p, s) for p in noise_patterns):
                continue
            lines.append(s)
    # 二次去重（保留顺序）
    seen = set()
    dedup = []
    for ln in lines:
        if ln in seen:
            continue
        seen.add(ln)
        dedup.append(ln)
    return "\n".join(dedup)


@app.route("/api/clean", methods=["POST"])
def clean():
    data = request.get_json(force=True)
    results = data.get("results", [])
    use_llm = bool(data.get("use_llm", False))
    provider_key = data.get("provider", "deepseek")
    api_key = data.get("api_key", "").strip()
    base_url = data.get("base_url", "").strip()
    model = data.get("model", "").strip()
    clean_prompt = data.get("clean_prompt", "") or DEFAULT_CLEAN_PROMPT

    if not results:
        return jsonify({"error": "没有可清洗的内容"}), 400

    cleaned = builtin_clean(results)
    if not use_llm:
        return jsonify({"cleaned": cleaned})

    if not api_key:
        return jsonify({"error": "使用 LLM 清洗需要 API Key"}), 400
    provider_cfg = {
        "base_url": base_url or PROVIDERS[provider_key]["base_url"],
        "model": model or PROVIDERS[provider_key]["model"],
    }
    if not provider_cfg["base_url"] or not provider_cfg["model"]:
        return jsonify({"error": "请填写 Base URL 和模型名"}), 400

    prompt = clean_prompt.format(content=cleaned)
    body = {
        "model": provider_cfg["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = provider_cfg["base_url"].rstrip("/") + "/chat/completions"
    r = requests.post(url, json=body, headers=headers, timeout=180)
    if r.status_code != 200:
        return jsonify({"error": f"清洗 API {r.status_code}: {r.text[:300]}"}), 500
    try:
        final = r.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return jsonify({"error": "清洗 API 返回格式异常"}), 500
    return jsonify({"cleaned": final})


# ---------------------------------------------------------------------------
# 导出
# ---------------------------------------------------------------------------
@app.route("/api/export", methods=["POST"])
def export():
    data = request.get_json(force=True)
    content = data.get("content", "")
    filename = data.get("filename", "document")
    if not content:
        return jsonify({"error": "内容为空"}), 400
    safe = re.sub(r"[^\w\u4e00-\u9fff\-]+", "_", filename).strip("_") or "document"
    safe = safe[:80]
    out = os.path.join(OUTPUT_DIR, safe + ".md")
    with open(out, "w", encoding="utf-8") as fp:
        fp.write(content)
    return jsonify({"path": out, "filename": safe + ".md"})


@app.route("/api/download/<path:name>")
def download(name):
    return send_file(os.path.join(OUTPUT_DIR, name), as_attachment=True)


if __name__ == "__main__":
    print("=" * 50)
    print("录屏文档清洗工具已启动")
    print("请在浏览器打开: http://127.0.0.1:8765")
    print("=" * 50)
    app.run(host="127.0.0.1", port=8765, debug=False, threaded=True)
