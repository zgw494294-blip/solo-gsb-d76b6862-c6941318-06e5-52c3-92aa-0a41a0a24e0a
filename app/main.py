"""FastAPI 入口：计算接口 + 静态页面托管。"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import PIL.Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import UploadFile

from .core import (
    CalibrationError,
    _check_order_relations,
    _err,
    _parse_full_order,
    _parse_inputs,
    _raise_if_cycle,
    compose_channel_tables,
    confirm_cycle_resolution,
    infer_opacities,
    plan_transition,
    resolve_cycle,
    review_measurement,
    solve,
    suggest_background,
)

app = FastAPI(title="透明图层叠序校准台", version="1.7.0")

# 背景图片合成预览的输入限制（仅接受 RGB 模式 PNG，不转换、不缩放）
PREVIEW_MAX_BYTES = 2 * 1024 * 1024  # 2 MiB
PREVIEW_MAX_SIDE = 512


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/solve")
async def api_solve(request: Request) -> JSONResponse:
    try:
        payload: Any = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"errors": [{"scope": "form", "message": "请求体不是合法的 JSON"}]},
        )
    try:
        result = solve(payload)
    except CalibrationError as exc:
        return JSONResponse(status_code=400, content={"errors": exc.errors})
    return JSONResponse(status_code=200, content=result)


@app.post("/api/infer-opacity")
async def api_infer_opacity(request: Request) -> JSONResponse:
    try:
        payload: Any = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"errors": [{"scope": "form", "message": "请求体不是合法的 JSON"}]},
        )
    try:
        result = infer_opacities(payload)
    except CalibrationError as exc:
        return JSONResponse(status_code=400, content={"errors": exc.errors})
    return JSONResponse(status_code=200, content=result)


@app.post("/api/suggest-background")
async def api_suggest_background(request: Request) -> JSONResponse:
    try:
        payload: Any = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"errors": [{"scope": "form", "message": "请求体不是合法的 JSON"}]},
        )
    try:
        result = suggest_background(payload)
    except CalibrationError as exc:
        return JSONResponse(status_code=400, content={"errors": exc.errors})
    return JSONResponse(status_code=200, content=result)


@app.post("/api/review-measurement")
async def api_review_measurement(request: Request) -> JSONResponse:
    try:
        payload: Any = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"errors": [{"scope": "form", "message": "请求体不是合法的 JSON"}]},
        )
    try:
        result = review_measurement(payload)
    except CalibrationError as exc:
        return JSONResponse(status_code=400, content={"errors": exc.errors})
    return JSONResponse(status_code=200, content=result)


@app.post("/api/plan-transition")
async def api_plan_transition(request: Request) -> JSONResponse:
    try:
        payload: Any = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"errors": [{"scope": "form", "message": "请求体不是合法的 JSON"}]},
        )
    try:
        result = plan_transition(payload)
    except CalibrationError as exc:
        return JSONResponse(status_code=400, content={"errors": exc.errors})
    return JSONResponse(status_code=200, content=result)


@app.post("/api/resolve-cycle")
async def api_resolve_cycle(request: Request) -> JSONResponse:
    try:
        payload: Any = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"errors": [{"scope": "form", "message": "请求体不是合法的 JSON"}]},
        )
    try:
        result = resolve_cycle(payload)
    except CalibrationError as exc:
        return JSONResponse(status_code=400, content={"errors": exc.errors})
    return JSONResponse(status_code=200, content=result)


@app.post("/api/confirm-cycle-resolution")
async def api_confirm_cycle_resolution(request: Request) -> JSONResponse:
    try:
        payload: Any = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"errors": [{"scope": "form", "message": "请求体不是合法的 JSON"}]},
        )
    try:
        result = confirm_cycle_resolution(payload)
    except CalibrationError as exc:
        return JSONResponse(status_code=400, content={"errors": exc.errors})
    return JSONResponse(status_code=200, content=result)


@app.post("/api/compose-preview")
async def api_compose_preview(request: Request) -> Response:
    """把给定底→顶完整叠序应用到上传的背景 PNG 上，返回同尺寸 RGB PNG。

    multipart/form-data：
      - payload：JSON 字符串，含 layers / samples / relations / order；
        order 必须恰好包含全部图层 ID 各一次并满足先后关系（顺序固定，
        服务端不重新择序、不修改任何输入）
      - image：背景图片，PNG、RGB 模式、不超过 2 MiB、宽高均不超过 512 像素

    合成复用 core 的线性光 Porter-Duff "over"（按每通道 256 色阶查表逐像素
    应用），输出取整沿用 HEX 同款规则（clamp × 255 后 round 夹回 0～255），
    不缩放、不重新择序、不改写输入。
    """
    # Starlette 默认单个表单部分上限仅 1 MiB；放宽到略高于业务上限（2 MiB），
    # 让 /api/compose-preview 自己的体积校验成为唯一权威，并给出中文 400
    try:
        try:
            form = await request.form(
                max_part_size=PREVIEW_MAX_BYTES + 64 * 1024,
                max_fields=16,
            )
        except TypeError:
            # 兼容不支持 form() 关键字参数的旧版 Starlette（其默认上限通常更高）
            form = await request.form()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"errors": [_err("preview", "表单解析失败：请确认以 multipart/form-data 提交 payload 与 image，且图片不超过 2 MiB")]},
        )

    # ---- 解析 JSON 校准输入 ------------------------------------------------
    raw_payload = form.get("payload")
    if not isinstance(raw_payload, str) or not raw_payload:
        return JSONResponse(
            status_code=400,
            content={"errors": [_err("preview", "缺少 payload 字段：应为包含 layers / samples / relations / order 的 JSON 字符串")]},
        )
    try:
        payload: Any = json.loads(raw_payload)
    except (json.JSONDecodeError, ValueError):
        return JSONResponse(
            status_code=400,
            content={"errors": [_err("preview", "payload 不是合法的 JSON")]},
        )
    if not isinstance(payload, dict):
        return JSONResponse(
            status_code=400,
            content={"errors": [_err("preview", "payload 必须是 JSON 对象")]},
        )

    errors: list[dict[str, Any]] = []
    try:
        parsed, parse_errors = _parse_inputs(payload)
    except CalibrationError as exc:
        return JSONResponse(status_code=400, content={"errors": exc.errors})
    errors.extend(parse_errors)
    order_idx = _parse_full_order(
        payload.get("order"), parsed, errors, field="order", scope="order"
    )
    if errors:
        return JSONResponse(status_code=400, content={"errors": errors})
    assert order_idx is not None
    try:
        _raise_if_cycle(parsed)
        _check_order_relations(parsed, order_idx, "order", "所给顺序不符合先后关系：")
    except CalibrationError as exc:
        return JSONResponse(status_code=400, content={"errors": exc.errors})

    # ---- 读取并校验图片 ----------------------------------------------------
    image = form.get("image")
    if not isinstance(image, UploadFile):
        return JSONResponse(
            status_code=400,
            content={"errors": [_err("image", "缺少 image 文件字段：请上传一张背景 PNG 图片")]},
        )
    raw_image = image.file.read(PREVIEW_MAX_BYTES + 1)
    if len(raw_image) > PREVIEW_MAX_BYTES:
        return JSONResponse(
            status_code=400,
            content={"errors": [_err("image", "背景图片超过 2 MiB 上限，请压缩或更换图片")]},
        )
    if not raw_image:
        return JSONResponse(
            status_code=400,
            content={"errors": [_err("image", "背景图片为空，请重新选择文件")]},
        )

    try:
        with PIL.Image.open(io.BytesIO(raw_image)) as bg:
            if bg.format != "PNG":
                raise CalibrationError(
                    [_err("image", f"背景图片必须是 PNG 格式，实际识别为 {bg.format}（不做格式转换）")]
                )
            if bg.mode != "RGB":
                raise CalibrationError(
                    [_err("image", f"背景图片必须是 RGB 模式 PNG，实际为 {bg.mode} 模式（不做模式转换）")]
                )
            width, height = bg.size
            if width <= 0 or height <= 0 or width > PREVIEW_MAX_SIDE or height > PREVIEW_MAX_SIDE:
                raise CalibrationError(
                    [
                        _err(
                            "image",
                            f"背景图片宽高必须均为 1～{PREVIEW_MAX_SIDE} 像素，实际为 {width}×{height}",
                        )
                    ]
                )
            bg.load()  # 立即完整解码，截断 / 损坏文件在此处暴露
            tables = compose_channel_tables(order_idx, parsed.lins, parsed.alphas)
            # split 对 16 位 RGB 会先量化到 0～255；逐通道查表后合并回 8 位 RGB，尺寸不变
            bands = bg.split()
            merged = PIL.Image.merge(
                "RGB", [bands[c].point(tables[c]) for c in range(3)]
            )
    except CalibrationError as exc:
        return JSONResponse(status_code=400, content={"errors": exc.errors})
    except (PIL.UnidentifiedImageError, OSError, ValueError) as exc:
        return JSONResponse(
            status_code=400,
            content={"errors": [_err("image", f"背景图片无法解码为有效 PNG：{exc}")]},
        )

    buf = io.BytesIO()
    merged.save(buf, format="PNG")
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={"Content-Disposition": 'inline; filename="composite-preview.png"'},
    )


STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
