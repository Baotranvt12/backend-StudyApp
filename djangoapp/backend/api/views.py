import os
import re
import json
import logging

from django.db import transaction
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.views.decorators.csrf import ensure_csrf_cookie

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework import status

from openai import OpenAI

from .models import ProgressLog
from .serializers import (
    ProgressLogSerializer,
    RegisterSerializer,
    UserSerializer,
)

# =========================
# Logging
# =========================
logger = logging.getLogger(__name__)

# =========================
# OpenAI (DeepInfra) client
# =========================
DEEPINFRA_API_KEY = os.environ.get("DEEPINFRA_API_KEY")
DEEPINFRA_MODEL = os.environ.get("DEEPINFRA_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")

if not DEEPINFRA_API_KEY:
    if os.environ.get("DJANGO_DEBUG", "false").lower() == "true":
        logger.warning("DEEPINFRA_API_KEY is missing. Using a placeholder for development only!")
        DEEPINFRA_API_KEY = "your_development_key_here"
    else:
        raise ValueError("DEEPINFRA_API_KEY environment variable is required in production")

try:
    openai = OpenAI(
        api_key=DEEPINFRA_API_KEY,
        base_url="https://api.deepinfra.com/v1/openai",
    )
    logger.info("OpenAI client initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    raise

# =========================
# Helpers
# =========================
def normalize_status(value: str) -> str:
    """Normalize arbitrary status strings into 'pending' or 'done'."""
    if not value:
        return "pending"
    v = str(value).strip().lower()
    return "done" if v == "done" else "pending"

# =========================
# LLM helpers (JSON Lines)
# =========================
LLM_MODEL = DEEPINFRA_MODEL
DOMAIN_TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9\.\-/]+")
FALLBACK_DOMAIN = "khanacademy.org"  # fallback chung khi thiếu/không hợp lệ

def _llm_call(messages, max_tokens=2000, temperature=0.0, stop=None):
    return openai.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        stream=False,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1,
        stop=stop or [],  # JSONL không cần stop "Ngày 29"
    )

def _normalize_domain_token(token: str) -> str:
    """
    Chuẩn hoá domain/path: bỏ http/https, bỏ www., loại bỏ khoảng trắng, lấy phần gọn.
    """
    x = (token or "").strip()
    x = x.replace("http://", "").replace("https://", "")
    x = x.replace("www.", "")
    x = x.split()[0]  # lấy token đầu nếu ai đó chèn khoảng trắng
    m = DOMAIN_TOKEN_RE.search(x)
    return m.group(0) if m else ""

def _ensure_keyword_domain(obj: dict):
    """
    Đảm bảo obj['keyword'] là domain ngắn. Nếu thiếu/không hợp lệ → dùng FALLBACK_DOMAIN.
    """
    raw_site = _normalize_domain_token(obj.get("keyword") or "")
    # loại trường hợp quá dài/lạc đề bằng một check nhẹ
    if not raw_site or len(raw_site) > 60 or "/" in raw_site and len(raw_site) > 40:
        obj["keyword"] = FALLBACK_DOMAIN
    else:
        obj["keyword"] = raw_site

def _ensure_tool_only_day1(obj: dict, day: int, subject: str):
    """Chỉ Ngày 1 có tool. Các ngày khác set tool=None."""
    if day == 1:
        t = (obj.get("tool") or "").strip()
        if not t:
            obj["tool"] = f"Google Classroom (môn {subject})"
    else:
        obj["tool"] = None

def _shorten(text: str, limit: int = 60) -> str:
    """Rút gọn chuỗi để giữ tổng JSON line ngắn (ưu tiên content/exercise)."""
    t = (text or "").strip()
    return t if len(t) <= limit else (t[:limit - 1] + "…")

def _make_task_title(day: int, obj: dict) -> str:
    """Ghép thành 1 dòng text để lưu vào task_title (ngắn gọn)."""
    parts = [f"Ngày {day}", f"Nội dung: {obj.get('content','').strip()}"]
    if obj.get("keyword"):
        parts.append(f"Web: {obj['keyword']}")
    if obj.get("exercise"):
        parts.append(f"BT: {obj['exercise']}")
    if day == 1 and obj.get("tool"):
        parts.append(f"Tool: {obj['tool']}")
    title = " | ".join(parts)
    return title if len(title) <= 120 else (title[:119] + "…")

# =========================================
# Generate learning path (store per-user)
# =========================================
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def generate_learning_path(request):
    """
    Body JSON:
    {
      "class_level": "10",
      "subject": "Tin học",
      "study_time": "1 giờ",
      "goal": "Nắm vững Python cơ bản"
    }
    """
    try:
        logger.info(f"generate_learning_path called by user: {request.user.username}")
        logger.info(f"Request data: {request.data}")

        data = request.data
        class_level = (data.get("class_level") or "").strip()
        subject     = (data.get("subject") or "").strip()
        study_time  = (data.get("study_time") or "").strip()
        goal        = (data.get("goal") or "").strip()

        if not all([class_level, subject, study_time, goal]):
            return Response(
                {"error": "Thiếu thông tin bắt buộc."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # --------- Prompt JSONL (không whitelist) ---------
        # Nói rõ keyword phải là domain ngắn, liên quan môn (mô tả), ưu tiên học liệu; cho ví dụ—không ràng buộc
        examples = "ví dụ: khanacademy.org, vietjack.com, cppreference.com, britannica.com"
        system_msg = (
            "Bạn là chuyên gia lập kế hoạch tự học. Trả lời 100% tiếng Việt.\n"
            "YÊU CẦU ĐẦU RA (BẮT BUỘC):\n"
            "• In CHÍNH XÁC 28 dòng JSON (mỗi dòng 1 object, không mảng, không giải thích).\n"
            "• Mỗi object có khóa: day (int 1..28), content (string ≤ 60 ký tự), "
            "keyword (string – TÊN MIỀN trang web ngắn), exercise (string ≤ 60 ký tự), tool (string | null).\n"
            "• Chỉ Ngày 1 có tool (ghi tên ứng dụng/công cụ), ngày 2..28 để tool = null hoặc bỏ field.\n"
            "• keyword PHẢI là TÊN MIỀN NGẮN (không http/https/www), liên quan trực tiếp tới môn học, "
            f"ưu tiên trang học liệu uy tín ({examples}).\n"
            "• Không in tiêu đề/markdown, không dòng trống, không text thừa ngoài 28 dòng JSON.\n"
        )

        # Skeleton JSON Lines (28 dòng)
        sk_lines = []
        for d in range(1, 29):
            if d == 1:
                sk_lines.append(
                    '{"day": 1, "content": "<nội dung>", "keyword": "<domain>", '
                    f'"exercise": "<bài tập>", "tool": "<app môn {subject}>"}'
                )
            elif d == 28:
                sk_lines.append(
                    '{"day": 28, "content": "Ôn & kiểm tra tổng hợp - <tóm tắt>", '
                    '"keyword": "<domain>", "exercise": "<đề thử>"}'
                )
            else:
                sk_lines.append(
                    f'{{"day": {d}, "content": "<nội dung>", "keyword": "<domain>", '
                    '"exercise": "<bài tập>"}}'
                )
        skeleton_text = "\n".join(sk_lines)

        user_msg = f"""
Học sinh lớp {class_level}, môn {subject}, học {study_time}/ngày. Mục tiêu: {goal}.
Bám CT GDPT 2018, tăng dần độ khó, không gộp 2 ngày vào 1.

Hãy THAY THẾ các <...> trong KHUNG JSONLINES sau. 
Ràng buộc:
- keyword: TÊN MIỀN NGẮN, liên quan tới môn {subject}, không http/https/www (vd: khanacademy.org).
- tool: chỉ Ngày 1 có; các ngày khác để null hoặc bỏ field.
- Ưu tiên rút gọn content & exercise để mỗi dòng ngắn gọn (~≤ 90 ký tự tổng thể).

KHUNG (28 dòng, từ day=1 đến day=28):
{skeleton_text}

Chỉ in đúng 28 dòng JSON như KHUNG, không in gì khác.
""".strip()

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ]

        # --------- Gọi LLM lần 1 ---------
        logger.info(f"Calling DeepInfra model={LLM_MODEL} (first pass)…")
        try:
            resp = _llm_call(messages, temperature=0.0)
        except Exception as api_error:
            logger.error(f"DeepInfra API error (pass1): {api_error}")
            err = str(api_error).lower()
            if "api_key" in err:
                return Response({"error": "Lỗi xác thực API key. Vui lòng kiểm tra cấu hình."}, status=500)
            if "rate" in err:
                return Response({"error": "Đã vượt giới hạn API. Vui lòng thử lại sau."}, status=429)
            return Response({"error": "Lỗi khi gọi AI service", "details": str(api_error)[:200]}, status=500)

        text1 = (resp.choices[0].message.content or "").strip()
        lines1 = [ln for ln in text1.splitlines() if ln.strip()]
        logger.info(f"First pass lines: {len(lines1)}")

        # Parse JSON Lines → dict day->obj
        plan_objs = {}
        for ln in lines1:
            try:
                obj = json.loads(ln)
                d = int(obj.get("day", 0))
                if 1 <= d <= 28:
                    plan_objs[d] = obj
            except Exception:
                continue

        # --------- Nếu thiếu, yêu cầu in tiếp các day còn thiếu dưới dạng JSONL ---------
        missing = [d for d in range(1, 29) if d not in plan_objs]
        text2 = ""
        if missing:
            logger.warning(f"Missing days after pass1: {missing}")
            sk2 = []
            for d in missing:
                if d == 1:
                    sk2.append(
                        '{"day": 1, "content": "<nội dung>", "keyword": "<domain>", '
                        f'"exercise": "<bài tập>", "tool": "<app môn {subject}>"}'
                    )
                elif d == 28:
                    sk2.append(
                        '{"day": 28, "content": "Ôn & kiểm tra tổng hợp - <tóm tắt>", '
                        '"keyword": "<domain>", "exercise": "<đề thử>"}'
                    )
                else:
                    sk2.append(
                        f'{{"day": {d}, "content": "<nội dung>", "keyword": "<domain>", '
                        '"exercise": "<bài tập>"}}'
                    )
            cont_prompt = (
                "In TIẾP đúng các dòng JSON còn thiếu, MỖI DÒNG 1 OBJECT, không giải thích, không mảng:\n" +
                "\n".join(sk2)
            )
            messages2 = [
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": cont_prompt},
            ]
            logger.info("Calling DeepInfra (continuation)…")
            try:
                resp2 = _llm_call(messages2, max_tokens=1200, temperature=0.0)
            except Exception as api_error2:
                logger.error(f"DeepInfra API error (pass2): {api_error2}")
                return Response({"error": "Lỗi khi gọi AI service (bổ sung)"}, status=500)

            text2 = (resp2.choices[0].message.content or "").strip()
            lines2 = [ln for ln in text2.splitlines() if ln.strip()]
            for ln in lines2:
                try:
                    obj = json.loads(ln)
                    d = int(obj.get("day", 0))
                    if 1 <= d <= 28:
                        plan_objs[d] = obj
                except Exception:
                    continue

        # --------- Hậu kiểm & chuẩn hoá từng object ---------
        for d in range(1, 29):
            if d not in plan_objs:
                # tạo object tối thiểu nếu vẫn thiếu
                plan_objs[d] = {
                    "day": d,
                    "content": f"Học {subject} theo mục tiêu {goal}",
                    "keyword": FALLBACK_DOMAIN,
                    "exercise": "15' thực hành",
                    "tool": f"Google Classroom (môn {subject})" if d == 1 else None
                }

            # rút gọn content/exercise để ngắn gọn
            plan_objs[d]["content"]  = _shorten(plan_objs[d].get("content", ""), 60)
            plan_objs[d]["exercise"] = _shorten(plan_objs[d].get("exercise", ""), 60)

            # chuẩn hoá domain keyword (không dùng whitelist)
            _ensure_keyword_domain(plan_objs[d])
            # chỉ ngày 1 có tool
            _ensure_tool_only_day1(plan_objs[d], d, subject)

        # --------- Lưu DB ---------
        user = request.user
        try:
            with transaction.atomic():
                deleted = ProgressLog.objects.filter(
                    user=user,
                    subject=subject
                ).delete()[0]
                logger.info(f"Deleted {deleted} old progress logs")

                objs = []
                for day in range(1, 29):
                    title = _make_task_title(day, plan_objs[day])
                    week = (day - 1) // 7 + 1
                    objs.append(ProgressLog(
                        user=user,
                        subject=subject,
                        week=week,
                        day_number=day,
                        task_title=title,   # gộp ngắn gọn; tùy bạn mở rộng model để lưu 4 field
                        status="pending",
                    ))
                ProgressLog.objects.bulk_create(objs)
                logger.info(f"Created {len(objs)} progress logs")
        except Exception as db_error:
            logger.error(f"Database error: {db_error}")
            return Response({"error": "Lỗi lưu dữ liệu vào database"}, status=500)

        logs = ProgressLog.objects.filter(
            user=user,
            subject=subject
        ).order_by("week", "day_number")

        raw_out = (text1 or "") + ("\n" + text2 if text2 else "")
        return Response(
            {
                "message": "✅ Đã tạo lộ trình học 28 ngày (JSONL, không whitelist)!",
                "subject": subject,
                "items": ProgressLogSerializer(logs, many=True).data,
                "raw_gpt_output": raw_out.strip()[:1200],
                "model": LLM_MODEL,
            },
            status=201,
        )

    except Exception as unexpected_error:
        logger.error(f"Unexpected error in generate_learning_path: {unexpected_error}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return Response(
            {"error": "Lỗi không xác định", "details": str(unexpected_error)[:200]},
            status=500,
        )

# =========================================
# Get progress list
# =========================================
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_progress_list(request):
    try:
        subject = request.query_params.get("subject")
        user = request.user
        qs = ProgressLog.objects.filter(user=user).order_by("subject", "week", "day_number")
        if subject:
            qs = qs.filter(subject=subject)
        return Response(ProgressLogSerializer(qs, many=True).data, status=200)
    except Exception as e:
        logger.error(f"Error in get_progress_list: {e}")
        return Response({"error": "Lỗi lấy danh sách tiến độ"}, status=500)

# =========================================
# Update progress status
# =========================================
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def update_progress_status(request):
    try:
        log_id = request.data.get("id")
        new_status_raw = request.data.get("status")

        if not log_id or new_status_raw is None:
            return Response(
                {"error": "Thiếu thông tin 'id' hoặc 'status'."},
                status=400,
            )

        try:
            log = ProgressLog.objects.get(id=log_id, user=request.user)
        except ProgressLog.DoesNotExist:
            return Response(
                {"error": "Không tìm thấy bản ghi của bạn"},
                status=404,
            )

        log.status = normalize_status(new_status_raw)
        log.save(update_fields=["status"])

        return Response(
            {"message": "Cập nhật trạng thái thành công!", "item": ProgressLogSerializer(log).data},
            status=200,
        )
    except Exception as e:
        logger.error(f"Error in update_progress_status: {e}")
        return Response({"error": "Lỗi cập nhật trạng thái"}, status=500)

# =========================================
# Auth & CSRF endpoints
# =========================================
@api_view(["GET"])
@ensure_csrf_cookie
@permission_classes([AllowAny])
def get_csrf(request):
    return Response({"csrfToken": request.META.get("CSRF_COOKIE", "")})

@api_view(["POST"])
@permission_classes([AllowAny])
def register(request):
    serializer = RegisterSerializer(data=request.data)
    if serializer.is_valid():
        user = serializer.save()
        return Response(
            {"message": "Đăng ký thành công!", "user": UserSerializer(user).data},
            status=201,
        )
    return Response(serializer.errors, status=400)

@api_view(["POST"])
@permission_classes([AllowAny])
def login_view(request):
    username = request.data.get("username")
    password = request.data.get("password")
    user = authenticate(request, username=username, password=password)

    if not user:
        return Response({"detail": "Sai tên đăng nhập hoặc mật khẩu."}, status=400)

    login(request, user)
    return Response({"message": "Đăng nhập thành công!", "username": user.username})

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def logout_view(request):
    logout(request)
    return Response({"message": "Đã đăng xuất."})

@api_view(["GET"])
@permission_classes([AllowAny])
def whoami(request):
    if request.user.is_authenticated:
        return Response({"username": request.user.username})
    return Response({"username": None}, status=200)
