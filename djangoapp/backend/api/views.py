import os
import re
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
    # Cho phép dev chạy tạm nếu DJANGO_DEBUG=true
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
# LLM helpers
# =========================
LLM_MODEL = DEEPINFRA_MODEL
DAY_LINE_RE = re.compile(r"^Ngày\s+([1-9]|1\d|2[0-8])\s*:", re.IGNORECASE)

def _llm_call(messages, max_tokens=2000, temperature=0.0, stop=None):
    """
    Gọi DeepInfra (OpenAI-compatible).
    """
    return openai.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        stream=False,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1,
        stop=stop or ["\nNgày 29", "Ngày 29:"],
        # timeout có thể không được tất cả backend hỗ trợ; bỏ nếu không cần
        # timeout=120.0,
    )

def _extract_plan_lines(text: str) -> dict:
    """
    Trích các dòng hợp lệ theo mẫu 'Ngày N: ...' -> dict {day:int -> line:str}
    """
    plan = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = DAY_LINE_RE.match(line)
        if not m:
            continue
        day = int(m.group(1))
        if 1 <= day <= 28:
            plan[day] = line
    return plan

def _missing_days(plan: dict) -> list:
    return [d for d in range(1, 29) if d not in plan]

def _cleanup_tools(plan: dict):
    """
    Chỉ cho phép 'CÔNG CỤ HỖ TRỢ' ở Ngày 1.
    Với ngày 2..28, loại bỏ phần ' | CÔNG CỤ HỖ TRỢ: ...' nếu có.
    """
    for d, line in list(plan.items()):
        if d == 1:
            continue
        plan[d] = re.sub(
            r"\s*\|\s*CÔNG CỤ HỖ TRỢ:.*$",
            "",
            line,
            flags=re.IGNORECASE,
        )

def _enforce_len(plan: dict, limit=90):
    """
    (Tuỳ chọn) Chỉ cảnh báo nếu vượt 90 ký tự; không tự động cắt để tránh mất nghĩa.
    """
    too_long = [d for d, line in plan.items() if len(line) > limit]
    if too_long:
        logger.warning(f"Có {len(too_long)} dòng vượt {limit} ký tự: {too_long}")

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

        # --------- Prompt chặt chẽ + skeleton 28 dòng ---------
        system_msg = (
            "Bạn là chuyên gia lập kế hoạch tự học. Trả lời 100% tiếng Việt. "
            "PHẢI in đúng 28 dòng từ 'Ngày 1:' đến 'Ngày 28:'; mỗi dòng ≤ 90 ký tự; "
            "không tiêu đề/markdown; không dòng trống; mỗi số ngày dùng đúng 1 lần; "
            "không thêm mô tả ngoài 28 dòng."
        )

        skeleton = []
        for d in range(1, 29):
            if d == 1:
                skeleton.append(
                    f"Ngày {d}: <nội dung ngắn> | TỪ KHÓA: <từ khóa> | BT: <gợi ý> | CÔNG CỤ HỖ TRỢ: <app môn {subject}>"
                )
            elif d == 28:
                skeleton.append(
                    "Ngày 28: ÔN & KIỂM TRA TỔNG HỢP - <tóm tắt mục tiêu> | TỪ KHÓA: <từ khóa> | BT: <đề thử>"
                )
            else:
                skeleton.append(
                    f"Ngày {d}: <nội dung ngắn> | TỪ KHÓA: <từ khóa> | BT: <gợi ý>"
                )
        skeleton_text = "\n".join(skeleton)

        user_msg = f"""
Học sinh lớp {class_level}, môn {subject}, thời lượng {study_time}/ngày. Mục tiêu: {goal}.
Bám CT GDPT 2018, tăng dần độ khó, không gộp 2 ngày vào 1.

Hãy THAY THẾ các <...> trong KHUNG sau và GIỮ NGUYÊN tiền tố "Ngày N:".
Nếu thiếu dòng nào, tự bổ sung đến đủ 28 dòng.
Chỉ ghi "CÔNG CỤ HỖ TRỢ" ở Ngày 1; các ngày còn lại KHÔNG có phần này.

{skeleton_text}

Chỉ in đúng 28 dòng ở trên, không thêm nội dung khác.
""".strip()

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ]

        # --------- Lần gọi 1 ---------
        logger.info(f"Calling DeepInfra model={LLM_MODEL} (first pass)…")
        try:
            resp = _llm_call(messages, temperature=0.0)
        except Exception as api_error:
            logger.error(f"DeepInfra API error (pass1): {api_error}")
            err = str(api_error).lower()
            if "api_key" in err:
                return Response(
                    {"error": "Lỗi xác thực API key. Vui lòng kiểm tra cấu hình."},
                    status=500,
                )
            if "rate" in err:
                return Response(
                    {"error": "Đã vượt giới hạn API. Vui lòng thử lại sau."},
                    status=429,
                )
            return Response(
                {"error": "Lỗi khi gọi AI service", "details": str(api_error)[:200]},
                status=500,
            )

        text1 = (resp.choices[0].message.content or "").strip()
        plan = _extract_plan_lines(text1)
        logger.info(f"Pass1 parsed days: {sorted(plan.keys())}")

        # --------- Nếu thiếu ngày, gọi tiếp để in nốt phần còn thiếu ---------
        text2 = ""
        missing = _missing_days(plan)
        if missing:
            logger.warning(f"Missing days after pass1: {missing}")
            cont_lines = []
            cont_prompt = "In TIẾP đúng các dòng còn thiếu (không lặp, không giải thích):\n"
            for d in missing:
                if d == 28:
                    cont_lines.append(
                        "Ngày 28: ÔN & KIỂM TRA TỔNG HỢP - <tóm tắt mục tiêu> | TỪ KHÓA: <từ khóa> | BT: <đề thử>"
                    )
                elif d == 1:
                    cont_lines.append(
                        f"Ngày 1: <nội dung ngắn> | TỪ KHÓA: <từ khóa> | BT: <gợi ý> | CÔNG CỤ HỖ TRỢ: <app môn {subject}>"
                    )
                else:
                    cont_lines.append(
                        f"Ngày {d}: <nội dung ngắn> | TỪ KHÓA: <từ khóa> | BT: <gợi ý>"
                    )
            cont_prompt += "\n".join(cont_lines)

            messages2 = [
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": cont_prompt},
            ]
            logger.info("Calling DeepInfra (continuation)…")
            try:
                resp2 = _llm_call(messages2, max_tokens=800, temperature=0.0)
            except Exception as api_error2:
                logger.error(f"DeepInfra API error (pass2): {api_error2}")
                return Response(
                    {"error": "Lỗi khi gọi AI service (bổ sung)"},
                    status=500,
                )

            text2 = (resp2.choices[0].message.content or "").strip()
            plan2 = _extract_plan_lines(text2)
            plan.update(plan2)
            logger.info(f"After continuation, parsed days: {sorted(plan.keys())}")

        # --------- Hậu kiểm & làm sạch ---------
        _cleanup_tools(plan)       # chỉ tool ở Ngày 1
        _enforce_len(plan, 90)     # cảnh báo nếu > 90 ký tự/dòng

        # Nếu vẫn chưa đủ 28 ngày, fallback mặc định
        if len(plan) != 28:
            logger.warning(f"Expected 28 days but got {len(plan)}. Fallback default plan.")
            plan = {}
            for day in range(1, 29):
                if day == 1:
                    line = (
                        f"Ngày 1: Ôn {subject} cơ bản, đặt mục tiêu {goal} | "
                        f"TỪ KHÓA: {subject} nhập môn | BT: 15' thực hành | "
                        f"CÔNG CỤ HỖ TRỢ: Google Classroom"
                    )
                elif day == 28:
                    line = (
                        f"Ngày 28: ÔN & KIỂM TRA TỔNG HỢP - mục tiêu {goal} | "
                        f"TỪ KHÓA: đề tổng hợp | BT: đề thử"
                    )
                else:
                    line = (
                        f"Ngày {day}: Học {subject} theo mục tiêu {goal} | "
                        f"TỪ KHÓA: {subject} {goal} | BT: 15' thực hành"
                    )
                plan[day] = line

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
                    week = (day - 1) // 7 + 1
                    task_text = plan[day]
                    objs.append(ProgressLog(
                        user=user,
                        subject=subject,
                        week=week,
                        day_number=day,
                        task_title=task_text,
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
                "message": "✅ Đã tạo lộ trình học 28 ngày!",
                "subject": subject,
                "items": ProgressLogSerializer(logs, many=True).data,
                "raw_gpt_output": raw_out.strip()[:1000],
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
