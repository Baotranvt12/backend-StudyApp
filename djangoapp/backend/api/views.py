import os
import re
import logging
import traceback
from typing import Dict, List

from django.db import transaction
from django.contrib.auth import authenticate, login, logout
from django.views.decorators.csrf import ensure_csrf_cookie
from django.contrib.auth.models import User

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

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# OpenAI (DeepInfra) client
# ------------------------------------------------------------------------------
DEEPINFRA_API_KEY = os.environ.get("DEEPINFRA_API_KEY")

logger.info(f"DEEPINFRA_API_KEY exists: {bool(DEEPINFRA_API_KEY)}")
if DEEPINFRA_API_KEY:
    logger.info(f"API Key length: {len(DEEPINFRA_API_KEY)}")
    logger.info(f"API Key first 10 chars: {DEEPINFRA_API_KEY[:10]}...")

if not DEEPINFRA_API_KEY:
    logger.error("DEEPINFRA_API_KEY environment variable is not set!")
    if os.environ.get("DJANGO_DEBUG", "false").lower() == "true":
        logger.warning("Using default API key for development only!")
        DEEPINFRA_API_KEY = "your_development_key_here"
    else:
        raise ValueError("DEEPINFRA_API_KEY environment variable is required in production")

try:
    openai = OpenAI(
        api_key=DEEPINFRA_API_KEY,
        base_url="https://api.deepinfra.com/v1/openai",
    )
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    raise

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def normalize_status(value: str) -> str:
    """Chuẩn hoá status bất kỳ về 'pending' hoặc 'done'."""
    if not value:
        return "pending"
    v = value.strip().lower()
    return "done" if v == "done" else "pending"


def build_prompt_vi(class_level: str, subject: str, study_time: str, goal: str) -> List[Dict[str, str]]:
    """
    Tạo prompt tiếng Việt, ép LLM trả về 28 dòng đúng format.
    Mỗi dòng gồm: Tiêu đề, Kỹ năng, Hoạt động (bài tập/project/tài liệu), Tài nguyên, Tự kiểm.
    Ngày 28 bắt buộc có rubric/checklist chi tiết.
    """
    user_content = f"""
Hãy lập kế hoạch tự học 4 tuần (28 ngày) cho học sinh lớp {class_level}, nhằm cải thiện môn {subject}.
Học sinh học {study_time} mỗi ngày. Mục tiêu: {goal}.

YÊU CẦU BẮT BUỘC:
- Theo chương trình Giáo dục phổ thông 2018 (Bộ GD&ĐT), đi từ cơ bản đến nâng cao.
- Xuất RA CHÍNH XÁC 28 DÒNG, tương ứng Ngày 1 → Ngày 28.
- KHÔNG thêm tiêu đề, KHÔNG giải thích, KHÔNG markdown, KHÔNG code block.
- MỖI DÒNG PHẢI THEO ĐÚNG ĐỊNH DẠNG:

Ngày N | Tiêu đề: <tiêu đề ngắn> | Kỹ năng: <kiến thức/kỹ năng cần đạt> | Hoạt động: <bài tập cụ thể / project nhỏ / nguồn tài liệu cần đọc/xem> | Tài nguyên: <sách/trang web/khóa học/công cụ> | Tự kiểm: <cách tự kiểm (Ngày 28: viết RUBRIC/Checklist chi tiết: tiêu chí, trọng số, thang điểm, ngưỡng đạt; Ngày 1–27: viết 1 câu tự kiểm ngắn)>

- Ngày 28 bắt buộc có nội dung "ÔN TẬP & KIỂM TRA TỔNG HỢP" trong Tiêu đề và phần Tự kiểm phải là rubric/checklist chi tiết (tiêu chí, trọng số, thang điểm, ngưỡng đạt).
- Chỉ in đúng 28 dòng theo mẫu trên, không thêm nội dung nào khác.
"""
    return [
        {
            "role": "system",
            "content": (
                "Bạn là chuyên gia lập kế hoạch tự học (10+ năm kinh nghiệm) thiết kế lộ trình cá nhân hoá. "
                "Trả lời HOÀN TOÀN bằng tiếng Việt, tuân thủ chặt chẽ định dạng yêu cầu."
            ),
        },
        {
            "role": "user",
            "content": user_content.strip(),
        },
    ]


# Regex bắt đúng đủ 6 trường, phân tách bằng " | "
LINE_REGEX = re.compile(
    r"^Ngày\s+(?P<day>\d{1,2})\s*\|\s*Tiêu đề:\s*(?P<title>[^|]+)\|\s*Kỹ năng:\s*(?P<skills>[^|]+)\|\s*Hoạt động:\s*(?P<activities>[^|]+)\|\s*Tài nguyên:\s*(?P<resources>[^|]+)\|\s*Tự kiểm:\s*(?P<selfcheck>.+)$",
    re.IGNORECASE
)


def parse_plan_lines(text: str) -> Dict[int, str]:
    """
    Parse 28 dòng theo format cố định. Lưu cả dòng gốc để hiển thị nguyên bản.
    Trả về dict {day_number: full_line}.
    """
    plan: Dict[int, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = LINE_REGEX.match(line)
        if not m:
            # Bỏ các dòng không đúng định dạng
            continue
        try:
            day = int(m.group("day"))
        except Exception:
            continue
        if 1 <= day <= 28:
            plan[day] = line
    return plan


def ensure_day28_has_rubric(full_line: str) -> bool:
    """
    Kiểm tra dòng Ngày 28 có chứa rubric/checklist (dựa theo từ khoá).
    """
    text = full_line.lower()
    keywords = ["rubric", "checklist", "tiêu chí", "trọng số", "thang điểm", "ngưỡng"]
    return any(k in text for k in keywords)


def fallback_plan(subject: str, goal: str) -> Dict[int, str]:
    """
    Kế hoạch dự phòng 28 dòng, đúng format để UI/DB không lỗi.
    """
    plan = {}
    for day in range(1, 29):
        week = (day - 1) // 7 + 1
        title = "Ôn tập & Kiểm tra tổng hợp" if day == 28 else f"Chủ đề tuần {week}"
        selfcheck = (
            "Rubric: Tiêu chí (Hiểu bài 40%, Vận dụng 40%, Trình bày 20%); Thang điểm 10; Ngưỡng đạt ≥7."
            if day == 28
            else "Tự đánh giá bằng câu hỏi nhanh 3–5 phút."
        )
        plan[day] = (
            f"Ngày {day} | Tiêu đề: {title} | "
            f"Kỹ năng: {subject} liên quan đến {goal} | "
            f"Hoạt động: Thực hành 15–30 phút theo mục tiêu ngày; ghi chép lại khó khăn | "
            f"Tài nguyên: SGK/Video chính thống; công cụ Google Tài liệu/Slides | "
            f"Tự kiểm: {selfcheck}"
        )
    return plan


# ------------------------------------------------------------------------------
# API: Generate learning path (store per-user)
# ------------------------------------------------------------------------------
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def generate_learning_path(request):
    """
    Body JSON:
    {
        "class_level": "10",
        "subject": "Tin học",
        "study_time": "1 giờ",
        "goal": "Nắm vững Python cơ bản",
        "model": "anthropic/claude-4-sonnet"   # (tuỳ chọn) override model
    }
    """
    try:
        logger.info(f"generate_learning_path called by user: {request.user.username}")
        logger.info(f"Request data: {request.data}")

        data = request.data
        class_level = (data.get("class_level") or "").strip()
        subject = (data.get("subject") or "").strip()
        study_time = (data.get("study_time") or "").strip()
        goal = (data.get("goal") or "").strip()
        # Mặc định dùng Anthropic Claude 4 Sonnet qua DeepInfra (OpenAI-compatible)
        model = (data.get("model") or "anthropic/claude-4-sonnet").strip()

        if not all([class_level, subject, study_time, goal]):
            logger.warning("Missing required fields")
            return Response(
                {"error": "Thiếu thông tin bắt buộc (class_level, subject, study_time, goal)."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        messages = build_prompt_vi(class_level, subject, study_time, goal)

        # Gọi LLM
        logger.info(f"Calling DeepInfra API with model={model} ...")
        try:
            resp = openai.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                max_tokens=2400,  # đủ cho 28 dòng dài
                temperature=0.5,  # cân bằng tính nhất quán & sáng tạo
            )
            logger.info("DeepInfra API call successful")
        except Exception as api_error:
            logger.error(f"DeepInfra API error: {str(api_error)}")
            error_message = str(api_error)
            if "api_key" in error_message.lower():
                return Response(
                    {"error": "Lỗi xác thực API key. Vui lòng kiểm tra cấu hình."},
                    status=500,
                )
            elif "rate" in error_message.lower():
                return Response(
                    {"error": "Đã vượt quá giới hạn API. Vui lòng thử lại sau."},
                    status=429,
                )
            else:
                return Response(
                    {"error": "Lỗi khi gọi AI service", "details": error_message[:200]},
                    status=500,
                )

        # Parse phản hồi
        try:
            gpt_text_vi = (resp.choices[0].message.content or "").strip()
            logger.info(f"GPT response length: {len(gpt_text_vi)}")
            if not gpt_text_vi:
                logger.error("Empty response from GPT")
                return Response({"error": "Không nhận được phản hồi từ AI"}, status=500)
        except Exception as parse_error:
            logger.error(f"Error parsing GPT response: {parse_error}")
            return Response({"error": "Lỗi xử lý phản hồi từ AI"}, status=500)

        # === Parse 28 dòng ===
        plan = parse_plan_lines(gpt_text_vi)
        logger.info(f"Parsed {len(plan)} days from GPT response")

        # Validate 28 ngày
        if len(plan) != 28:
            logger.warning(f"Expected 28 days but got {len(plan)}. Fallback to default plan.")
            plan = fallback_plan(subject, goal)
        else:
            # Bảo đảm ngày 28 có rubric/checklist
            if 28 not in plan or not ensure_day28_has_rubric(plan[28]):
                logger.warning("Day 28 missing detailed rubric. Injecting safe rubric.")
                line28 = plan.get(28, "")
                if line28:
                    plan[28] = (
                        line28.strip()
                        + " (Rubric: Tiêu chí Hiểu bài 40%, Vận dụng 40%, Trình bày 20%; Thang điểm 10; Ngưỡng đạt ≥7)"
                    )
                else:
                    plan = fallback_plan(subject, goal)

        # Lưu DB (per user & subject)
        user = request.user
        try:
            with transaction.atomic():
                deleted_count = ProgressLog.objects.filter(
                    user=user,
                    subject=subject,
                ).delete()[0]
                logger.info(f"Deleted {deleted_count} old progress logs for subject={subject}")

                objs = []
                for day_number in range(1, 29):
                    task_text = str(plan[day_number])
                    week = (day_number - 1) // 7 + 1
                    objs.append(
                        ProgressLog(
                            user=user,
                            subject=subject,
                            week=week,
                            day_number=day_number,
                            task_title=task_text,
                            status="pending",
                        )
                    )
                created_logs = ProgressLog.objects.bulk_create(objs)
                logger.info(f"Created {len(created_logs)} new progress logs")
        except Exception as db_error:
            logger.error(f"Database error: {db_error}")
            logger.debug(traceback.format_exc())
            return Response({"error": "Lỗi lưu dữ liệu vào database"}, status=500)

        # Trả về
        logs = (
            ProgressLog.objects.filter(user=user, subject=subject)
            .order_by("week", "day_number")
        )
        return Response(
            {
                "message": "✅ Đã tạo lộ trình học 28 ngày!",
                "subject": subject,
                "items": ProgressLogSerializer(logs, many=True).data,
                "raw_gpt_output": gpt_text_vi[:2000],  # cắt ngắn để payload gọn
            },
            status=201,
        )

    except Exception as unexpected_error:
        logger.error(f"Unexpected error in generate_learning_path: {unexpected_error}")
        logger.debug(traceback.format_exc())
        return Response(
            {"error": "Lỗi không xác định", "details": str(unexpected_error)[:200]},
            status=500,
        )


# ------------------------------------------------------------------------------
# API: Get progress list
# ------------------------------------------------------------------------------
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


# ------------------------------------------------------------------------------
# API: Update progress status
# ------------------------------------------------------------------------------
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def update_progress_status(request):
    try:
        log_id = request.data.get("id")
        new_status_raw = request.data.get("status")

        if not log_id or new_status_raw is None:
            return Response({"error": "Thiếu thông tin 'id' hoặc 'status'."}, status=400)

        try:
            log = ProgressLog.objects.get(id=log_id, user=request.user)
        except ProgressLog.DoesNotExist:
            return Response({"error": "Không tìm thấy bản ghi của bạn"}, status=404)

        log.status = normalize_status(new_status_raw)
        log.save(update_fields=["status"])

        return Response(
            {"message": "Cập nhật trạng thái thành công!", "item": ProgressLogSerializer(log).data},
            status=200,
        )
    except Exception as e:
        logger.error(f"Error in update_progress_status: {e}")
        return Response({"error": "Lỗi cập nhật trạng thái"}, status=500)


# ------------------------------------------------------------------------------
# Auth & CSRF endpoints
# ------------------------------------------------------------------------------
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
