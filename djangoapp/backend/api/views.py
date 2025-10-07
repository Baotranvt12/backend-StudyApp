import os
import re
import logging
import unicodedata
from django.db import transaction
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from django.views.decorators.csrf import ensure_csrf_cookie
from openai import OpenAI
from .models import ProgressLog
from .serializers import (
    ProgressLogSerializer,
    RegisterSerializer,
    UserSerializer,
)

# =========================
# Logging setup
# =========================
logger = logging.getLogger(__name__)

# =========================
# OpenAI (DeepInfra) client
# =========================
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


# =========================
# Helpers
# =========================
def normalize_status(value: str) -> str:
    """Normalize arbitrary status strings into 'pending' or 'done'."""
    if not value:
        return "pending"
    v = value.strip().lower()
    return "done" if v == "done" else "pending"


# =========================================
# Generate learning path (store per-user)
# =========================================
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def generate_learning_path(request):
    """
    Gọi Claude tạo lộ trình 4 tuần (28 ngày) và lưu đúng từng dòng.
    """
    try:
        logger.info(f"generate_learning_path called by user: {request.user.username}")
        logger.info(f"Request data: {request.data}")

        data = request.data
        class_level = (data.get("class_level") or "").strip()
        subject = (data.get("subject") or "").strip()
        study_time = (data.get("study_time") or "").strip()
        goal = (data.get("goal") or "").strip()

        if not all([class_level, subject, study_time, goal]):
            return Response({"error": "Thiếu thông tin bắt buộc."}, status=400)

        messages = [
            {
                "role": "system",
                "content": (
                    "Bạn là chuyên gia thiết kế lộ trình học, trả lời HOÀN TOÀN bằng tiếng Việt."
                ),
            },
            {
                "role": "user",
                "content": f"""
Hãy lập kế hoạch tự học 4 tuần (28 ngày) cho học sinh lớp {class_level} để học môn {subject}. 
Thời gian học mỗi ngày: {study_time}, mục tiêu: {goal}.
YÊU CẦU:
- Xuất ra CHÍNH XÁC 28 dòng (Ngày 1 → Ngày 28).
- Mỗi dòng dạng: 
Ngày N: <nội dung> | TỪ KHÓA TÌM KIẾM: <từ khóa> | Bài tập tự luyện: <gợi ý bài tập> | CÔNG CỤ HỖ TRỢ: <ứng dụng/công cụ>.
- Tuyệt đối không thêm tiêu đề, markdown, code block.
""",
            },
        ]

        # =========================
        # Call AI API
        # =========================
        logger.info("Calling DeepInfra Claude API...")
        try:
            resp = openai.chat.completions.create(
                model="anthropic/claude-4-sonnet",
                messages=messages,
                stream=False,
                max_tokens=2500,
                temperature=0.7,
            )
            logger.info("Claude API call successful")
        except Exception as api_error:
            logger.error(f"API Error: {api_error}")
            return Response({"error": f"Lỗi gọi API: {api_error}"}, status=500)

        # Lấy nội dung
        try:
            gpt_text_vi = (resp.choices[0].message.content or "").strip()
            if not gpt_text_vi:
                raise ValueError("Empty response from Claude")
        except Exception as e:
            logger.error(f"Lỗi lấy dữ liệu từ Claude: {e}")
            return Response({"error": "Không nhận được phản hồi từ AI"}, status=500)

        # =========================
        # Parse từng dòng Claude trả về
        # =========================
        plan = {}
        line_regex = re.compile(r"ngày\s*(\d{1,2})\s*[:\-–]\s*(.+)", re.IGNORECASE)

        for raw_line in gpt_text_vi.splitlines():
            # Làm sạch Unicode & các ký tự khoảng trắng lạ
            line = unicodedata.normalize("NFKC", raw_line or "").replace("\u00a0", " ").strip()
            if not line:
                continue

            m = line_regex.search(line)
            if not m:
                continue

            try:
                day_num = int(m.group(1))
                content = m.group(0).strip()
            except Exception:
                continue

            if 1 <= day_num <= 28:
                plan[day_num] = content

        logger.info(f"Parsed {len(plan)} lines from Claude")

        # =========================
        # Fallback chỉ khi Claude thất bại (dưới 14 dòng)
        # =========================
        if len(plan) < 14:
            logger.warning(f"Claude output chỉ có {len(plan)} dòng, dùng fallback mặc định")
            plan = {}
            for day in range(1, 29):
                plan[day] = (
                    f"Ngày {day}: Học {subject} - Ôn tập/chủ đề liên quan {goal} | "
                    f"TỪ KHÓA TÌM KIẾM: {subject} {goal} | "
                    f"Bài tập tự luyện: Thực hành 15 phút | "
                    f"CÔNG CỤ HỖ TRỢ: Google Classroom"
                )
        else:
            logger.info("✅ Claude output hợp lệ, sử dụng nội dung thật")

        # =========================
        # Save vào database
        # =========================
        user = request.user
        try:
            with transaction.atomic():
                ProgressLog.objects.filter(user=user, subject=subject).delete()

                objs = []
                for day_number in sorted(plan.keys()):
                    week = (day_number - 1) // 7 + 1
                    task_text = plan[day_number]

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

                ProgressLog.objects.bulk_create(objs)
        except Exception as db_error:
            logger.error(f"Lỗi database: {db_error}")
            return Response({"error": "Lỗi lưu dữ liệu"}, status=500)

        logs = ProgressLog.objects.filter(user=user, subject=subject).order_by("week", "day_number")

        return Response(
            {
                "message": "✅ Đã tạo lộ trình học!",
                "subject": subject,
                "items": ProgressLogSerializer(logs, many=True).data,
                "raw_gpt_output": gpt_text_vi[:3000],  # trả về bản gốc Claude
            },
            status=201,
        )

    except Exception as unexpected:
        logger.error(f"Unexpected error in generate_learning_path: {unexpected}")
        import traceback
        logger.error(traceback.format_exc())
        return Response({"error": str(unexpected)}, status=500)


# =========================================
# Get progress list
# =========================================
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_progress_list(request):
    try:
        subject = request.query_params.get("subject")
        qs = ProgressLog.objects.filter(user=request.user).order_by("subject", "week", "day_number")
        if subject:
            qs = qs.filter(subject=subject)
        return Response(ProgressLogSerializer(qs, many=True).data, status=200)
    except Exception as e:
        logger.error(f"Error loading progress list: {e}")
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
            return Response({"error": "Thiếu id hoặc trạng thái"}, status=400)

        try:
            log = ProgressLog.objects.get(id=log_id, user=request.user)
        except ProgressLog.DoesNotExist:
            return Response({"error": "Không tìm thấy bản ghi"}, status=404)

        log.status = normalize_status(new_status_raw)
        log.save(update_fields=["status"])
        return Response(
            {
                "message": "✅ Cập nhật thành công",
                "item": ProgressLogSerializer(log).data,
            },
            status=200,
        )
    except Exception as e:
        logger.error(f"update_progress_status error: {e}")
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
