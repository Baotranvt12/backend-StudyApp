import os
import re
import logging
from functools import lru_cache

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
# Logging & helpers
# =========================
logger = logging.getLogger(__name__)

def _env_true(v, default=False):
    if v is None:
        return default
    return str(v).lower() in ("1", "true", "yes", "y", "on")

IS_DEBUG = _env_true(os.environ.get("DJANGO_DEBUG"), default=False)


# =========================
# OpenAI (DeepInfra) client - LAZY INIT
# =========================
@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """
    Khởi tạo client khi thực sự cần.
    - Prod: yêu cầu DEEPINFRA_API_KEY
    - Dev: cho phép dùng DEEPINFRA_DEV_KEY (nếu không có thì raise)
    """
    api_key = os.environ.get("DEEPINFRA_API_KEY")
    if not api_key:
        if IS_DEBUG:
            api_key = os.environ.get("DEEPINFRA_DEV_KEY")
            if not api_key:
                raise RuntimeError(
                    "DEV mode: missing DEEPINFRA_DEV_KEY for DeepInfra."
                )
            logger.warning("Using DEV DeepInfra key (DEV only).")
        else:
            raise RuntimeError("DEEPINFRA_API_KEY is missing in production.")

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepinfra.com/v1/openai",
        )
        return client
    except Exception:
        logger.exception("Failed to initialize OpenAI (DeepInfra) client")
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
    Body JSON:
    {
      "class_level": "10",
      "subject": "Tin học",
      "study_time": "1 giờ",
      "goal": "Nắm vững Python cơ bản"
    }
    """
    try:
        logger.info("generate_learning_path called by %s", request.user.username)
        logger.debug("Request data: %s", request.data)

        data = request.data
        class_level = (data.get("class_level") or "").strip()
        subject = (data.get("subject") or "").strip()
        study_time = (data.get("study_time") or "").strip()
        goal = (data.get("goal") or "").strip()

        if not all([class_level, subject, study_time, goal]):
            return Response(
                {"error": "Thiếu thông tin bắt buộc."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "Bạn là chuyên gia lập kế hoạch tự học, có hơn 10 năm kinh nghiệm "
                    "thiết kế chương trình học tập cá nhân hoá. Trả lời HOÀN TOÀN bằng tiếng Việt."
                ),
            },
            {
                "role": "user",
                "content": f"""
Hãy lập kế hoạch tự học 4 tuần (28 ngày) cho học sinh lớp {class_level}, nhằm cải thiện môn {subject}. 
Học sinh học {study_time} mỗi ngày. Mục tiêu: {goal}.
YÊU CẦU:
- Xuất ra CHÍNH XÁC 28 dòng (tương ứng Ngày 1 → Ngày 28). 
- Mỗi dòng là một hoạt động học ngắn gọn, theo tiến trình từ cơ bản đến nâng cao. 
- Nội dung theo chương trình Giáo dục phổ thông 2018 của Bộ Giáo dục và Đào tạo. 
- Ngày 28 phải là phần ÔN TẬP & KIỂM TRA TỔNG HỢP. 
- KHÔNG in thêm tiêu đề, KHÔNG giải thích, KHÔNG markdown, KHÔNG code block.
Định dạng MỖI DÒNG:
Ngày N: <nội dung> | TỪ KHÓA TÌM KIẾM: <từ khóa> | Bài tập tự luyện: <gợi ý bài tập ứng dụng thực tế> | CÔNG CỤ HỖ TRỢ: <ứng dụng/công cụ số học tập liên quan đến môn {subject}>
Chỉ in ra đúng 28 dòng theo mẫu trên, không thêm nội dung nào khác.
""",
            },
        ]

        # --- Call DeepInfra safely ---
        try:
            client = get_openai_client()
            resp = client.chat.completions.create(
                model="openchat/openchat_3.5",  # confirm model/quota trên DeepInfra
                messages=messages,
                stream=False,
                max_tokens=2000,
                temperature=0.7,
                timeout=60,  # tránh treo httpx
            )
        except Exception as api_error:
            msg = str(api_error)
            logger.exception("DeepInfra API error: %s", msg)

            lower = msg.lower()
            if "401" in lower or "unauthorized" in lower or "api key" in lower:
                return Response(
                    {"error": "Lỗi xác thực API key. Vui lòng kiểm tra cấu hình."},
                    status=status.HTTP_401_UNAUTHORIZED,
                )
            if "429" in lower or "rate" in lower or "quota" in lower:
                return Response(
                    {"error": "Đã vượt quá giới hạn API. Vui lòng thử lại sau."},
                    status=status.HTTP_429_TOO_MANY_REQUESTS,
                )
            # lỗi kết nối/service
            return Response(
                {"error": "Lỗi khi gọi AI service", "details": msg[:200]},
                status=status.HTTP_502_BAD_GATEWAY,
            )

        # --- Parse response ---
        try:
            gpt_text_vi = (resp.choices[0].message.content or "").strip()
            if not gpt_text_vi:
                logger.error("Empty response from LLM")
                return Response(
                    {"error": "Không nhận được phản hồi từ AI"},
                    status=status.HTTP_502_BAD_GATEWAY,
                )
        except Exception:
            logger.exception("Error parsing LLM response")
            return Response(
                {"error": "Lỗi xử lý phản hồi từ AI"},
                status=status.HTTP_502_BAD_GATEWAY,
            )

        # --- Parse 28 lines "Ngày N: ..." ---
        line_regex = re.compile(r"^Ngày\s+(\d{1,2})\s*[:\-\–].+$", re.IGNORECASE)
        plan = {}

        for raw_line in gpt_text_vi.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            m = line_regex.match(line)
            if not m:
                continue
            try:
                day_num = int(m.group(1))
            except Exception:
                continue
            if 1 <= day_num <= 28:
                plan[day_num] = line

        logger.info("Parsed %d day lines", len(plan))

        # Fallback nếu không đủ 28 dòng
        if len(plan) != 28:
            logger.warning("Expected 28 lines, got %d. Fallback default plan.", len(plan))
            plan = {}
            for day in range(1, 29):
                week = (day - 1) // 7 + 1
                plan[day] = (
                    f"Ngày {day}: Học {subject} - Ôn tập/chủ đề liên quan {goal} | "
                    f"TỪ KHÓA TÌM KIẾM: {subject} {goal} | "
                    f"Bài tập tự luyện: Thực hành 15 phút | "
                    f"CÔNG CỤ HỖ TRỢ: Google Classroom"
                )

        # --- Save DB ---
        user = request.user
        try:
            with transaction.atomic():
                deleted_count = ProgressLog.objects.filter(
                    user=user, subject=subject
                ).delete()[0]
                logger.info("Deleted %d old ProgressLog rows", deleted_count)

                objs = []
                for day_number in sorted(plan.keys()):
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
                ProgressLog.objects.bulk_create(objs)
                logger.info("Created %d new ProgressLog rows", len(objs))
        except Exception:
            logger.exception("Database error on ProgressLog save")
            return Response(
                {"error": "Lỗi lưu dữ liệu vào database"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # --- Return ---
        logs = ProgressLog.objects.filter(
            user=user, subject=subject
        ).order_by("week", "day_number")

        return Response(
            {
                "message": "✅ Đã tạo lộ trình học!",
                "subject": subject,
                "items": ProgressLogSerializer(logs, many=True).data,
                "raw_gpt_output": gpt_text_vi[:1000],
            },
            status=status.HTTP_201_CREATED,
        )

    except Exception as unexpected_error:
        logger.exception("Unexpected error in generate_learning_path")
        return Response(
            {
                "error": "Lỗi không xác định",
                "details": str(unexpected_error)[:200],
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
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
        qs = ProgressLog.objects.filter(user=user).order_by(
            "subject", "week", "day_number"
        )
        if subject:
            qs = qs.filter(subject=subject)
        return Response(ProgressLogSerializer(qs, many=True).data, status=200)
    except Exception as e:
        logger.exception("Error in get_progress_list: %s", e)
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
                {"error": "Thiếu thông tin 'id' hoặc 'status'."}, status=400
            )

        try:
            log = ProgressLog.objects.get(id=log_id, user=request.user)
        except ProgressLog.DoesNotExist:
            return Response({"error": "Không tìm thấy bản ghi của bạn"}, status=404)

        log.status = normalize_status(new_status_raw)
        log.save(update_fields=["status"])

        return Response(
            {
                "message": "Cập nhật trạng thái thành công!",
                "item": ProgressLogSerializer(log).data,
            },
            status=200,
        )
    except Exception as e:
        logger.exception("Error in update_progress_status: %s", e)
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
