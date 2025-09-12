# api/views.py
import os
import re
import time
import random
import logging
from functools import lru_cache
from typing import Dict, List

from django.db import transaction
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.views.decorators.csrf import ensure_csrf_cookie

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework import status

from openai import (
    OpenAI,
    RateLimitError,
    APIConnectionError,
    APIError,
    AuthenticationError,
)

from .models import ProgressLog
from .serializers import (
    ProgressLogSerializer,
    RegisterSerializer,
    UserSerializer,
)

# =====================================================
# Logging
# =====================================================
logger = logging.getLogger(__name__)

# =====================================================
# OpenAI (DeepInfra) client - lazy init + cache
# =====================================================
@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    api_key = os.environ.get("DEEPINFRA_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPINFRA_API_KEY is required")
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepinfra.com/v1/openai",
        timeout=60.0,  # client-level timeout
    )

# =====================================================
# Helpers
# =====================================================
def normalize_status(value: str) -> str:
    if not value:
        return "pending"
    return "done" if value.strip().lower() == "done" else "pending"


# Bắt buộc dòng hợp lệ bắt đầu "Ngày N:" và có nội dung, chấp nhận :, ：, -, –, —
LINE_REGEX = re.compile(r"^Ngày\s+(\d{1,2})\s*[:：\-–—]\s*(.+)$", re.IGNORECASE)


def call_with_backoff(
    messages: List[Dict],
    *,
    model: str = "openchat/openchat_3.5",
    max_tokens: int = 1100,           # giảm tải phản hồi
    temperature: float = 0.6,
    overall_deadline: float = 55.0,   # < gunicorn --timeout (khuyến nghị 120)
    per_attempt_timeout: float = 22.0,
    max_attempts: int = 4,
    stop: List[str] | None = None,
):
    """
    Gọi LLM với retry + exponential backoff, luôn tôn trọng tổng deadline
    để tránh worker timeout và giảm tải chờ đợi.
    """
    client = get_openai_client()
    start = time.perf_counter()

    for attempt in range(1, max_attempts + 1):
        elapsed = time.perf_counter() - start
        remain = overall_deadline - elapsed
        if remain <= 0:
            raise TimeoutError("Overall LLM deadline exceeded")

        # Không để mỗi attempt vượt quá thời gian tổng còn lại
        per_timeout = max(5.0, min(per_attempt_timeout, remain - 1.0))

        try:
            t0 = time.perf_counter()
            kwargs = dict(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=per_timeout,
            )
            if stop:
                kwargs["stop"] = stop
            resp = client.chat.completions.create(**kwargs)
            logger.info("✅ LLM call in %.2fs (attempt %d)", time.perf_counter() - t0, attempt)
            return resp

        except AuthenticationError as e:
            logger.error("Auth error: %s", e)
            raise

        except RateLimitError:
            # backoff nhưng không vượt deadline
            sleep = min(40, 2 ** attempt) + random.uniform(0, 0.4)
            if (time.perf_counter() - start) + sleep >= overall_deadline:
                raise TimeoutError("Deadline would be exceeded during backoff sleep")
            logger.warning("⏳ 429 (attempt %d) → retry in %.1fs", attempt, sleep)
            time.sleep(sleep)

        except APIConnectionError:
            sleep = min(15, 2 ** attempt) + random.uniform(0, 0.4)
            if (time.perf_counter() - start) + sleep >= overall_deadline:
                raise TimeoutError("Deadline would be exceeded during backoff sleep")
            logger.warning("🌐 Network (attempt %d) → retry in %.1fs", attempt, sleep)
            time.sleep(sleep)

        except APIError:
            sleep = min(15, 2 ** attempt) + random.uniform(0, 0.4)
            if (time.perf_counter() - start) + sleep >= overall_deadline:
                raise TimeoutError("Deadline would be exceeded during backoff sleep")
            logger.warning("🛠️ Service error (attempt %d) → retry in %.1fs", attempt, sleep)
            time.sleep(sleep)

    raise TimeoutError("LLM call failed after retries within deadline")


def parse_plan_lines(text: str, plan: Dict[int, str]) -> None:
    """Thêm dòng hợp lệ vào plan (không ghi đè ngày đã có)."""
    if not text:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = LINE_REGEX.match(line)
        if not m:
            continue
        try:
            day_num = int(m.group(1))
        except Exception:
            continue
        if 1 <= day_num <= 28 and day_num not in plan:
            plan[day_num] = line


def make_main_messages(class_level: str, subject: str, study_time: str, goal: str) -> List[Dict]:
    """
    Prompt chính, giảm token:
      - Ngày 1: có 'CÔNG CỤ HỖ TRỢ'
      - Ngày 2→28: KHÔNG lặp lại phần 'CÔNG CỤ HỖ TRỢ'
      - Mỗi dòng ≤ 115 ký tự để tiết kiệm tokens
    """
    return [
        {
            "role": "system",
            "content": "Bạn là chuyên gia lập kế hoạch tự học. Trả lời 100% bằng tiếng Việt."
        },
        {
            "role": "user",
            "content": f"""
Lập kế hoạch 4 tuần (28 ngày) cho lớp {class_level}, môn {subject}, mỗi ngày {study_time}. Mục tiêu: {goal}.
YÊU CẦU:
- CHÍNH XÁC 28 dòng (Ngày 1→28), mỗi dòng ≤ 115 ký tự, KHÔNG dòng trống.
- Mỗi dòng BẮT ĐẦU: 'Ngày N:' (có dấu hai chấm), không ký tự khác phía trước.
- Theo CT GDPT 2018. Ngày 28 = ÔN TẬP & KIỂM TRA TỔNG HỢP.
- KHÔNG tiêu đề/markdown/code block/giải thích.

QUY TẮC 'CÔNG CỤ HỖ TRỢ':
- Ngày 1: PHẢI có ' | CÔNG CỤ HỖ TRỢ: <ứng dụng liên quan đến môn {subject}>'.
- Ngày 2→28: KHÔNG thêm phần 'CÔNG CỤ HỖ TRỢ'.

ĐỊNH DẠNG:
- Ngày 1:
  Ngày 1: <nội dung> | TỪ KHÓA TÌM KIẾM: <từ khóa> | Bài tập tự luyện: <gợi ý> | CÔNG CỤ HỖ TRỢ: <ứng dụng>
- Ngày 2→28:
  Ngày N: <nội dung> | TỪ KHÓA TÌM KIẾM: <từ khóa> | Bài tập tự luyện: <gợi ý>

Chỉ in đúng 28 dòng theo mẫu, không thêm gì khác.
""".strip(),
        },
    ]


def make_continue_messages(missing_days: List[int], subject: str, day1_tools_required: bool) -> List[Dict]:
    """
    Prompt bổ sung:
      - Nếu thiếu Ngày 1 → Ngày 1 phải có 'CÔNG CỤ HỖ TRỢ'
      - Các ngày khác KHÔNG 'CÔNG CỤ HỖ TRỢ'
    """
    days_str = ", ".join(str(d) for d in missing_days)
    count = len(missing_days)
    tools_rule = (
        f"- Trong danh sách có Ngày 1 → Ngày 1 PHẢI có ' | CÔNG CỤ HỖ TRỢ: <ứng dụng liên quan đến môn {subject}>'\n"
        "- Các ngày khác KHÔNG thêm phần 'CÔNG CỤ HỖ TRỢ'."
        if day1_tools_required
        else "- KHÔNG thêm 'CÔNG CỤ HỖ TRỢ' cho bất kỳ ngày nào (Ngày 1 đã có trước đó)."
    )
    return [
        {"role": "system", "content": "Tiếp tục kế hoạch, giữ nguyên định dạng và yêu cầu."},
        {
            "role": "user",
            "content": f"""
In CHÍNH XÁC {count} dòng tương ứng các ngày: {days_str}.
Mỗi dòng BẮT ĐẦU 'Ngày N:' (có dấu hai chấm), mỗi dòng ≤ 115 ký tự, KHÔNG dòng trống.
KHÔNG in ngày ngoài danh sách. KHÔNG tiêu đề/markdown/giải thích.

{tools_rule}

ĐỊNH DẠNG:
- Nếu là Ngày 1 (khi nằm trong danh sách):
  Ngày 1: <nội dung> | TỪ KHÓA TÌM KIẾM: <từ khóa> | Bài tập tự luyện: <gợi ý> | CÔNG CỤ HỖ TRỢ: <ứng dụng>
- Các ngày khác:
  Ngày N: <nội dung> | TỪ KHÓA TÌM KIẾM: <từ khóa> | Bài tập tự luyện: <gợi ý>

Chỉ in đúng {count} dòng theo mẫu, không thêm gì khác.
""".strip(),
        },
    ]


def generate_fallback_plan(class_level: str, subject: str, study_time: str, goal: str) -> Dict[int, str]:
    """
    Fallback ngắn gọn (giảm token):
    - Ngày 1: có 'CÔNG CỤ HỖ TRỢ'
    - Ngày 2→28: không có phần 'CÔNG CỤ HỖ TRỢ'
    """
    plan: Dict[int, str] = {}
    tools = "Google Classroom, YouTube, Khan Academy"

    for day in range(1, 29):
        if day == 1:
            plan[day] = (
                f"Ngày 1: Định hướng & tài nguyên học - {subject} | "
                f"TỪ KHÓA TÌM KIẾM: {subject} lớp {class_level} tài nguyên | "
                f"Bài tập tự luyện: Thiết lập môi trường | CÔNG CỤ HỖ TRỢ: {tools}"
            )
        elif day == 28:
            plan[day] = (
                f"Ngày 28: ÔN TẬP & KIỂM TRA TỔNG HỢP - {goal} | "
                f"TỪ KHÓA TÌM KIẾM: {subject} đề thi thử | "
                f"Bài tập tự luyện: Làm đề full {study_time}"
            )
        else:
            plan[day] = (
                f"Ngày {day}: Ôn tập/chủ đề liên quan {goal} - {subject} | "
                f"TỪ KHÓA TÌM KIẾM: {subject} lớp {class_level} {goal} | "
                f"Bài tập tự luyện: Thực hành {study_time}"
            )
    return plan

# =====================================================
# Generate learning path
# =====================================================
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def generate_learning_path(request):
    """
    Sinh kế hoạch 28 ngày (giảm tải bằng prompt ngắn gọn + max_tokens nhỏ + deadline tổng).
    """
    try:
        user = request.user
        data = request.data

        class_level = (data.get("class_level") or "").strip()
        subject     = (data.get("subject") or "").strip()
        study_time  = (data.get("study_time") or "").strip()
        goal        = (data.get("goal") or "").strip()

        if not all([class_level, subject, study_time, goal]):
            return Response({"error": "Thiếu thông tin bắt buộc."}, status=400)

        # 1) Lần 1: yêu cầu đủ 28 dòng
        messages = make_main_messages(class_level, subject, study_time, goal)
        logger.info("Calling DeepInfra (main 28 lines)...")

        plan: Dict[int, str] = {}
        ai_used = False

        try:
            resp = call_with_backoff(
                messages,
                max_tokens=1100,
                temperature=0.6,
                overall_deadline=55.0,     # < gunicorn timeout
                per_attempt_timeout=22.0,
                max_attempts=4,
                stop=None,
            )
        except TimeoutError as e:
            logger.warning("LLM overall timeout: %s", e)
            return Response({"error": "AI đang chậm, vui lòng thử lại sau."}, status=504)

        text = (resp.choices[0].message.content or "").strip()
        parse_plan_lines(text, plan)
        logger.info("Parsed %d/28 days (first pass)", len(plan))

        # 2) Nếu thiếu → in CHÍNH XÁC các ngày còn thiếu (tối đa 2 vòng)
        tries = 0
        while len(plan) < 28 and tries < 2:
            missing = sorted([d for d in range(1, 29) if d not in plan])
            logger.info("Missing %d days: %s", len(missing), missing)
            cont_msgs = make_continue_messages(
                missing,
                subject,
                day1_tools_required=(1 in missing),
            )
            try:
                resp2 = call_with_backoff(
                    cont_msgs,
                    max_tokens=400,           # nhỏ hơn để giảm tải
                    temperature=0.5,
                    overall_deadline=40.0,
                    per_attempt_timeout=18.0,
                    max_attempts=3,
                    stop=None,
                )
            except TimeoutError:
                logger.warning("Continue phase timed out; will fill missing with fallback later.")
                break

            text2 = (resp2.choices[0].message.content or "").strip()
            parse_plan_lines(text2, plan)
            logger.info("After continue #%d → %d/28 days", tries + 1, len(plan))
            tries += 1

        if len(plan) > 0:
            ai_used = True

        # 3) Còn thiếu → chỉ fill phần thiếu bằng fallback (không đè phần đã có)
        if len(plan) < 28:
            fb = generate_fallback_plan(class_level, subject, study_time, goal)
            for d in range(1, 29):
                if d not in plan:
                    plan[d] = fb[d]
            logger.info("Filled missing days with fallback → %d/28 days", len(plan))

        # 4) Lưu DB
        with transaction.atomic():
            ProgressLog.objects.filter(user=user, subject=subject).delete()
            objs = []
            for day in range(1, 29):
                week = (day - 1) // 7 + 1
                objs.append(
                    ProgressLog(
                        user=user,
                        subject=subject,
                        week=week,
                        day_number=day,
                        task_title=plan[day],
                        status="pending",
                    )
                )
            ProgressLog.objects.bulk_create(objs)

        logs = ProgressLog.objects.filter(user=user, subject=subject).order_by("week", "day_number")
        return Response(
            {
                "message": "✅ Đã tạo lộ trình học!",
                "subject": subject,
                "items": ProgressLogSerializer(logs, many=True).data,
                "ai_generated": ai_used,
                "note": "Ngày 1 có 'CÔNG CỤ HỖ TRỢ', các ngày sau không lặp lại; tối ưu token + deadline.",
            },
            status=201,
        )

    except AuthenticationError:
        return Response({"error": "Lỗi xác thực API key. Vui lòng kiểm tra cấu hình."}, status=401)
    except RateLimitError:
        return Response({"error": "Đã vượt quá giới hạn AI. Vui lòng thử lại sau."}, status=429)
    except APIConnectionError:
        return Response({"error": "Không kết nối được tới AI service."}, status=502)
    except APIError as e:
        return Response({"error": "AI service gặp sự cố.", "details": str(e)[:200]}, status=502)
    except Exception as e:
        logger.exception("Unexpected error in generate_learning_path")
        return Response({"error": "Lỗi không xác định", "details": str(e)[:200]}, status=500)

# =====================================================
# Get progress list
# =====================================================
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
        logger.exception("Error in get_progress_list: %s", e)
        return Response({"error": "Lỗi lấy danh sách tiến độ"}, status=500)

# =====================================================
# Update progress status
# =====================================================
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
        logger.exception("Error in update_progress_status: %s", e)
        return Response({"error": "Lỗi cập nhật trạng thái"}, status=500)

# =====================================================
# Auth & CSRF endpoints
# =====================================================
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
