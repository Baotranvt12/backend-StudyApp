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
# OpenAI (DeepInfra → Claude)
# =========================
DEEPINFRA_API_KEY = os.environ.get("DEEPINFRA_API_KEY")

if not DEEPINFRA_API_KEY:
    raise ValueError("Missing DEEPINFRA_API_KEY")

openai = OpenAI(
    api_key=DEEPINFRA_API_KEY,
    base_url="https://api.deepinfra.com/v1/openai",
)

# =========================
# Helper
# =========================
def normalize_status(value: str) -> str:
    if not value:
        return "pending"
    v = value.strip().lower()
    return "done" if v == "done" else "pending"


# =========================
# Generate learning path
# =========================
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def generate_learning_path(request):
    """Tạo kế hoạch tự học 4 tuần / 28 ngày"""
    try:
        logger.info(f"User: {request.user.username}")
        data = request.data
        class_level = (data.get("class_level") or "").strip()
        subject = (data.get("subject") or "").strip()
        study_time = (data.get("study_time") or "").strip()
        goal = (data.get("goal") or "").strip()

        if not all([class_level, subject, study_time, goal]):
            return Response({"error": "Thiếu dữ liệu đầu vào."}, status=400)

        messages = [
            {
                "role": "system",
                "content": (
                    "Bạn là chuyên gia lập kế hoạch học tập cá nhân hóa, trả lời hoàn toàn bằng tiếng Việt."
                ),
            },
            {
                "role": "user",
                "content": f"""
Lập kế hoạch tự học 4 tuần (28 ngày) cho học sinh lớp {class_level}, môn {subject}.
Thời lượng học: {study_time}/ngày. Mục tiêu: {goal}.
Yêu cầu:
- Có 28 dòng, mỗi dòng tương ứng Ngày 1 → Ngày 28.
- Mỗi dòng là 1 hoạt động học thực tế, đi từ cơ bản đến nâng cao.
- Ngày 28 là ÔN TẬP & KIỂM TRA TỔNG HỢP.
- Không markdown, không code block, không giải thích.
Định dạng mỗi dòng:
Ngày N: <nội dung> | TỪ KHÓA TÌM KIẾM: <từ khóa> | Bài tập tự luyện: <bài tập> | CÔNG CỤ HỖ TRỢ: <công cụ>
Chỉ in đúng 28 dòng theo mẫu trên.
""",
            },
        ]

        logger.info("🧠 Calling Claude API (DeepInfra)...")
        resp = openai.chat.completions.create(
            model="anthropic/claude-4-sonnet",
            messages=messages,
            max_tokens=4000,  # <-- tăng giới hạn
            temperature=0.4,  # <-- giảm để ổn định format
            stream=False,
        )

        gpt_text_vi = (resp.choices[0].message.content or "").strip()
        if not gpt_text_vi:
            return Response({"error": "Claude không trả về nội dung."}, status=500)

        logger.info(f"Claude response length: {len(gpt_text_vi)}")

        # ==============
        # Parse output
        # ==============
        text_clean = (
            unicodedata.normalize("NFKC", gpt_text_vi)
            .replace("\u00a0", " ")
            .strip()
        )

        lines = [l.strip() for l in text_clean.splitlines() if l.strip()]
        plan = {}
        buffer = ""

        for line in lines:
            # nếu là đầu "Ngày N"
            if re.match(r"^ngày\s*\d{1,2}\b", line, flags=re.IGNORECASE):
                # lưu dòng cũ
                if buffer:
                    m = re.search(r"ngày\s*(\d{1,2})", buffer, flags=re.IGNORECASE)
                    if m:
                        day = int(m.group(1))
                        if 1 <= day <= 28:
                            plan[day] = buffer.strip()
                    buffer = line
                else:
                    buffer = line
            else:
                buffer += " " + line

        # Lưu dòng cuối
        if buffer:
            m = re.search(r"ngày\s*(\d{1,2})", buffer, flags=re.IGNORECASE)
            if m:
                day = int(m.group(1))
                if 1 <= day <= 28:
                    plan[day] = buffer.strip()

        logger.info(f"📊 Parsed {len(plan)} unique days from Claude output")

        # Nếu còn thiếu -> bổ sung placeholder
        missing_days = [d for d in range(1, 29) if d not in plan]
        if missing_days:
            logger.warning(f"⚠️ Claude trả thiếu {len(missing_days)} ngày: {missing_days}")
            for d in missing_days:
                plan[d] = (
                    f"Ngày {d}: Ôn tập kiến thức theo chủ đề | "
                    f"TỪ KHÓA TÌM KIẾM: {subject} | "
                    f"Bài tập tự luyện: Tự học, luyện thêm qua sách giáo khoa | "
                    f"CÔNG CỤ HỖ TRỢ: Google Classroom"
                )

        logger.info(f"✅ Tổng số ngày: {len(plan)} (Expected 28)")

        # =========================
        # Save to DB
        # =========================
        user = request.user
        with transaction.atomic():
            ProgressLog.objects.filter(user=user, subject=subject).delete()

            objs = []
            for day_number in sorted(plan.keys()):
                week = (day_number - 1) // 7 + 1
                objs.append(
                    ProgressLog(
                        user=user,
                        subject=subject,
                        week=week,
                        day_number=day_number,
                        task_title=plan[day_number],
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
                "raw_gpt_output": gpt_text_vi[:2000],
            },
            status=201,
        )

    except Exception as e:
        logger.exception("Error in generate_learning_path")
        return Response(
            {"error": "Lỗi hệ thống", "details": str(e)[:200]},
            status=500,
        )


# =========================
# Get progress list
# =========================
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_progress_list(request):
    subject = request.query_params.get("subject")
    user = request.user
    qs = ProgressLog.objects.filter(user=user)
    if subject:
        qs = qs.filter(subject=subject)
    qs = qs.order_by("subject", "week", "day_number")
    return Response(ProgressLogSerializer(qs, many=True).data)


# =========================
# Update progress status
# =========================
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def update_progress_status(request):
    log_id = request.data.get("id")
    new_status = request.data.get("status")
    if not log_id or not new_status:
        return Response({"error": "Thiếu id hoặc status"}, status=400)

    try:
        log = ProgressLog.objects.get(id=log_id, user=request.user)
        log.status = normalize_status(new_status)
        log.save(update_fields=["status"])
        return Response(
            {"message": "Cập nhật thành công", "item": ProgressLogSerializer(log).data}
        )
    except ProgressLog.DoesNotExist:
        return Response({"error": "Không tìm thấy bản ghi"}, status=404)


# =========================
# Auth handlers
# =========================
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
        return Response({"user": UserSerializer(user).data}, status=201)
    return Response(serializer.errors, status=400)


@api_view(["POST"])
@permission_classes([AllowAny])
def login_view(request):
    user = authenticate(
        request,
        username=request.data.get("username"),
        password=request.data.get("password"),
    )
    if not user:
        return Response({"error": "Sai tên đăng nhập hoặc mật khẩu"}, status=400)
    login(request, user)
    return Response({"message": "Đăng nhập thành công", "username": user.username})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def logout_view(request):
    logout(request)
    return Response({"message": "Đã đăng xuất"})


@api_view(["GET"])
@permission_classes([AllowAny])
def whoami(request):
    if request.user.is_authenticated:
        return Response({"username": request.user.username})
    return Response({"username": None})
