import os
import re
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
# OpenAI (DeepInfra) client
# =========================
DEEPINFRA_API_KEY = os.environ.get("DEEPINFRA_API_KEY")
if not DEEPINFRA_API_KEY:
    raise ValueError("DEEPINFRA_API_KEY environment variable is required")

openai = OpenAI(
    api_key=DEEPINFRA_API_KEY,
    base_url="https://api.deepinfra.com/v1/openai",
)

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
    data = request.data
    class_level = (data.get("class_level") or "").strip()
    subject = (data.get("subject") or "").strip()
    study_time = (data.get("study_time") or "").strip()
    goal = (data.get("goal") or "").strip()
    
    if not all([class_level, subject, study_time, goal]):
        return Response(
            {"error": "Thiếu thông tin bắt buộc."}, 
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Prompt
    messages = [
        {
            "role": "system",
            "content": "Bạn là chuyên gia lập kế hoạch tự học. Trả lời HOÀN TOÀN bằng tiếng Việt."
        },
        {
            "role": "user",
            "content": f"""
Hãy lập kế hoạch tự học 4 tuần cho học sinh lớp {class_level} muốn cải thiện môn {subject}.
Học sinh học {study_time} mỗi ngày. Mục tiêu: {goal}.
Trả lời theo đúng format sau:
Tuần 1:
Ngày 1: [Nội dung học] | Link tài liệu: [link]
...
Ngày 28: ...
"""
        }
    ]
    
    # Call LLM
    try:
        resp = openai.chat.completions.create(
            model="openchat/openchat_3.5",
            messages=messages,
            stream=False,
        )
    except Exception as e:
        return Response(
            {"error": "Lỗi GPT", "details": str(e)}, 
            status=500
        )
    
    gpt_text_vi = (resp.choices[0].message.content or "").strip()
    
    # Parse "Ngày N: ..."
    day_line_regex = re.compile(r"^Ngày\s+(\d{1,2})\s*[:\-\–]\s*(.+)$", re.IGNORECASE)
    plan = {}
    
    for raw in gpt_text_vi.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = day_line_regex.match(line)
        if not m:
            continue
        day_num = int(m.group(1))
        if 1 <= day_num <= 28:
            plan[day_num] = m.group(2).strip()
    
    if not plan:
        return Response(
            {"error": "Không thể phân tích nội dung từ GPT."}, 
            status=500
        )
    
    # Save to DB (per user & subject)
    user = request.user
    with transaction.atomic():
        ProgressLog.objects.filter(user=user, subject=subject).delete()
        objs = []
        for day_number in sorted(plan.keys()):
            task_text = str(plan[day_number])
            week = (day_number - 1) // 7 + 1
            objs.append(ProgressLog(
                user=user,
                subject=subject,
                week=week,
                day_number=day_number,
                task_title=task_text,
                status="pending",
            ))
        ProgressLog.objects.bulk_create(objs)
    
    logs = ProgressLog.objects.filter(user=user, subject=subject).order_by("week", "day_number")
    return Response(
        {
            "message": "✅ Đã tạo lộ trình học!",
            "subject": subject,
            "items": ProgressLogSerializer(logs, many=True).data,
            "raw_gpt_output": gpt_text_vi,
        },
        status=201,
    )

# =========================================
# Get progress list (chỉ user đăng nhập mới xem được)
# =========================================
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_progress_list(request):
    subject = request.query_params.get("subject")
    user = request.user
    qs = ProgressLog.objects.filter(user=user).order_by("subject", "week", "day_number")
    if subject:
        qs = qs.filter(subject=subject)
    return Response(ProgressLogSerializer(qs, many=True).data, status=200)

# =========================================
# Update progress status
# =========================================
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def update_progress_status(request):
    """
    Body:
      { "id": <int>, "status": "done" | "pending" | "not_done" }
    """
    log_id = request.data.get("id")
    new_status_raw = request.data.get("status")
    
    if not log_id or new_status_raw is None:
        return Response(
            {"error": "Thiếu thông tin 'id' hoặc 'status'."}, 
            status=400
        )
    
    try:
        log = ProgressLog.objects.get(id=log_id, user=request.user)  # enforce ownership
    except ProgressLog.DoesNotExist:
        return Response(
            {"error": "Không tìm thấy bản ghi của bạn"}, 
            status=404
        )
    
    log.status = normalize_status(new_status_raw)  # Fixed function name
    log.save(update_fields=["status"])
    
    return Response(
        {
            "message": "Cập nhật trạng thái thành công!", 
            "item": ProgressLogSerializer(log).data
        },
        status=200,
    )

# =========================================
# Auth & CSRF
# =========================================
@api_view(["GET"])
@ensure_csrf_cookie             # set csrftoken cookie
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
            {
                "message": "Đăng ký thành công!", 
                "user": UserSerializer(user).data
            }, 
            status=201
        )
    return Response(serializer.errors, status=400)

@api_view(["POST"])
@permission_classes([AllowAny])
def login_view(request):
    username = request.data.get("username")
    password = request.data.get("password")
    user = authenticate(request, username=username, password=password)
    
    if not user:
        return Response(
            {"detail": "Sai tên đăng nhập hoặc mật khẩu."}, 
            status=400
        )
    
    login(request, user)
    return Response(
        {
            "message": "Đăng nhập thành công!", 
            "username": user.username
        }
    )

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
