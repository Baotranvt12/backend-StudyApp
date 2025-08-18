# api/views.py
import os
import re
from django.db import transaction
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import authenticate, login,  logout
from django.contrib.auth.models import User

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from django.middleware import csrf
from django.contrib.auth import logout
from django.views.decorators.csrf import ensure_csrf_cookie
from .serializers import RegisterSerializer, UserSerializer



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
DEEPINFRA_API_KEY = os.environ.get("DEEPINFRA_API_KEY", "rt4TcoQRkgGx6YAlEPArqiVKJd1lPAxx")
openai = OpenAI(
    api_key=DEEPINFRA_API_KEY,
    base_url="https://api.deepinfra.com/v1/openai",
)

# =========================
# Helpers
# =========================
def _get_current_user(request):
    """
    Use the signed-in user if available; otherwise fall back to 'default_user'
    for demo/testing so the app can still function without auth.
    """
    if getattr(request, "user", None) and request.user.is_authenticated:
        return request.user
    user, _ = User.objects.get_or_create(username="default_user")
    return user


def _normalize_status(value: str) -> str:
    """
    Normalize arbitrary status strings into 'pending' or 'done'.
    (Support legacy 'not_done' -> 'pending')
    """
    if not value:
        return "pending"
    v = value.strip().lower()
    return "done" if v == "done" else "pending"


# =========================================
# Generate learning path (store per-user)
# =========================================
@csrf_exempt
@api_view(["POST"])
@permission_classes([AllowAny])  # switch to IsAuthenticated when auth is enforced
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
        return Response({"error": "Thiếu thông tin bắt buộc."}, status=status.HTTP_400_BAD_REQUEST)

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
Ngày 7: ...

Tuần 2:
Ngày 8: ...
...
Ngày 14: ...

Tuần 3:
Ngày 15: ...
...
Ngày 21: ...

Tuần 4:
Ngày 22: ...
...
Ngày 28: ...

Chú ý: Mỗi dòng là 1 ngày học, có mô tả bài học và link web hữu ích miễn phí cho học sinh.
Không thêm lời mở đầu hay kết luận. Trả đúng format trên.
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
        return Response({"error": "Lỗi GPT", "details": str(e)}, status=500)

    gpt_text_vi = (resp.choices[0].message.content or "").strip()

    # Parse "Ngày N: ..."
    # Accept: "Ngày 1:", "Ngày 1 -", "Ngày 1 –", "Ngày 1 —"
    day_line_regex = re.compile(r"^Ngày\s+(\d{1,2})\s*[:\-–—]\s*(.+)$", re.IGNORECASE)
    plan = {}  # { int(day_number): str(task_text) }
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
        return Response({"error": "Không thể phân tích nội dung từ GPT."}, status=500)

    # Save to DB (per user & subject)
    user = _get_current_user(request)
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
                task_title=task_text,  # includes: "... | Link tài liệu: https://..."
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
# Get progress list (filter by subject)
# =========================================
@api_view(["GET"])
@permission_classes([AllowAny])  # switch to IsAuthenticated when auth is enforced
def get_progress_list(request):
    subject = request.query_params.get("subject")
    user = _get_current_user(request)

    qs = ProgressLog.objects.filter(user=user).order_by("subject", "week", "day_number")
    if subject:
        qs = qs.filter(subject=subject)

    return Response(ProgressLogSerializer(qs, many=True).data, status=200)


# =========================================
# Update progress status
# =========================================
@csrf_exempt
@api_view(["POST"])
@permission_classes([AllowAny])  # switch to IsAuthenticated and enforce ownership if needed
def update_progress_status(request):
    """
    Body:
      { "id": <int>, "status": "done" | "pending" | "not_done" }
    """
    log_id = request.data.get("id")
    new_status_raw = request.data.get("status")

    if not log_id or new_status_raw is None:
        return Response({"error": "Thiếu thông tin 'id' hoặc 'status'."}, status=400)

    try:
        log = ProgressLog.objects.get(id=log_id)
    except ProgressLog.DoesNotExist:
        return Response({"error": "Không tìm thấy bản ghi"}, status=404)

    # Optional: enforce owner
    # current_user = _get_current_user(request)
    # if log.user_id != current_user.id:
    #     return Response({"error": "Không có quyền cập nhật bản ghi này."}, status=403)

    log.status = _normalize_status(new_status_raw)
    log.save(update_fields=["status"])

    return Response(
        {"message": "Cập nhật trạng thái thành công!", "item": ProgressLogSerializer(log).data},
        status=200,
    )


# =========================================
# Auth endpoints (Register / Login / Me)
# =========================================
# @csrf_exempt
# @api_view(["POST"])
# @permission_classes([AllowAny])
# def register(request):
#     """
#     Body: { "username": "...", "email": "...", "password": "...", "password2": "...", "first_name": "...", "last_name": "..." }
#     """
#     serializer = RegisterSerializer(data=request.data)
#     if serializer.is_valid():
#         user = serializer.save()
#         return Response(
#             {"message": "Đăng ký thành công!", "user": UserSerializer(user).data},
#             status=201,
#         )
#     return Response(serializer.errors, status=400)


# # api/views.py
# @csrf_exempt
# @api_view(["POST"])
# @permission_classes([AllowAny])
# def login_view(request):
#     username = request.data.get("username")
#     password = request.data.get("password")
#     user = authenticate(request, username=username, password=password)
#     if user is None:
#         return Response({"detail": "Sai thông tin đăng nhập"}, status=400)

#     login(request, user)

#     # Xóa dữ liệu demo default_user
#     ProgressLog.objects.filter(user__username="default_user").delete()

#     return Response({"username": user.username}, status=200)


# @api_view(["GET"])
# @permission_classes([IsAuthenticated])
# def me(request):
#     """Return current user info (requires Authorization or session)."""
#     return Response(UserSerializer(request.user).data, status=200)

# @api_view(["GET"])
# @permission_classes([AllowAny])
# def get_csrf_token(request):
#     """
#     Trả về JSON: {"csrfToken": "<value>"} và đặt cookie csrftoken.
#     """
#     token = csrf.get_token(request)   # tạo (hoặc lấy) token và set cookie
#     return Response({"csrfToken": token})



# @api_view(["POST"])
# @permission_classes([IsAuthenticated])          # chỉ người đã đăng nhập mới được gọi
# def logout_view(request):
#     """
#     Đánh dấu session đã hoàn tất và xóa cookie `sessionid`.
#     Nếu bạn đang dùng CSRF, client phải gửi header X‑CSRFToken
#     (có thể lấy từ cookie `csrftoken`).
#     """
#     logout(request)               # Django sẽ xóa session & cookie tương ứng
#     return Response(
#         {"detail": "Bạn đã thoát tài khoản."},
#         status=status.HTTP_200_OK,
#     )

@api_view(["GET"])
@permission_classes([AllowAny])
@ensure_csrf_cookie
def get_csrf(request):
    # Đặt csrftoken cookie để FE lấy và gửi lại khi POST
    return Response({"csrfToken": request.META.get("CSRF_COOKIE", "")})

@api_view(["POST"])
@permission_classes([AllowAny])
def register(request):
    serializer = RegisterSerializer(data=request.data)
    if serializer.is_valid():
        user = serializer.save()
        return Response({"message": "Đăng ký thành công!", "user": UserSerializer(user).data}, status=201)
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
def logout_view(request):
    logout(request)
    return Response({"message": "Đã đăng xuất."})

@api_view(["GET"])
def whoami(request):
    if request.user.is_authenticated:
        return Response({"username": request.user.username})
    return Response({"username": None}, status=200)