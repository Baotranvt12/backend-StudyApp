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

from openai import OpenAI  # DeepInfra OpenAI-compatible

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
# OpenAI (DeepInfra) client - lazy init
# =====================================================
@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """
    Ưu tiên DEEPINFRA_KEY; fallback DEEPINFRA_API_KEY cho tương thích cũ.
    """
    api_key = os.environ.get("DEEPINFRA_KEY") or os.environ.get("DEEPINFRA_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPINFRA_KEY (hoặc DEEPINFRA_API_KEY) is required")
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepinfra.com/v1/openai",
    )

# =====================================================
# Helpers
# =====================================================
def normalize_status(value: str) -> str:
    if not value:
        return "pending"
    return "done" if value.strip().lower() == "done" else "pending"

# Bắt đầu đúng "Ngày N:" (1..28)
DAY_HEAD_RE = re.compile(r"^Ngày\s+([1-9]|1\d|2[0-8]):", re.IGNORECASE)

def _count_ok(lines: List[str]) -> int:
    return sum(1 for ln in lines if DAY_HEAD_RE.match(ln.strip()))

def _parse_lines(text: str) -> List[str]:
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]

def _make_skeleton(subject: str) -> str:
    sk = []
    for d in range(1, 29):
        if d == 1:
            sk.append(
                f"Ngày {d}: <nội dung ngắn> | TỪ KHÓA: <từ khóa> | BT: <gợi ý> | CÔNG CỤ HỖ TRỢ: <app môn {subject}>"
            )
        elif d == 28:
            sk.append(
                "Ngày 28: ÔN & KIỂM TRA TỔNG HỢP - <tóm tắt mục tiêu> | TỪ KHÓA: <từ khóa> | BT: <đề thử>"
            )
        else:
            sk.append(f"Ngày {d}: <nội dung ngắn> | TỪ KHÓA: <từ khóa> | BT: <gợi ý>")
    return "\n".join(sk)

SYSTEM_MSG = (
    "Bạn là chuyên gia lập kế hoạch tự học. Trả lời 100% tiếng Việt. "
    "PHẢI in đúng 28 dòng từ 'Ngày 1:' đến 'Ngày 28:'; mỗi dòng ≤ 90 ký tự; "
    "không tiêu đề/markdown; không dòng trống; mỗi số ngày dùng đúng 1 lần; "
    "không thêm mô tả ngoài 28 dòng."
)

def _make_user_msg(class_level: str, subject: str, study_time: str, goal: str, skeleton_text: str) -> str:
    return f"""
Học sinh lớp {class_level}, môn {subject}, thời lượng {study_time}/ngày. Mục tiêu: {goal}.
Bám CT GDPT 2018, tăng dần độ khó, không gộp 2 ngày vào 1.

Hãy THAY THẾ các <...> trong KHUNG sau và GIỮ NGUYÊN tiền tố "Ngày N:".
Nếu thiếu dòng nào, tự bổ sung đến đủ 28 dòng.
Chỉ ghi "CÔNG CỤ HỖ TRỢ" ở Ngày 1; các ngày còn lại KHÔNG lặp lại phần này.

{skeleton_text}

Chỉ in đúng 28 dòng ở trên, không thêm nội dung khác.
""".strip()

def generate_fallback_plan(class_level: str, subject: str, study_time: str, goal: str) -> Dict[int, str]:
    """
    Fallback ngắn gọn: Ngày 1 có 'CÔNG CỤ HỖ TRỢ', các ngày còn lại không lặp lại.
    """
    plan: Dict[int, str] = {}
    tools = "YouTube, VietJack, OLM"
    for day in range(1, 29):
        if day == 1:
            plan[day] = (
                f"Ngày 1: Định hướng & tài nguyên học - {subject} | "
                f"TỪ KHÓA: {subject} lớp {class_level} tài nguyên | "
                f"BT: Thiết lập môi trường | CÔNG CỤ HỖ TRỢ: {tools}"
            )
        elif day == 28:
            plan[day] = (
                f"Ngày 28: ÔN & KIỂM TRA TỔNG HỢP - {goal} | "
                f"TỪ KHÓA: {subject} đề thi thử | BT: Làm đề full {study_time}"
            )
        else:
            plan[day] = (
                f"Ngày {day}: Chủ đề liên quan {goal} - {subject} | "
                f"TỪ KHÓA: {subject} lớp {class_level} {goal} | BT: Thực hành {study_time}"
            )
    return plan

# =====================================================
# LLM core (theo script của bạn: 1 lượt chính + 1 lượt bù)
# =====================================================
def generate_28_lines_with_llm(class_level: str, subject: str, study_time: str, goal: str,
                               model: str = "meta-llama/Meta-Llama-3-8B-Instruct") -> List[str]:
    """
    Trả về danh sách các dòng 'Ngày N: ...' (có thể < 28 nếu model thiếu; view sẽ fallback).
    Timeout ngắn để tránh 504.
    """
    client = get_openai_client()
    skeleton_text = _make_skeleton(subject)
    user_msg = _make_user_msg(class_level, subject, study_time, goal, skeleton_text)

    # ---- Lượt 1
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_MSG},
                  {"role": "user", "content": user_msg}],
        temperature=0.0,
        top_p=1,
        max_tokens=2000,
        stop=["\nNgày 29", "Ngày 29:"],
        timeout=18.0,  # ngắn để không treo
    )
    lines = _parse_lines(resp.choices[0].message.content)
    if _count_ok(lines) >= 28:
        ok_lines = [ln for ln in lines if DAY_HEAD_RE.match(ln)]
        return ok_lines[:28]

    # ---- Lượt 2 (bù phần thiếu)
    printed_days = set()
    for ln in lines:
        m = re.match(r"^Ngày\s+(\d+):", ln)
        if m:
            printed_days.add(int(m.group(1)))
    missing = [d for d in range(1, 29) if d not in printed_days]

    cont_prompt = "In TIẾP đúng các dòng còn thiếu theo format trước, không lặp, không giải thích:\n"
    for d in missing:
        if d == 28:
            cont_prompt += "Ngày 28: ÔN & KIỂM TRA TỔNG HỢP - <tóm tắt mục tiêu> | TỪ KHÓA: <từ khóa> | BT: <đề thử>\n"
        else:
            cont_prompt += f"Ngày {d}: <nội dung ngắn> | TỪ KHÓA: <từ khóa> | BT: <gợi ý>\n"

    resp2 = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_MSG},
                  {"role": "user", "content": cont_prompt.strip()}],
        temperature=0.0,
        max_tokens=800,
        stop=["\nNgày 29", "Ngày 29:"],
        timeout=10.0,
    )
    more_lines = _parse_lines(resp2.choices[0].message.content)

    # Hợp nhất theo ngày (ưu tiên dòng xuất hiện sau cùng cho cùng ngày)
    all_lines: Dict[int, str] = {}
    for ln in lines + more_lines:
        m = re.match(r"^Ngày\s+(\d+):", ln)
        if m:
            all_lines[int(m.group(1))] = ln.strip()
    return [all_lines[d] for d in range(1, 29) if d in all_lines]

# =====================================================
# Generate learning path (API)
# =====================================================
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def generate_learning_path(request):
    """
    Sinh kế hoạch 28 ngày:
      - LLM (2 lượt, timeout ngắn) → hậu kiểm
      - Thiếu thì fallback để luôn đủ 28 ngày
      - Lưu vào ProgressLog
    """
    try:
        user = request.user
        data = request.data

        class_level = (data.get("class_level") or "").strip()
        subject     = (data.get("subject") or "").strip()
        study_time  = (data.get("study_time") or "").strip()
        goal        = (data.get("goal") or "").strip()

        if not all([class_level, subject, study_time, goal]):
            return Response({"error": "Thiếu thông tin bắt buộc."}, status=status.HTTP_400_BAD_REQUEST)

        # 1) Gọi LLM theo script
        try:
            lines = generate_28_lines_with_llm(class_level, subject, study_time, goal)
        except Exception as e:
            logger.warning("LLM error, will fallback. Details: %s", str(e)[:200])
            lines = []

        # 2) Ghép vào 'plan' theo số ngày; nếu thiếu sẽ fallback
        plan: Dict[int, str] = {}
        for ln in lines:
            m = re.match(r"^Ngày\s+(\d+):", ln)
            if m:
                d = int(m.group(1))
                if 1 <= d <= 28:
                    plan[d] = ln

        if len(plan) < 28:
            fb = generate_fallback_plan(class_level, subject, study_time, goal)
            for d in range(1, 29):
                if d not in plan:
                    plan[d] = fb[d]

        # 3) Lưu DB
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
                "ai_generated": bool(lines),  # True nếu có ít nhất một dòng từ LLM
                "note": "Áp dụng prompt skeleton 28 dòng; có lượt bổ sung + fallback an toàn, timeout ngắn.",
            },
            status=status.HTTP_201_CREATED,
        )

    except Exception as e:
        logger.exception("Unexpected error in generate_learning_path")
        return Response({"error": "Lỗi không xác định", "details": str(e)[:200]}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

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
        return Response(ProgressLogSerializer(qs, many=True).data, status=status.HTTP_200_OK)
    except Exception as e:
        logger.exception("Error in get_progress_list: %s", e)
        return Response({"error": "Lỗi lấy danh sách tiến độ"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

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
            return Response({"error": "Thiếu thông tin 'id' hoặc 'status'."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            log = ProgressLog.objects.get(id=log_id, user=request.user)
        except ProgressLog.DoesNotExist:
            return Response({"error": "Không tìm thấy bản ghi của bạn"}, status=status.HTTP_404_NOT_FOUND)

        log.status = normalize_status(new_status_raw)
        log.save(update_fields=["status"])

        return Response(
            {"message": "Cập nhật trạng thái thành công!", "item": ProgressLogSerializer(log).data},
            status=status.HTTP_200_OK,
        )
    except Exception as e:
        logger.exception("Error in update_progress_status: %s", e)
        return Response({"error": "Lỗi cập nhật trạng thái"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

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
            status=status.HTTP_201_CREATED,
        )
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(["POST"])
@permission_classes([AllowAny])
def login_view(request):
    username = request.data.get("username")
    password = request.data.get("password")
    user = authenticate(request, username=username, password=password)

    if not user:
        return Response({"detail": "Sai tên đăng nhập hoặc mật khẩu."}, status=status.HTTP_400_BAD_REQUEST)

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
    return Response({"username": None}, status=status.HTTP_200_OK)
