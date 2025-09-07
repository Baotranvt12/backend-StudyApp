import os
import re
import logging
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

# Setup logging
logger = logging.getLogger(__name__)

# =========================
# OpenAI (DeepInfra) client
# =========================
DEEPINFRA_API_KEY = os.environ.get("DEEPINFRA_API_KEY")

# Log để debug
logger.info(f"DEEPINFRA_API_KEY exists: {bool(DEEPINFRA_API_KEY)}")
if DEEPINFRA_API_KEY:
    logger.info(f"API Key length: {len(DEEPINFRA_API_KEY)}")
    logger.info(f"API Key first 10 chars: {DEEPINFRA_API_KEY[:10]}...")

if not DEEPINFRA_API_KEY:
    logger.error("DEEPINFRA_API_KEY environment variable is not set!")
    # Trong development, có thể dùng một key mặc định
    # Nhưng trong production PHẢI raise error
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
    Body JSON:
    {
      "class_level": "10",
      "subject": "Tin học",
      "study_time": "1 giờ",
      "goal": "Nắm vững Python cơ bản"
    }
    """
    try:
        # Log request data
        logger.info(f"generate_learning_path called by user: {request.user.username}")
        logger.info(f"Request data: {request.data}")
        
        data = request.data
        class_level = (data.get("class_level") or "").strip()
        subject = (data.get("subject") or "").strip()
        study_time = (data.get("study_time") or "").strip()
        goal = (data.get("goal") or "").strip()
        
        if not all([class_level, subject, study_time, goal]):
            logger.warning("Missing required fields")
            return Response(
                {"error": "Thiếu thông tin bắt buộc."}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Prompt
        messages = [
    {
        "role": "system",
        "content": (
            "Bạn là chuyên gia lập kế hoạch tự học, có hơn 10 năm kinh nghiệm thiết kế chương trình học tập cá nhân hoá. Trả lời HOÀN TOÀN bằng tiếng Việt."
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
        
        # Call LLM with better error handling
        logger.info("Calling DeepInfra API...")
        try:
            resp = openai.chat.completions.create(
                model="openchat/openchat_3.5",
                messages=messages,
                stream=False,
                max_tokens=2000,  # Limit response length
                temperature=0.7   # Add some creativity
            )
            logger.info("DeepInfra API call successful")
        except Exception as api_error:
            logger.error(f"DeepInfra API error: {str(api_error)}")
            logger.error(f"Error type: {type(api_error).__name__}")
            
            # Check specific error types
            error_message = str(api_error)
            if "api_key" in error_message.lower():
                return Response(
                    {"error": "Lỗi xác thực API key. Vui lòng kiểm tra cấu hình."}, 
                    status=500
                )
            elif "rate" in error_message.lower():
                return Response(
                    {"error": "Đã vượt quá giới hạn API. Vui lòng thử lại sau."}, 
                    status=429
                )
            else:
                return Response(
                    {
                        "error": "Lỗi khi gọi AI service",
                        "details": str(api_error)[:200]  # Limit error message length
                    }, 
                    status=500
                )
        
        # Parse response
        try:
            gpt_text_vi = (resp.choices[0].message.content or "").strip()
            logger.info(f"GPT response length: {len(gpt_text_vi)}")
            
            if not gpt_text_vi:
                logger.error("Empty response from GPT")
                return Response(
                    {"error": "Không nhận được phản hồi từ AI"}, 
                    status=500
                )
        except Exception as parse_error:
            logger.error(f"Error parsing GPT response: {parse_error}")
            return Response(
                {"error": "Lỗi xử lý phản hồi từ AI"}, 
                status=500
            )
        
        # === Parse "Ngày N: ..." lines ===
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
                day_num = int(re.search(r"Ngày\s+(\d{1,2})", line).group(1))
            except Exception:
                continue
            if 1 <= day_num <= 28:
                # Giữ toàn bộ dòng
                plan[day_num] = line

        logger.info(f"Parsed {len(plan)} days from GPT response")

        # Validate đủ 28 ngày
        if len(plan) != 28:
            logger.warning(f"Expected 28 days but got {len(plan)}. Fallback to default plan.")
            plan = {}
            for day in range(1, 29):
                week = (day - 1) // 7 + 1
                plan[day] = f"Ngày {day}: Học {subject} - Ôn tập/chủ đề liên quan {goal} | TỪ KHÓA TÌM KIẾM: {subject} {goal} | Bài tập tự luyện: Thực hành 15 phút | CÔNG CỤ HỖ TRỢ: Google Classroom"
       
        # Save to DB (per user & subject)
        user = request.user
        try:
            with transaction.atomic():
                # Delete old progress for this subject
                deleted_count = ProgressLog.objects.filter(
                    user=user, 
                    subject=subject
                ).delete()[0]
                logger.info(f"Deleted {deleted_count} old progress logs")
                
                # Create new progress logs
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
                
                created_logs = ProgressLog.objects.bulk_create(objs)
                logger.info(f"Created {len(created_logs)} new progress logs")
        except Exception as db_error:
            logger.error(f"Database error: {db_error}")
            return Response(
                {"error": "Lỗi lưu dữ liệu vào database"}, 
                status=500
            )
        
        # Return success response
        logs = ProgressLog.objects.filter(
            user=user, 
            subject=subject
        ).order_by("week", "day_number")
        
        return Response(
            {
                "message": "✅ Đã tạo lộ trình học!",
                "subject": subject,
                "items": ProgressLogSerializer(logs, many=True).data,
                "raw_gpt_output": gpt_text_vi[:1000],  # Limit output size
            },
            status=201,
        )
        
    except Exception as unexpected_error:
        logger.error(f"Unexpected error in generate_learning_path: {unexpected_error}")
        logger.error(f"Error type: {type(unexpected_error).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return Response(
            {
                "error": "Lỗi không xác định",
                "details": str(unexpected_error)[:200]
            },
            status=500
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
                status=400
            )
        
        try:
            log = ProgressLog.objects.get(id=log_id, user=request.user)
        except ProgressLog.DoesNotExist:
            return Response(
                {"error": "Không tìm thấy bản ghi của bạn"}, 
                status=404
            )
        
        log.status = normalize_status(new_status_raw)
        log.save(update_fields=["status"])
        
        return Response(
            {
                "message": "Cập nhật trạng thái thành công!", 
                "item": ProgressLogSerializer(log).data
            },
            status=200,
        )
    except Exception as e:
        logger.error(f"Error in update_progress_status: {e}")
        return Response({"error": "Lỗi cập nhật trạng thái"}, status=500)

# =========================================
# Auth & CSRF endpoints (không thay đổi)
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
