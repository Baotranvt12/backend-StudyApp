import os
import re
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
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
    if os.environ.get("DJANGO_DEBUG", "false").lower() == "true":
        logger.warning("Using default API key for development only!")
        DEEPINFRA_API_KEY = "your_development_key_here"
    else:
        raise ValueError("DEEPINFRA_API_KEY environment variable is required in production")

try:
    openai = OpenAI(
        api_key=DEEPINFRA_API_KEY,
        base_url="https://api.deepinfra.com/v1/openai",
        timeout=60.0,  # Set timeout cho OpenAI client
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

def call_deepinfra_api(messages, timeout=50):
    """
    Helper function để gọi DeepInfra API với timeout
    """
    try:
        resp = openai.chat.completions.create(
            model="openchat/openchat_3.5",
            messages=messages,
            stream=False,
            max_tokens=1200,  
            temperature=0.6,
            timeout=timeout  # Timeout cho request cụ thể
        )
        return resp, None
    except Exception as e:
        logger.error(f"DeepInfra API error: {str(e)}")
        return None, e

def generate_fallback_plan(class_level, subject, study_time, goal):
    """
    Tạo kế hoạch dự phòng khi API fails
    """
    logger.info("Generating fallback learning plan")
    plan = {}
    
    # Tạo kế hoạch cơ bản 28 ngày
    topics = [
        "Giới thiệu và làm quen cơ bản",
        "Khái niệm và định nghĩa quan trọng", 
        "Thực hành cơ bản",
        "Ứng dụng thực tế",
        "Ôn tập và củng cố"
    ]
    
    for day in range(1, 29):
        if day == 28:
            task = f"Ngày {day}: ÔN TẬP & KIỂM TRA TỔNG HỢP - {goal} | TỪ KHÓA TÌM KIẾM: {subject} ôn tập tổng hợp | Bài tập tự luyện: Làm bài kiểm tra tổng hợp 60 phút | CÔNG CỤ HỖ TRỢ: Google Forms, Kahoot"
        else:
            week = (day - 1) // 7 + 1
            topic_idx = (day - 1) % len(topics)
            topic = topics[topic_idx]
            
            task = f"Ngày {day}: {topic} - {subject} liên quan đến {goal} | TỪ KHÓA TÌM KIẾM: {subject} {goal} ngày {day} | Bài tập tự luyện: Thực hành {study_time} với chủ đề {topic} | CÔNG CỤ HỖ TRỢ: Google Classroom, Khan Academy"
        
        plan[day] = task
    
    return plan

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
        
        # Call LLM với timeout và ThreadPoolExecutor
        logger.info("Calling DeepInfra API with timeout...")
        
        plan = {}
        ai_success = False
        
        try:
            # Sử dụng ThreadPoolExecutor để có thể set timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(call_deepinfra_api, messages, 45)  # 45s timeout cho API call
                
                try:
                    resp, api_error = future.result(timeout=50)  # 50s timeout cho toàn bộ operation
                    
                    if api_error:
                        raise api_error
                    
                    if resp:
                        logger.info("DeepInfra API call successful")
                        
                        # Parse response
                        gpt_text_vi = (resp.choices[0].message.content or "").strip()
                        logger.info(f"GPT response length: {len(gpt_text_vi)}")
                        
                        if gpt_text_vi:
                            # Parse "Ngày N: ..." lines
                            line_regex = re.compile(r"^Ngày\s+(\d{1,2})\s*[:\-\–].+$", re.IGNORECASE)
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
                                    plan[day_num] = line
                            
                            logger.info(f"Parsed {len(plan)} days from GPT response")
                            
                            # Check if we got all 28 days
                            if len(plan) == 28:
                                ai_success = True
                            else:
                                logger.warning(f"Expected 28 days but got {len(plan)}. Will use fallback.")
                        
                except FutureTimeoutError:
                    logger.error("API call timed out after 50 seconds")
                    future.cancel()
                    
        except Exception as e:
            logger.error(f"Error during API call: {str(e)}")
        
        # Nếu AI không thành công, dùng fallback plan
        if not ai_success:
            logger.info("Using fallback plan due to AI failure or timeout")
            plan = generate_fallback_plan(class_level, subject, study_time, goal)
        
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
        
        response_data = {
            "message": "✅ Đã tạo lộ trình học!",
            "subject": subject,
            "items": ProgressLogSerializer(logs, many=True).data,
            "ai_generated": ai_success,  # Cho biết có dùng AI hay fallback
        }
        
        # Only include raw output if AI was successful
        if ai_success and 'gpt_text_vi' in locals():
            response_data["raw_gpt_output"] = gpt_text_vi[:1000]
        
        return Response(response_data, status=201)
        
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
