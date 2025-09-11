import os
import re
import logging
import time
import random
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from django.db import transaction
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from django.views.decorators.csrf import ensure_csrf_cookie
from openai import OpenAI, RateLimitError, APIConnectionError, APIError
from .models import ProgressLog
from .serializers import (
    ProgressLogSerializer,
    RegisterSerializer,
    UserSerializer,
)

# Setup logging
logger = logging.getLogger(__name__)

# =========================
# OpenAI (DeepInfra) client với cấu hình từ test thành công
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
    # Cấu hình giống test thành công trong Colab
    openai_client = OpenAI(
        api_key=DEEPINFRA_API_KEY,
        base_url="https://api.deepinfra.com/v1/openai",
        timeout=60.0,  # Giống Colab test
    )
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    raise

# =========================
# Helpers với retry logic từ Colab test
# =========================
def normalize_status(value: str) -> str:
    """Normalize arbitrary status strings into 'pending' or 'done'."""
    if not value:
        return "pending"
    v = value.strip().lower()
    return "done" if v == "done" else "pending"

def call_with_backoff(messages, model="openchat/openchat_3.5", 
                      max_tokens=1400, temperature=0.6, max_attempts=5):
    """
    Function từ Colab test - đã hoạt động thành công
    """
    for attempt in range(1, max_attempts + 1):
        try:
            t0 = time.perf_counter()
            resp = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=60.0,  # Timeout giống Colab test
            )
            elapsed = time.perf_counter() - t0
            logger.info(f"✅ API call completed in {elapsed:.2f}s (attempt {attempt})")
            return resp
        except RateLimitError as e:
            # Model busy/quota/rate limit → exponential backoff + jitter
            sleep = min(60, 2 ** attempt) + random.uniform(0, 0.5)
            logger.warning(f"⏳ Rate limit (attempt {attempt}) → retry in {sleep:.1f}s")
            if attempt < max_attempts:
                time.sleep(sleep)
            else:
                raise
        except APIConnectionError as e:
            sleep = min(20, 2 ** attempt) + random.uniform(0, 0.5)
            logger.warning(f"🌐 Network issue (attempt {attempt}) → retry in {sleep:.1f}s")
            if attempt < max_attempts:
                time.sleep(sleep)
            else:
                raise
        except APIError as e:
            # 5xx service errors – retry 1-2 lần
            if attempt >= max_attempts:
                raise
            sleep = min(20, 2 ** attempt) + random.uniform(0, 0.5)
            logger.warning(f"🛠️ Service error (attempt {attempt}) → retry in {sleep:.1f}s")
            time.sleep(sleep)
    
    raise RuntimeError("Failed after all retry attempts")

def generate_fallback_plan(class_level, subject, study_time, goal):
    """
    Enhanced fallback plan cho từng môn học cụ thể
    """
    logger.info("Generating enhanced fallback learning plan")
    plan = {}
    
    # Chủ đề chi tiết theo môn học
    if "toán" in subject.lower():
        topics = [
            "Ôn tập kiến thức cơ bản và công thức quan trọng",
            "Học định lý mới và cách chứng minh",
            "Thực hành bài tập trắc nghiệm cơ bản", 
            "Giải bài tập tự luận dạng cơ bản",
            "Thực hành bài tập nâng cao và khó",
            "Ôn tập chuyên sâu các dạng bài hay gặp",
            "Kiểm tra và đánh giá kết quả học tập"
        ]
        tools = "GeoGebra, Photomath, Khan Academy, Wolfram Alpha"
    elif "tin học" in subject.lower() or "công nghệ" in subject.lower():
        topics = [
            "Làm quen với ngôn ngữ lập trình cơ bản",
            "Học cú pháp và cấu trúc dữ liệu", 
            "Thực hành thuật toán sắp xếp và tìm kiếm",
            "Lập trình giải quyết bài toán thực tế",
            "Học về cơ sở dữ liệu và SQL",
            "Thực hành project nhỏ và debugging",
            "Tổng hợp kiến thức và làm đồ án"
        ]
        tools = "Visual Studio Code, Scratch, Python IDLE, GitHub"
    elif "văn" in subject.lower():
        topics = [
            "Ôn tập lý thuyết văn học và tác giả",
            "Đọc hiểu và phân tích văn bản",
            "Luyện viết bài văn nghị luận",
            "Thực hành làm bài thi trắc nghiệm",
            "Viết bài văn tự luận theo đề cương",
            "Ôn tập toàn bộ chương trình và đề thi",
            "Kiểm tra và rút kinh nghiệm"
        ]
        tools = "Sách giáo khoa, Văn mẫu online, Quizlet"
    else:
        # Default cho các môn khác
        topics = [
            "Ôn tập kiến thức nền tảng cơ bản",
            "Học lý thuyết mới theo chương trình",
            "Thực hành bài tập áp dụng trực tiếp", 
            "Giải bài tập vận dụng và tổng hợp",
            "Ôn tập và làm đề thi thử",
            "Tổng hợp kiến thức toàn chương trình",
            "Đánh giá và hoàn thiện kiến thức"
        ]
        tools = "Google Classroom, YouTube, Khan Academy"
    
    for day in range(1, 29):
        if day == 28:
            task = f"Ngày {day}: ÔN TẬP & KIỂM TRA TỔNG HỢP - {goal} | TỪ KHÓA TÌM KIẾM: {subject} ôn tập tổng hợp lớp {class_level} thi cuối kỳ | Bài tập tự luyện: Làm đề thi thử hoàn chỉnh trong 120 phút | CÔNG CỤ HỖ TRỢ: {tools}, Google Forms"
        else:
            week = (day - 1) // 7 + 1
            topic_idx = (day - 1) % len(topics)
            topic = topics[topic_idx]
            
            task = f"Ngày {day}: {topic} - {subject} lớp {class_level} | TỪ KHÓA TÌM KIẾM: {subject} lớp {class_level} {topic.lower()} | Bài tập tự luyện: Thực hành {study_time} với {topic.lower()} | CÔNG CỤ HỖ TRỢ: {tools}"
        
        plan[day] = task
    
    return plan

# =========================================
# Generate learning path với timeout từ Colab test
# =========================================
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def generate_learning_path(request):
    """
    Optimized version dựa trên test thành công trong Colab
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
        
        # Prompt giống như trong Colab test thành công
        messages = [
            {
                "role": "system", 
                "content": "Bạn là chuyên gia lập kế hoạch tự học, trả lời 100% tiếng Việt."
            },
            {
                "role": "user", 
                "content": f"""
Hãy lập kế hoạch tự học 4 tuần (28 ngày) cho học sinh lớp {class_level}, nhằm cải thiện môn {subject}.
Học sinh học {study_time} mỗi ngày. Mục tiêu: {goal}.
YÊU CẦU:
- Xuất ra CHÍNH XÁC 28 dòng (Ngày 1 → Ngày 28), mỗi dòng ≤ 120 ký tự, không dòng trống.
- Nội dung theo CT GDPT 2018. Ngày 28 = ÔN TẬP & KIỂM TRA TỔNG HỢP.
- KHÔNG tiêu đề, KHÔNG markdown, KHÔNG code block.
Định dạng:
Ngày N: <nội dung> | TỪ KHÓA TÌM KIẾM: <từ khóa> | Bài tập tự luyện: <gợi ý> | CÔNG CỤ HỖ TRỢ: <ứng dụng liên quan đến môn {subject}>
Chỉ in đúng 28 dòng theo mẫu trên, không thêm gì khác.
""".strip()
            }
        ]
        
        # Gọi API với cấu hình từ Colab test
        logger.info("Calling DeepInfra API with proven configuration...")
        
        plan = {}
        ai_success = False
        api_response_time = 0
        
        try:
            # Sử dụng ThreadPoolExecutor với timeout lớn hơn
            with ThreadPoolExecutor(max_workers=1) as executor:
                start_time = time.time()
                # Tăng timeout để phù hợp với Colab test (60s + buffer)
                future = executor.submit(
                    call_with_backoff, 
                    messages, 
                    model="openchat/openchat_3.5",
                    max_tokens=1400, 
                    temperature=0.6
                )
                
                try:
                    # Timeout 90s cho toàn bộ operation (bao gồm retry)
                    resp = future.result(timeout=90)  
                    api_response_time = time.time() - start_time
                    
                    if resp:
                        logger.info(f"API call successful in {api_response_time:.2f}s")
                        
                        # Parse response giống Colab
                        text = (resp.choices[0].message.content or "").strip()
                        logger.info(f"GPT response length: {len(text)}")
                        
                        if text:
                            # Parse lines
                            lines = [ln for ln in text.splitlines() if ln.strip()]
                            valid_lines = [ln for ln in lines if re.match(r"^Ngày\s+(\d{1,2})\s*[:\-–].+", ln)]
                            
                            logger.info(f"Lines: total={len(lines)}, valid='Ngày N'={len(valid_lines)}")
                            
                            # Parse days
                            for line in valid_lines:
                                try:
                                    day_match = re.search(r"Ngày\s+(\d{1,2})", line)
                                    if day_match:
                                        day_num = int(day_match.group(1))
                                        if 1 <= day_num <= 28:
                                            plan[day_num] = line
                                except Exception as e:
                                    logger.warning(f"Error parsing line: {line[:50]}... - {e}")
                                    continue
                            
                            logger.info(f"Successfully parsed {len(plan)} days from GPT response")
                            
                            # Check success threshold
                            if len(plan) >= 20:  # Accept if we got at least 20 days
                                ai_success = True
                                # Fill missing days with fallback
                                if len(plan) < 28:
                                    logger.info(f"Filling {28 - len(plan)} missing days with fallback")
                                    fallback_plan = generate_fallback_plan(class_level, subject, study_time, goal)
                                    for day in range(1, 29):
                                        if day not in plan:
                                            plan[day] = fallback_plan[day]
                            else:
                                logger.warning(f"Only got {len(plan)} days, using full fallback")
                        
                except FutureTimeoutError:
                    logger.error("API call timed out after 90 seconds")
                    future.cancel()
                    
        except Exception as e:
            logger.error(f"Error during API call: {str(e)}")
        
        # Nếu AI không thành công, dùng fallback plan
        if not ai_success:
            logger.info(f"Using fallback plan (API response time: {api_response_time:.2f}s)")
            plan = generate_fallback_plan(class_level, subject, study_time, goal)
        
        # Save to DB
        user = request.user
        try:
            with transaction.atomic():
                deleted_count = ProgressLog.objects.filter(
                    user=user, 
                    subject=subject
                ).delete()[0]
                logger.info(f"Deleted {deleted_count} old progress logs")
                
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
            "ai_generated": ai_success,
            "response_time": f"{api_response_time:.2f}s" if api_response_time > 0 else "fallback",
        }
        
        return Response(response_data, status=201)
        
    except Exception as unexpected_error:
        logger.error(f"Unexpected error in generate_learning_path: {unexpected_error}")
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
# Các endpoints khác giữ nguyên
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

# Auth endpoints
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
