import os
import re
import logging
import time
import random
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from openai import OpenAI, RateLimitError, APIConnectionError, APIError

logger = logging.getLogger(__name__)

DEEPINFRA_API_KEY = os.environ.get("DEEPINFRA_API_KEY")
openai_client = OpenAI(
    api_key=DEEPINFRA_API_KEY,
    base_url="https://api.deepinfra.com/v1/openai",
    timeout=60.0,
)

# Regex “chặt” hơn: chấp nhận :, ：, -, –, — sau “Ngày N”
LINE_REGEX = re.compile(r"^Ngày\s+(\d{1,2})\s*[:：\-–—]\s*(.+)$", re.IGNORECASE)

# ---- ADD/REPLACE: helpers ----
def normalize_status(value: str) -> str:
    if not value:
        return "pending"
    v = value.strip().lower()
    return "done" if v == "done" else "pending"

def call_with_backoff(messages, model="openchat/openchat_3.5",
                      max_tokens=1400, temperature=0.6, timeout=60.0,
                      max_attempts=5, stop=None):
    """Gọi LLM với retry/backoff; có thể truyền stop."""
    for attempt in range(1, max_attempts + 1):
        try:
            t0 = time.perf_counter()
            resp = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                **({"stop": stop} if stop else {}),
            )
            logger.info("✅ API call in %.2fs (attempt %d)", time.perf_counter() - t0, attempt)
            return resp
        except RateLimitError:
            sleep = min(60, 2 ** attempt) + random.uniform(0, 0.5)
            logger.warning("⏳ 429 (attempt %d) → retry in %.1fs", attempt, sleep)
            if attempt < max_attempts:
                time.sleep(sleep)
            else:
                raise
        except APIConnectionError:
            sleep = min(20, 2 ** attempt) + random.uniform(0, 0.5)
            logger.warning("🌐 Network (attempt %d) → retry in %.1fs", attempt, sleep)
            if attempt < max_attempts:
                time.sleep(sleep)
            else:
                raise
        except APIError:
            sleep = min(20, 2 ** attempt) + random.uniform(0, 0.5)
            logger.warning("🛠️ Service error (attempt %d) → retry in %.1fs", attempt, sleep)
            if attempt < max_attempts:
                time.sleep(sleep)
            else:
                raise
    raise RuntimeError("Failed after all retry attempts")

def parse_plan_lines(text: str, plan: dict[int, str]) -> None:
    """Thêm các dòng hợp lệ vào plan (không ghi đè ngày đã có)."""
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

def make_main_messages(class_level, subject, study_time, goal):
    """Prompt siết chặt định dạng."""
    return [
        {"role": "system",
         "content": "Bạn là chuyên gia lập kế hoạch tự học. Trả lời 100% bằng tiếng Việt."},
        {"role": "user",
         "content": f"""
Hãy lập kế hoạch tự học 4 tuần (28 ngày) cho học sinh lớp {class_level}, môn {subject}, mỗi ngày {study_time}. Mục tiêu: {goal}.
YÊU CẦU:
- CHÍNH XÁC 28 dòng (Ngày 1 → Ngày 28), mỗi dòng ≤ 120 ký tự, KHÔNG dòng trống.
- Mỗi dòng BẮT ĐẦU chính xác: 'Ngày N:' (có dấu hai chấm), không ký tự nào khác phía trước.
- Nội dung theo CT GDPT 2018. Ngày 28 = ÔN TẬP & KIỂM TRA TỔNG HỢP.
- KHÔNG tiêu đề/markdown/code block/giải thích.
ĐỊNH DẠNG (mỗi dòng):
Ngày N: <nội dung> | TỪ KHÓA TÌM KIẾM: <từ khóa> | Bài tập tự luyện: <gợi ý> | CÔNG CỤ HỖ TRỢ: <ứng dụng liên quan đến môn {subject}>
Chỉ in đúng 28 dòng theo mẫu, không thêm gì khác.
""".strip()},
    ]

def make_continue_messages(missing_days: list[int], subject: str):
    """Yêu cầu IN CHÍNH XÁC các ngày còn thiếu (danh sách số ngày)."""
    days_str = ", ".join(str(d) for d in missing_days)
    count = len(missing_days)
    return [
        {"role": "system", "content": "Tiếp tục kế hoạch, giữ nguyên định dạng và yêu cầu."},
        {"role": "user",
         "content": f"""
In CHÍNH XÁC {count} dòng tương ứng các ngày: {days_str}.
Mỗi dòng BẮT ĐẦU đúng 'Ngày N:' (có dấu hai chấm), mỗi dòng ≤ 120 ký tự, KHÔNG dòng trống.
KHÔNG in lại các ngày không nằm trong danh sách trên. KHÔNG tiêu đề/markdown/giải thích.
ĐỊNH DẠNG (mỗi dòng):
Ngày N: <nội dung> | TỪ KHÓA TÌM KIẾM: <từ khóa> | Bài tập tự luyện: <gợi ý> | CÔNG CỤ HỖ TRỢ: <ứng dụng liên quan đến môn {subject}>
Chỉ in đúng {count} dòng theo mẫu trên, không thêm gì khác.
""".strip()},
    ]

def generate_fallback_plan(class_level, subject, study_time, goal):
    # (giữ nguyên logic fallback của bạn — rút gọn phần hiển thị ở đây)
    plan = {}
    tools = "Google Classroom, YouTube, Khan Academy"
    for day in range(1, 29):
        if day == 28:
            task = (f"Ngày {day}: ÔN TẬP & KIỂM TRA TỔNG HỢP - {goal} | "
                    f"TỪ KHÓA TÌM KIẾM: {subject} ôn tập tổng hợp lớp {class_level} | "
                    f"Bài tập tự luyện: Làm đề thi thử 120 phút | CÔNG CỤ HỖ TRỢ: {tools}")
        else:
            task = (f"Ngày {day}: Ôn tập/chủ đề liên quan {goal} - {subject} | "
                    f"TỪ KHÓA TÌM KIẾM: {subject} lớp {class_level} {goal} | "
                    f"Bài tập tự luyện: Thực hành {study_time} | CÔNG CỤ HỖ TRỢ: {tools}")
        plan[day] = task
    return plan

# ---- REPLACE: generate_learning_path view ----
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def generate_learning_path(request):
    try:
        logger.info("generate_learning_path called by %s", request.user.username)
        data = request.data
        class_level = (data.get("class_level") or "").strip()
        subject     = (data.get("subject") or "").strip()
        study_time  = (data.get("study_time") or "").strip()
        goal        = (data.get("goal") or "").strip()

        if not all([class_level, subject, study_time, goal]):
            return Response({"error": "Thiếu thông tin bắt buộc."}, status=400)

        # 1) Gọi lần 1 với prompt siết chặt
        messages = make_main_messages(class_level, subject, study_time, goal)
        logger.info("Calling DeepInfra (main 28 lines)...")

        plan: dict[int, str] = {}
        ai_used = False
        t0 = time.perf_counter()

        # Có thể dùng thẳng (đơn giản) — hoặc ThreadPoolExecutor nếu muốn kill khi quá hạn
        resp = call_with_backoff(messages, max_tokens=1400, temperature=0.6, timeout=55.0)
        text = (resp.choices[0].message.content or "").strip()
        logger.info("LLM len=%d, elapsed=%.2fs", len(text), time.perf_counter() - t0)

        parse_plan_lines(text, plan)
        logger.info("Parsed %d/28 days (first pass)", len(plan))

        # 2) Nếu thiếu → hỏi tiếp các ngày còn thiếu (tối đa 2 vòng)
        tries = 0
        while len(plan) < 28 and tries < 2:
            missing = sorted([d for d in range(1, 29) if d not in plan])
            logger.info("Still missing %d days: %s", len(missing), missing)
            cont_messages = make_continue_messages(missing, subject)

            # Ưu tiên tốc độ: max_tokens nhỏ, timeout ngắn
            resp2 = call_with_backoff(cont_messages, max_tokens=600, temperature=0.5, timeout=40.0)
            text2 = (resp2.choices[0].message.content or "").strip()
            parse_plan_lines(text2, plan)
            logger.info("After continue #%d → %d/28 days", tries + 1, len(plan))
            tries += 1

        if len(plan) > 0:
            ai_used = True

        # 3) Nếu vẫn thiếu → chỉ fill những ngày còn thiếu bằng fallback
        if len(plan) < 28:
            fb = generate_fallback_plan(class_level, subject, study_time, goal)
            for d in range(1, 29):
                if d not in plan:
                    plan[d] = fb[d]
            logger.info("Filled missing with fallback → %d/28 days", len(plan))

        # 4) Lưu DB
        from django.db import transaction
        from .models import ProgressLog
        from .serializers import ProgressLogSerializer

        user = request.user
        with transaction.atomic():
            ProgressLog.objects.filter(user=user, subject=subject).delete()
            objs = []
            for day in range(1, 29):
                task_text = plan[day]
                week = (day - 1) // 7 + 1
                objs.append(ProgressLog(
                    user=user, subject=subject, week=week,
                    day_number=day, task_title=task_text, status="pending"
                ))
            ProgressLog.objects.bulk_create(objs)

        logs = ProgressLog.objects.filter(user=user, subject=subject).order_by("week", "day_number")
        return Response({
            "message": "✅ Đã tạo lộ trình học!",
            "subject": subject,
            "items": ProgressLogSerializer(logs, many=True).data,
            "ai_generated": ai_used,
            "note": "Đã siết định dạng & bổ sung phần thiếu (nếu có)."
        }, status=201)

    except Exception as e:
        logger.exception("generate_learning_path failed")
        return Response({"error": "Lỗi khi tạo lộ trình", "details": str(e)[:200]}, status=500)
