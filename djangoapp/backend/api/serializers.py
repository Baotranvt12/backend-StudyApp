from rest_framework import serializers
from .models import ProgressLog
from django.contrib.auth.models import User
class ProgressLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProgressLog
        fields = [
            'id',
            'user',
            'subject',      # NEW
            'week',
            'day_number',   # NEW
            'task_title',
            'status',
            'created_at',
            'updated_at',
        ]
class RegisterSerializer(serializers.ModelSerializer):
    """
    Serializer cho endpoint /api/register/
    - Kiểm tra trùng username & email.
    - Kiểm tra password + password2 khớp.
    - Cho phép bỏ qua email (không bắt buộc).
    - Định nghĩa `extra_kwargs` để làm rõ các field là optional/blank.
    """

    password = serializers.CharField(
        write_only=True,
        min_length=8,
        style={"input_type": "password"},
    )
    password2 = serializers.CharField(
        write_only=True,
        min_length=8,
        style={"input_type": "password"},
    )
    # Email không bắt buộc, nhưng nếu có thì phải hợp lệ.
    email = serializers.EmailField(required=False, allow_blank=True)

    class Meta:
        model = User
        fields = (
            "username",
            "email",
            "password",
            "password2",
            "first_name",
            "last_name",
        )
        extra_kwargs = {
            "username": {"required": True},
            "first_name": {"required": False, "allow_blank": True},
            "last_name": {"required": False, "allow_blank": True},
        }

    # -------------------------------------------------------------
    # 1️⃣ Kiểm tra username đã tồn tại?
    # -------------------------------------------------------------
    def validate_username(self, value: str):
        if User.objects.filter(username=value).exists():
            raise serializers.ValidationError("Tên người dùng đã tồn tại.")
        return value

    # -------------------------------------------------------------
    # 2️⃣ Kiểm tra email (nếu người dùng cung cấp) đã tồn tại?
    # -------------------------------------------------------------
    def validate_email(self, value: str):
        # Khi value là rỗng hoặc None thì không cần kiểm tra
        if value and User.objects.filter(email=value).exists():
            raise serializers.ValidationError("Email đã được sử dụng.")
        return value

    # -------------------------------------------------------------
    # 3️⃣ Kiểm tra password và password2 khớp nhau
    # -------------------------------------------------------------
    def validate(self, attrs):
        if attrs["password"] != attrs["password2"]:
            # Nên trả lỗi ở trường password2 để người dùng biết ô nào sai
            raise serializers.ValidationError(
                {"password2": "Mật khẩu nhập lại không khớp."}
            )
        return attrs

    # -------------------------------------------------------------
    # 4️⃣ Tạo User thực tế
    # -------------------------------------------------------------
    def create(self, validated_data):
        # Loại bỏ password2, password sẽ được set bằng set_password()
        validated_data.pop("password2")
        raw_password = validated_data.pop("password")
        user = User(**validated_data)          # username, email, first_name, last_name…
        user.set_password(raw_password)        # Mã hoá mật khẩu
        user.save()
        return user


# -----------------------------------------------------------------
# Serializer dùng để trả thông tin người dùng (endpoint /api/me/)
# -----------------------------------------------------------------
class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ("id", "username", "email", "first_name", "last_name")