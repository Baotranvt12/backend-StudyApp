from rest_framework.views import exception_handler
from rest_framework import status
from rest_framework.response import Response
from rest_framework.exceptions import NotAuthenticated, PermissionDenied

def custom_exception_handler(exc, context):
    # Gọi handler mặc định trước
    response = exception_handler(exc, context)

    if isinstance(exc, NotAuthenticated):
        return Response(
            {"error": "Bạn chưa đăng nhập. Vui lòng đăng nhập để tiếp tục."},
            status=status.HTTP_401_UNAUTHORIZED
        )

    if isinstance(exc, PermissionDenied):
        return Response(
            {"error": "Bạn không có quyền thực hiện hành động này."},
            status=status.HTTP_403_FORBIDDEN
        )

    return response