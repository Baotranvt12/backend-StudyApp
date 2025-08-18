# from django.urls import path
# from .views import update_progress, get_progress_list, update_progress_status


# urlpatterns = [
#     path('update-progress/', update_progress, name='update_progress'),
#     path('progress/', get_progress_list, name='get_progress_list'),
#     path('progress/<int:pk>/update/', update_progress_status, name='update_progress_status'),
    
# ]
# from django.urls import path
# from .views import (
#     generate_learning_path, get_progress_list, update_progress_status,
#     register, me
# )
# from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

# from django.urls import path
# from .views import (
#     generate_learning_path, get_progress_list, update_progress_status,
#     register, login_view, me, get_csrf_token,logout_view,          # <‑‑ import view mới

# )

# urlpatterns = [
#     path('generate-learning-path/', generate_learning_path),
#     path('progress/', get_progress_list),
#     path('progress/update/', update_progress_status),
#     path('register/', register),
#     path('login/', login_view),
#     path('me/', me),
#     path('csrf/', get_csrf_token), 
#     path('logout/', logout_view), 
# ]

from django.urls import path
from .views import (
    get_csrf, register, login_view, logout_view, whoami,
    generate_learning_path, get_progress_list, update_progress_status
)

urlpatterns = [
    # Auth & session
    path("csrf/", get_csrf),
    path("register/", register),
    path("login/", login_view),
    path("logout/", logout_view),
    path("me/", whoami),

    # App APIs
    path("generate-learning-path/", generate_learning_path),
    path("progress/", get_progress_list),
    path("progress/update/", update_progress_status),
]



