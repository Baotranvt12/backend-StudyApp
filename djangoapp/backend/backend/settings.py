from pathlib import Path
import os
import dj_database_url


BASE_DIR = Path(__file__).resolve().parent.parent

# ========= Bảo mật / Debug =========
SECRET_KEY = (
    os.environ.get("DJANGO_SECRET_KEY")
    or os.environ.get("SECRET_KEY")
    or "dev-secret"
)
DEBUG = os.environ.get("DJANGO_DEBUG", "false").lower() == "true"

CUSTOM_DOMAIN = os.environ.get("CUSTOM_DOMAIN", "").strip()
RAILWAY_URL = os.environ.get("RAILWAY_URL") or os.environ.get("RAILWAY_PUBLIC_DOMAIN")
NETLIFY_URL = os.environ.get("NETLIFY_URL", "https://studyappmaze.netlify.app")

ALLOWED_HOSTS = ["localhost", "127.0.0.1", ".railway.app"]
if CUSTOM_DOMAIN:
    ALLOWED_HOSTS.append(CUSTOM_DOMAIN)

SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
USE_X_FORWARDED_HOST = True

# --- Database ---
DATABASES = {
    "default": dj_database_url.config(
        default=f"sqlite:///{BASE_DIR/'db.sqlite3'}",
        conn_max_age=600,
        ssl_require=True,
    )
}

# --- Static / WhiteNoise ---
STATIC_URL = "/static/"
STATIC_ROOT = os.path.join(BASE_DIR, "staticfiles")
if not DEBUG:
    STATICFILES_STORAGE = "whitenoise.storage.CompressedStaticFilesStorage"

# --- CORS / CSRF ---
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    NETLIFY_URL,
]
CORS_ALLOWED_ORIGIN_REGEXES = [r"^https://.*--studyappmaze\.netlify\.app$"]
CORS_ALLOW_CREDENTIALS = True

CSRF_TRUSTED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    "https://backend-studyapp-production.up.railway.app",
    NETLIFY_URL,
]
if CUSTOM_DOMAIN:
    CSRF_TRUSTED_ORIGINS.append(f"https://{CUSTOM_DOMAIN}")
if RAILWAY_URL:
    CSRF_TRUSTED_ORIGINS.append(f"https://{RAILWAY_URL}")

SESSION_COOKIE_SAMESITE = "None"
CSRF_COOKIE_SAMESITE   = "None"
SESSION_COOKIE_SECURE  = True
CSRF_COOKIE_SECURE     = True
CSRF_COOKIE_HTTPONLY   = False

# ========= Ứng dụng =========
INSTALLED_APPS = [
    "corsheaders",
    "rest_framework",
    "api",              

    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
]

# ========= Middleware =========
MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",            
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",       
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "backend.urls"

# ========= Templates =========
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],               # nếu không dùng, có thể để []
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "backend.wsgi.application"

# ========= Database =========
# Nếu không có DATABASE_URL (trên Railway sẽ có), fallback về sqlite
DATABASES = {
    "default": dj_database_url.config(
        default=f"sqlite:///{BASE_DIR/'db.sqlite3'}",
        conn_max_age=600,
        ssl_require=False,
    )
}

# ========= Password validation =========
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# ========= I18N / TZ =========
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

# ========= Static (WhiteNoise) =========
STATIC_URL = "/static/"
STATIC_ROOT = os.path.join(BASE_DIR, "staticfiles")
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"
# ========= Django REST Framework =========
REST_FRAMEWORK = {
    
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.SessionAuthentication",
        "rest_framework.authentication.BasicAuthentication",
    ],
    
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.AllowAny",
    ],
    #Custom exception handler
    "EXCEPTION_HANDLER": "api.exceptions.custom_exception_handler",
}

# ========= Mặc định primary key =========
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
