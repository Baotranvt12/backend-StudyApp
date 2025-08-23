from pathlib import Path
import os
import dj_database_url

BASE_DIR = Path(__file__).resolve().parent.parent

# ========= Security / Debug =========
SECRET_KEY = (
    os.environ.get("DJANGO_SECRET_KEY")
    or os.environ.get("SECRET_KEY")
    or "dev-secret-key-change-in-production"  # Warning comment added
)

DEBUG = os.environ.get("DJANGO_DEBUG", "false").lower() == "true"

# ========= Allowed Hosts =========
CUSTOM_DOMAIN = os.environ.get("CUSTOM_DOMAIN", "").strip()
RAILWAY_URL = os.environ.get("RAILWAY_URL") or os.environ.get("RAILWAY_PUBLIC_DOMAIN")
NETLIFY_URL = os.environ.get("NETLIFY_URL", "https://studyappmaze.netlify.app")

ALLOWED_HOSTS = ["localhost", "127.0.0.1", ".railway.app"]
if CUSTOM_DOMAIN:
    ALLOWED_HOSTS.append(CUSTOM_DOMAIN)
if RAILWAY_URL:
    ALLOWED_HOSTS.append(RAILWAY_URL)

# Security headers for proxy
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
USE_X_FORWARDED_HOST = True

# ========= Database =========
DATABASES = {
    "default": dj_database_url.config(
        default=f"sqlite:///{BASE_DIR / 'db.sqlite3'}",
        conn_max_age=600,
        ssl_require=False,  # Set to True in production if using PostgreSQL
    )
}

# ========= Static Files (WhiteNoise) =========
STATIC_URL = "/static/"
STATIC_ROOT = os.path.join(BASE_DIR, "staticfiles")

# Use compressed storage in production
if not DEBUG:
    STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"
else:
    STATICFILES_STORAGE = "django.contrib.staticfiles.storage.StaticFilesStorage"

# ========= CORS Configuration =========
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    NETLIFY_URL,
]

# Allow Netlify preview deployments
CORS_ALLOWED_ORIGIN_REGEXES = [
    r"^https://.*--studyappmaze\.netlify\.app$"
]

CORS_ALLOW_CREDENTIALS = True

# ========= CSRF Configuration =========
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

# Cookie configuration for cross-origin
SESSION_COOKIE_SAMESITE = "None"
CSRF_COOKIE_SAMESITE = "None"
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
CSRF_COOKIE_HTTPONLY = False  # Allow JavaScript to read CSRF token

# ========= Installed Apps =========
INSTALLED_APPS = [
    # Django default apps
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    
    # Third-party apps
    "corsheaders",
    "rest_framework",
    
    # Your apps
    "api",
]

# ========= Middleware =========
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "corsheaders.middleware.CorsMiddleware",  # Must be before CommonMiddleware
    "whitenoise.middleware.WhiteNoiseMiddleware",  # Serve static files
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
        "DIRS": [BASE_DIR / "templates"],  # Can be empty [] if not using templates
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

# ========= Password Validation =========
AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"
    },
]

# ========= Internationalization =========
LANGUAGE_CODE = "vi"  # Changed to Vietnamese since your app is in Vietnamese
TIME_ZONE = "Asia/Ho_Chi_Minh"  # Changed to Vietnam timezone
USE_I18N = True
USE_TZ = True

# ========= Django REST Framework =========
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.SessionAuthentication",
        "rest_framework.authentication.BasicAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.AllowAny",
    ],
    # Custom exception handler
    "EXCEPTION_HANDLER": "api.exceptions.custom_exception_handler",
}

# ========= Default Primary Key Field =========
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ========= Additional Security Settings (Production) =========
if not DEBUG:
    # HTTPS settings
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_SSL_REDIRECT = False  # Set to True if you want to force HTTPS
    
    # Security headers
    SECURE_HSTS_SECONDS = 31536000  # 1 year
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True
    SECURE_CONTENT_TYPE_NOSNIFF = True
    SECURE_BROWSER_XSS_FILTER = True
    X_FRAME_OPTIONS = "DENY"

# ========= Logging Configuration =========
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO" if not DEBUG else "DEBUG",
    },
    "loggers": {
        "django": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
    },
}
