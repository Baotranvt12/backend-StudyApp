from django.contrib import admin
from .models import ProgressLog

@admin.register(ProgressLog)
class ProgressLogAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'subject', 'week', 'day_number', 'status', 'updated_at')
    list_filter = ('status', 'week', 'subject')
    search_fields = ('task_title', 'user__username')
