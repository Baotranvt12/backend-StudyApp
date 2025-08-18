from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class ProgressLog(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Chưa hoàn thành'),
        ('done', 'Đã hoàn thành'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    subject = models.CharField(max_length=100, default="Tin học")  # NEW
    week = models.PositiveIntegerField()
    day_number = models.PositiveIntegerField(default=1)  # NEW
    task_title = models.TextField()
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.username} - Tuần {self.week} - Ngày {self.day_number}"
