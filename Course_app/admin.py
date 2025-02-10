from django.contrib import admin
from .models import CustomUser, Course, UserProfile, CourseInteraction

admin.site.register(Course)
admin.site.register(UserProfile)
admin.site.register(CourseInteraction)
admin.site.register(CustomUser)