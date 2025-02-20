from django.contrib import admin
from .models import CustomUser, Course, UserProfile, CourseInteraction, UserInterests

@admin.register(UserInterests)
class UserInterestsAdmin(admin.ModelAdmin):
    list_display = ('user_id', 'interests', 'difficulty_level')
    search_fields = ('user_id', 'interests')
    list_filter = ('difficulty_level',)

admin.site.register(Course)
admin.site.register(UserProfile)
admin.site.register(CourseInteraction)
admin.site.register(CustomUser)