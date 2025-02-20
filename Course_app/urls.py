from django.urls import path
from .views import (
    SignupView, LoginView, SearchCourseView, 
    ContentBasedRecommenderView, UserProfileView,
    CourseInteractionView, UserInteractionHistoryView, CoursePopularityView,
    CourseDetailView, UserInterestsView
)
from rest_framework_simplejwt.views import TokenRefreshView

urlpatterns = [
    path('signup/', SignupView.as_view(), name='signup'),
    path('login/', LoginView.as_view(), name='login'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('search/', SearchCourseView.as_view(), name='search_courses'),
    path('recommend/', ContentBasedRecommenderView.as_view(), name='recommend_courses'),
    path('profile/', UserProfileView.as_view(), name='user-profile'),
    path('interaction/', CourseInteractionView.as_view(), name='course-interaction'),
    path('interaction/history/', UserInteractionHistoryView.as_view(), name='interaction-history'),
    path('popular-courses/', CoursePopularityView.as_view(), name='popular-courses'),
    path('courses/<int:course_id>/', CourseDetailView.as_view(), name='course-detail'),
    path('interests/', UserInterestsView.as_view(), name='user-interests'),
]
