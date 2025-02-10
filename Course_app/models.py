from django.contrib.auth.models import AbstractUser
from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator

# from django.utils import timezone

class CustomUser(AbstractUser):
   
    groups = models.ManyToManyField(
        'auth.Group',
        related_name='customuser_groups', 
        blank=True
    )
    user_permissions = models.ManyToManyField(
        'auth.Permission',
        related_name='customuser_permissions', 
        blank=True
    )


# class Course(models.Model):
#     DIFFICULTY_CHOICES = [
#         ('Beginner', 'Beginner'),
#         ('Intermediate', 'Intermediate'),
#         ('Advanced', 'Advanced')
#     ]
#     course_id = models.IntegerField(unique=True,primary_key=True)
#     name = models.CharField(max_length=255)
#     university = models.CharField(max_length=150)
#     difficulty = models.CharField(max_length=15, choices=DIFFICULTY_CHOICES)
#     rating = models.FloatField()
#     url = models.URLField()
#     description = models.TextField()
#     skills = models.TextField()  

from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator

class Course(models.Model):
    DIFFICULTY_CHOICES = [
        ('Beginner', 'Beginner'),
        ('Intermediate', 'Intermediate'),
        ('Advanced', 'Advanced')
    ]
    
    course_id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255,null=True, blank=True)
    university = models.CharField(max_length=255, null=True, blank=True)
    difficulty = models.CharField(max_length=255, choices=DIFFICULTY_CHOICES,null=True, blank=True)
    rating = models.FloatField(null=True,default=None, validators=[MinValueValidator(0.0), MaxValueValidator(5.0)])
    url = models.URLField()
    description = models.TextField()
    skills = models.TextField()

    def __str__(self):
        return self.name


    
class UserProfile(models.Model):
    user_id = models.CharField(max_length=50)
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    course_name = models.CharField(max_length=255, default="No course name provided")
    course_description = models.TextField(default="No description available")
    skills = models.TextField(null=True, blank=True)
    difficulty_level = models.CharField(max_length=50, default="Medium")
    course_rating = models.FloatField(null=True, blank=True)
    description_keywords = models.TextField(default="No keywords provided")

    class Meta:
        unique_together = ['user_id', 'course']  

    def __str__(self):
        return f"{self.user_id}'s profile for {self.course_name}"


class CourseInteraction(models.Model):
    INTERACTION_TYPES = [
        ('view', 'View'),
        ('rate', 'Rate'),
    ]
    
    user_id = models.CharField(max_length=50)
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    interaction_type = models.CharField(max_length=10, choices=INTERACTION_TYPES)
    rating = models.FloatField(null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.user_id} - {self.course.name} - {self.interaction_type}"
