# Generated by Django 5.1.4 on 2025-02-05 15:31

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('Course_app', '0002_course_userprofile'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='userprofile',
            name='course',
        ),
        migrations.DeleteModel(
            name='Course',
        ),
        migrations.DeleteModel(
            name='UserProfile',
        ),
    ]
