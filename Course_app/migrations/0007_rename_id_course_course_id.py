# Generated by Django 5.1.4 on 2025-02-06 03:19

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('Course_app', '0006_course_userprofile'),
    ]

    operations = [
        migrations.RenameField(
            model_name='course',
            old_name='id',
            new_name='course_id',
        ),
    ]
