from django.core.management.base import BaseCommand
from Course_app.models import UserProfile
from django.db.models import Count

class Command(BaseCommand):
    help = 'Clean up duplicate user profiles'

    def handle(self, *args, **options):
        # Find duplicates
        duplicates = UserProfile.objects.values('user_id', 'course').annotate(
            count=Count('id')
        ).filter(count__gt=1)

        for dup in duplicates:
            # Get all profiles for this user-course combination
            profiles = UserProfile.objects.filter(
                user_id=dup['user_id'],
                course=dup['course']
            ).order_by('-id')  # Keep the most recent one
            
            # Keep the first one, delete the rest
            first_profile = profiles.first()
            profiles.exclude(id=first_profile.id).delete()
            
            self.stdout.write(
                self.style.SUCCESS(
                    f'Cleaned duplicates for user {dup["user_id"]} and course {dup["course"]}'
                )
            )

        self.stdout.write(self.style.SUCCESS('Successfully cleaned duplicate profiles'))
