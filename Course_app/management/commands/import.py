# import csv
# from django.core.management.base import BaseCommand
# from Course_app.models import Course

# class Command(BaseCommand):
#     help = "Import courses from CSV"

#     def clean_value(self, value):
#         """Helper function to clean CSV values."""
#         value = value.strip()
#         return None if value.lower() in {"not calibrated", "unknown", ""} else value

#     def handle(self, *args, **options):
#         csv_path = "Course_app/management/commands/Coursera.csv"  
#         skipped_rows = 0

#         with open(csv_path, "r", encoding="utf-8") as file:
#             reader = csv.DictReader(file)
#             for row_num, row in enumerate(reader, start=2):  # Start at row 2 (header is row 1)
#                 try:
#                     # Clean data for all fields
#                     course_id = self.clean_value(row.get("id", ""))  # Get ID
#                     name = self.clean_value(row.get("Course Name", ""))
#                     university = self.clean_value(row.get("University", ""))
#                     difficulty = self.clean_value(row.get("Difficulty Level", ""))
#                     url = self.clean_value(row.get("Course URL", ""))
#                     description = self.clean_value(row.get("Course Description", ""))
#                     skills = self.clean_value(row.get("Skills", ""))

#                     # Handle Course Rating separately
#                     rating_str = self.clean_value(row.get("Course Rating", ""))
#                     try:
#                         rating = float(rating_str) if rating_str else None
#                     except ValueError:
#                         rating = None  # Assign None for invalid values

#                     # Ensure mandatory fields are not missing
#                     if not name or not university or not difficulty:
#                         raise ValueError("Missing required fields")

#                     # Create or update Course object
#                     # Course.objects.update_or_create(
#                     #     id=course_id if course_id else None,  # Use id from CSV if available
#                     #     defaults={
#                     #         "id":id,
#                     #         "name": name,
#                     #         "university": university,
#                     #         "difficulty": difficulty,
#                     #         "rating": rating,
#                     #         "url": url,
#                     #         "description": description,
#                     #         "skills": skills,
#                     #     }
#                     # )
#                     Course.objects.update_or_create(
#                         id=row['Course ID'],  # Use Course ID as unique identifier
#                         defaults={
#                             'name': row['Course Name'],
#                             'university': row['University'],
#                             'difficulty': row['Difficulty Level'],
#                             'rating': rating,
#                             'url': row['Course URL'],
#                             'description': row['Course Description'],
#                             'skills': row['Skills']
#                         }
#                     )
                      


#                 except KeyError as e:
#                     skipped_rows += 1
#                     self.stdout.write(self.style.WARNING(
#                         f"Row {row_num} skipped: Missing column {str(e)}"
#                     ))

#                 except Exception as e:
#                     skipped_rows += 1
#                     self.stdout.write(self.style.WARNING(
#                         f"Row {row_num} skipped due to error: {str(e)}"
#                     ))

#         self.stdout.write(self.style.SUCCESS(
#             f"Import complete! Skipped {skipped_rows} invalid rows."
#         ))




import csv
from django.core.management.base import BaseCommand
from Course_app.models import Course

class Command(BaseCommand):
    help = "Import courses from CSV"

    def clean_value(self, value):
        """Helper function to clean CSV values."""
        value = value.strip()
        return None if value.lower() in {"not calibrated", "unknown", ""} else value

    def handle(self, *args, **options):
        csv_path = "Course_app/management/commands/Coursera.csv"  
        skipped_rows = 0

        with open(csv_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row_num, row in enumerate(reader, start=2):  # Start at row 2 (header is row 1)
                try:
                    # Clean data
                    course_id = self.clean_value(row.get("Course ID", ""))
                    if course_id and course_id.isdigit():
                        course_id = int(course_id)
                    else:
                        raise ValueError("Invalid Course ID")

                    name = self.clean_value(row.get("Course Name", ""))
                    university = self.clean_value(row.get("University", ""))
                    difficulty = self.clean_value(row.get("Difficulty Level", ""))
                    url = self.clean_value(row.get("Course URL", ""))
                    description = self.clean_value(row.get("Course Description", ""))
                    skills = self.clean_value(row.get("Skills", ""))

                    # Handle Course Rating
                    rating_str = self.clean_value(row.get("Course Rating", ""))
                    try:
                        rating = float(rating_str) if rating_str else None
                    except ValueError:
                        rating = None  

                    # Ensure mandatory fields are present
                    if not name or not university or not difficulty:
                        raise ValueError("Missing required fields: Name, University, or Difficulty")

                    # Create or update Course object
                    Course.objects.update_or_create(
                        course_id=course_id,  # Use course_id instead of id
                        defaults={
                            'name': name,
                            'university': university,
                            'difficulty': difficulty,
                            'rating': rating,
                            'url': url,
                            'description': description,
                            'skills': skills
                        }
                    )

                except KeyError as e:
                    skipped_rows += 1
                    self.stdout.write(self.style.WARNING(
                        f"Row {row_num} skipped: Missing column {str(e)}"
                    ))

                except Exception as e:
                    skipped_rows += 1
                    self.stdout.write(self.style.WARNING(
                        f"Row {row_num} skipped due to error: {str(e)}"
                    ))

        self.stdout.write(self.style.SUCCESS(
            f"Import complete! Skipped {skipped_rows} invalid rows."
        ))
