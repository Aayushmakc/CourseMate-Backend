from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from django.contrib.auth import authenticate, login
from rest_framework.parsers import JSONParser, FormParser, MultiPartParser
from django.db.models import Count, Avg, Q
from rest_framework.permissions import AllowAny
import pandas as pd
import numpy as np
from .models import Course, UserProfile, CourseInteraction, UserInterests
from .serializers import SignupSerializer, LoginSerializer, CourseSerializer, UserProfileSerializer
from .utils import (
    clean_and_process_data,
    preprocess_courses,
    recommend_courses_by_interests,
    send_welcome_email,
    PreprocessTexte,
    CustomTFIDFVectorizer,
    books_id_recommended,
    cosine_similarity,
    recommend_courses,
    filter_dataframe_function
)
from rest_framework_simplejwt.tokens import RefreshToken
from django.http import JsonResponse
from django.conf import settings
import json
from rest_framework.pagination import PageNumberPagination




class CustomPagination(PageNumberPagination):
    page_size = 10  
    page_size_query_param = 'page_size'
    max_page_size = 50
    page_query_param = 'page'


class SearchCourseView(APIView):
    def get(self, request):
        paginator = CustomPagination()  
        course_id = request.query_params.get('course_id')
        name = request.query_params.get('name', '').strip()
        description = request.query_params.get('description', '').strip()
        university = request.query_params.get('university', '').strip()
        difficulty_level = request.query_params.get('difficulty_level', '').strip()
        skills = request.query_params.get('skills', '').strip()
        min_rating = request.query_params.get('min_rating')
        user_id = request.query_params.get('user_id')

        
        if user_id:
            try:
                user_interests = UserInterests.objects.get(user_id=user_id)
                if not any([course_id, name, university, description, difficulty_level, skills, min_rating]):
                    difficulty_level = user_interests.difficulty_level
                    if user_interests.interests:
                        skills = user_interests.interests
            except UserInterests.DoesNotExist:
                return Response({
                    "results": {
                        "message": "User interests not found. Please provide search parameters.",
                        "recommendations": []
                    }
                }, status=status.HTTP_404_NOT_FOUND)
        elif not any([course_id, name, university, description, difficulty_level, skills, min_rating]):
            return Response({
                "results": {
                    "message": "Please provide at least one search parameter (course_id, name, description, difficulty_level, skills, or min_rating)",
                    "recommendations": []
                }
            }, status=status.HTTP_200_OK)

        try:
            queryset = Course.objects.all()

            if course_id:
                try:
                    course_id = int(course_id)
                    queryset = queryset.filter(course_id=course_id)
                except ValueError:
                    return Response({
                        "error": "Invalid course_id. Must be a number."
                    }, status=status.HTTP_400_BAD_REQUEST)

            if difficulty_level:
                queryset = queryset.filter(difficulty__iexact=difficulty_level)
            if university:
                queryset = queryset.filter(university__iexact=university)

            if min_rating:
                try:
                    min_rating = float(min_rating)
                    queryset = queryset.filter(rating__gte=min_rating)
                except ValueError:
                    pass

            if skills:
                skills_list = [skill.strip() for skill in skills.split(",")]
                for skill in skills_list:
                    queryset = queryset.filter(skills__icontains=skill)

            df = pd.DataFrame.from_records(queryset.values(
                'course_id', 'name', 'university', 'difficulty', 'rating', 'url', 'description', 'skills',
            ))

            if df.empty:
                return Response({
                    "results": {
                        "message": "No courses found matching your criteria.",
                        "recommendations": []
                    }
                }, status=status.HTTP_200_OK)

            
            df = clean_and_process_data(df)
            recommended_df = df.copy()

            if name or description:
                search_text = " ".join(filter(None, [name, description]))

               
                df["search_text_field"] = df.apply(
                    lambda x: ' '.join([
                        str(x['name']) * 3,
                        str(x['description']),
                        str(x['university']),
                        str(x['skills'])
                    ]), 
                    axis=1
                )
                df["search_text_field"] = df["search_text_field"].apply(PreprocessTexte)

               
                vectorizer = CustomTFIDFVectorizer(max_features=10000, stop_words='english')
                vectors = vectorizer.fit_transform(df["search_text_field"])

                
                recommended_indices, similarity_scores = books_id_recommended(search_text, vectorizer, vectors, number_of_recommendation=50, return_scores=True)
                
                
                valid_indices = [idx for idx, score in zip(recommended_indices, similarity_scores) if score > 0.1]
                
                if valid_indices:
                    recommended_df = df.iloc[valid_indices]
                else:
                    recommended_df = pd.DataFrame()  

            if recommended_df.empty:
                return Response({
                    "results": {
                        "message": "No courses found matching your criteria.",
                        "recommendations": []
                    }
                }, status=status.HTTP_200_OK)

            if "search_text_field" in recommended_df.columns:
                recommended_df = recommended_df.drop(columns=["search_text_field"], errors="ignore")

            recommended_df = recommended_df.replace({np.nan: None})
            recommendations = recommended_df.to_dict(orient='records')

            paginated_results = paginator.paginate_queryset(recommendations, request)
            return paginator.get_paginated_response({
                'message': f"Found {len(recommendations)} courses matching your criteria",
                'recommendations': paginated_results
            })

        except Exception as e:
            return Response({
                "error": str(e),
                "recommendations": []
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



class ContentBasedRecommenderView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        try:
            paginator = CustomPagination()
            user_id = request.query_params.get('user_id')

            if not user_id:
                return Response({"error": "user_id is required"}, status=status.HTTP_400_BAD_REQUEST)

            queryset = Course.objects.all()
            all_courses_df = pd.DataFrame.from_records(queryset.values(
                'course_id', 'name', 'university', 'difficulty', 'rating', 'url', 'description', 'skills'
            ))

            if all_courses_df.empty:
                return Response({
                    'message': 'No courses available',
                    'recommendations': []
                }, status=status.HTTP_200_OK)

          
            all_courses_df = all_courses_df.fillna({
                'name': '',
                'university': '',
                'description': '',
                'difficulty': 'Beginner',
                'rating': 0.0,
                'url': '',
                'skills': ''
            })

            
            user_interactions = CourseInteraction.objects.filter(user_id=user_id)
            viewed_courses = set(user_interactions.values_list('course_id', flat=True))
            
            print(f"DEBUG: User {user_id} interactions:", list(user_interactions.values()))  

           
            rated_courses = user_interactions.filter(
                interaction_type='rate',
                rating__gte=4  
            ).order_by('-timestamp')

            print(f"DEBUG: Found {rated_courses.count()} highly rated courses")  

            if rated_courses.exists():
                
                top_rated_courses = rated_courses.select_related('course')[:5]  
                
                print("DEBUG: Top rated courses:") 
                for course in top_rated_courses:
                    print(f"- {course.course.name} (Rating: {course.rating}, Skills: {course.course.skills})")

            
                user_interests_parts = []
                for idx, interaction in enumerate(top_rated_courses):
                    weight = 5 - idx  
                    course = interaction.course
                    if course.name:
                        user_interests_parts.extend([course.name] * weight)
                    if course.skills:
                        skills = [skill.strip() for skill in course.skills.split(',')]
                        user_interests_parts.extend(skills * weight)
                    if course.description:
                        user_interests_parts.append(course.description)

                user_interests = ' '.join(user_interests_parts)
                print("DEBUG: User Interests:", user_interests)  

                all_courses_df = clean_and_process_data(all_courses_df)
                all_courses_df = preprocess_courses(all_courses_df)

                recommended_courses = recommend_courses(user_interests, all_courses_df)
                print(f"DEBUG: Got {len(recommended_courses) if isinstance(recommended_courses, pd.DataFrame) else 0} recommended courses")  

                if isinstance(recommended_courses, pd.DataFrame) and not recommended_courses.empty:
                    filtered_recommendations = recommended_courses[~recommended_courses['course_id'].isin(viewed_courses)]
                    print(f"DEBUG: After filtering viewed courses: {len(filtered_recommendations)} courses") 
                    
                    if not filtered_recommendations.empty:
                        recent_categories = set()
                        for interaction in top_rated_courses:
                            if interaction.course.skills:
                                categories = [cat.strip().lower() for cat in interaction.course.skills.split(',')]
                                recent_categories.update(categories)
                        
                        print(f"DEBUG: Recent categories: {recent_categories}")  

                        recommendations = []
                        seen_categories = set()
                        for _, course in filtered_recommendations.iterrows():
                            course_dict = course.to_dict()
                            course_categories = set([cat.strip().lower() for cat in str(course_dict.get('skills', '')).split(',')])
                            
                            if not course_categories.isdisjoint(recent_categories):
                                recommendations.append(course_dict)
                                seen_categories.update(course_categories)
                            
                            if len(recommendations) >= 20:  
                                break
                        
                        print(f"DEBUG: Final recommendations count: {len(recommendations)}")  
                        
                        if not recommendations:
                            print("DEBUG: No diverse recommendations found, falling back to filtered recommendations")  
                            recommendations = filtered_recommendations.head(20).to_dict(orient='records')
                        
                        message = 'Recommendations based on your recent interests'
                    else:
                        print("DEBUG: All recommended courses were already viewed") 
                        return self.get_popular_recommendations(request, user_id, all_courses_df)
                else:
                    print("DEBUG: No recommendations from recommend_courses")  
                    return self.get_interest_based_recommendations(request, user_id, all_courses_df)

            else:
                print("DEBUG: No highly rated courses found")  
                return self.get_interest_based_recommendations(request, user_id, all_courses_df)

            paginated_results = paginator.paginate_queryset(recommendations, request)
            return paginator.get_paginated_response({
                'message': message,
                'recommendations': paginated_results
            })

        except Exception as e:
            print(f"ERROR: {str(e)}")  
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def get_interest_based_recommendations(self, request, user_id, all_courses_df):
        """
        Generates recommendations based on user interests when there are no highly rated courses.
        If no interests found, returns popular courses.
        """
        try:
            paginator = CustomPagination()
            user_interests = UserInterests.objects.get(user_id=user_id)
            if user_interests.interests:
                print("DEBUG: Using Profile Interests:", user_interests.interests) 
                user_interests_text = user_interests.interests
                
                all_courses_df = clean_and_process_data(all_courses_df)
                all_courses_df = preprocess_courses(all_courses_df)
                
                recommended_courses = recommend_courses(user_interests_text, all_courses_df)
                
                if isinstance(recommended_courses, pd.DataFrame) and not recommended_courses.empty:
                    recommendations = recommended_courses.to_dict(orient='records')
                    message = "Recommendations based on your interests"
                else:
                    
                    return self.get_popular_recommendations(request, user_id, all_courses_df)
            else:
                
                return self.get_popular_recommendations(request, user_id, all_courses_df)

            paginated_results = paginator.paginate_queryset(recommendations, request)
            return paginator.get_paginated_response({
                "message": message,
                "recommendations": paginated_results
            })

        except UserInterests.DoesNotExist:
            return self.get_popular_recommendations(request, user_id, all_courses_df)
        except Exception as e:
            print(f"Error in get_interest_based_recommendations: {str(e)}")  
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def get_popular_recommendations(self, request, user_id, all_courses_df):
        """
        Returns popular course recommendations based on overall ratings and interaction counts.
        """
        try:
            paginator = CustomPagination()
            popular_courses = CourseInteraction.objects.values('course_id').annotate(
                avg_rating=Avg('rating'),
                interaction_count=Count('id')
            ).filter(
                avg_rating__gte=4.0,
                interaction_count__gte=1
            ).order_by('-avg_rating', '-interaction_count')[:10]

            if not popular_courses:
                recommendations = all_courses_df.sort_values(
                    by=['rating'], 
                    ascending=False
                ).head(10).to_dict(orient='records')
            else:
                popular_course_ids = [course['course_id'] for course in popular_courses]
                recommendations = all_courses_df[
                    all_courses_df['course_id'].isin(popular_course_ids)
                ].to_dict(orient='records')

            paginated_results = paginator.paginate_queryset(recommendations, request)
            return paginator.get_paginated_response({
                "message": "Popular course recommendations for you",
                "recommendations": paginated_results
            })
        except Exception as e:
            print(f"Error in get_popular_recommendations: {str(e)}")  
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class SignupView(APIView):
    """
    API endpoint for user registration.
    """
    permission_classes = [AllowAny]
    parser_classes = (JSONParser, FormParser, MultiPartParser)
    serializer_class = SignupSerializer  

    def post(self, request):
        serializer = SignupSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            login(request, user)  
            refresh = RefreshToken.for_user(user)
            
            
            try:
                send_welcome_email(user.email, user.first_name)
            except Exception as e:
                print(f"Failed to send email: {str(e)}")
            
            return Response({
                'message': 'User created successfully',
                'user': {
                    'id': user.id,
                    'email': user.email,
                    'first_name': user.first_name,
                    'last_name': user.last_name
                },
                'tokens': {
                    'access': str(refresh.access_token),
                    'refresh': str(refresh)
                }
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
class LoginView(APIView):
    """
    API endpoint for user login.
    """
    parser_classes = (JSONParser, FormParser, MultiPartParser)
    serializer_class = LoginSerializer  

    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        if serializer.is_valid():
            user = authenticate(
                username=serializer.validated_data['email'],
                password=serializer.validated_data['password']
            )
            if user:
                login(request, user)
                refresh = RefreshToken.for_user(user)

                return Response({
                    'message': 'Login successful',
                    'user': {
                        'id': user.id,
                        'email': user.email,
                        'first_name': user.first_name,
                        'last_name': user.last_name
                    },
                    'tokens': {
                        'access': str(refresh.access_token),
                        'refresh': str(refresh)
                    }
                })
            return Response(
                {'error': 'Invalid credentials'}, 
                status=status.HTTP_401_UNAUTHORIZED
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class UserProfileView(APIView):
    """
    API endpoint for managing user profiles.
    """
    permission_classes = [AllowAny]
    
    def post(self, request):
        """Create a new user profile"""
        try:
           
            user_id = request.data.get('user_id')
            course_id = request.data.get('course_id')
            
            if not user_id or not course_id:
                return Response(
                    {"error": "Both user_id and course_id are required"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
        
            try:
                course = Course.objects.get(course_id=course_id)
            except Course.DoesNotExist:
                return Response(
                    {"error": f"Course with id {course_id} not found"},
                    status=status.HTTP_404_NOT_FOUND
                )
            
           
            existing_profile = UserProfile.objects.filter(user_id=user_id, course=course).first()
            if existing_profile:
                return Response(
                    {"error": f"Profile already exists for user {user_id} and course {course_id}"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            
            profile_data = {
                'user_id': user_id,
                'course': course,
                'course_name': request.data.get('course_name', course.name),
                'course_description': request.data.get('course_description', course.description),
                'skills': request.data.get('skills', ''),
                'difficulty_level': request.data.get('difficulty_level', 'Medium'),
                'course_rating': request.data.get('course_rating'),
                'description_keywords': request.data.get('description_keywords', '')
            }
            
            profile = UserProfile.objects.create(**profile_data)
            
            serializer = UserProfileSerializer(profile)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def get(self, request):
        """Get user profiles"""
        user_id = request.query_params.get('user_id')
        if not user_id:
            return Response(
                {"error": "user_id is required"},
                status=status.HTTP_400_BAD_REQUEST
            )
            
        profiles = UserProfile.objects.filter(user_id=user_id)
        serializer = UserProfileSerializer(profiles, many=True)
        return Response(serializer.data)

    def put(self, request):
        """Update a user profile"""
        try:
            user_id = request.data.get('user_id')
            course_id = request.data.get('course_id')
            
            if not user_id or not course_id:
                return Response(
                    {"error": "Both user_id and course_id are required"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            try:
                profile = UserProfile.objects.get(user_id=user_id, course__course_id=course_id)
            except UserProfile.DoesNotExist:
                return Response(
                    {"error": f"Profile not found for user {user_id} and course {course_id}"},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Update fields
            if 'course_name' in request.data:
                profile.course_name = request.data['course_name']
            if 'course_description' in request.data:
                profile.course_description = request.data['course_description']
            if 'skills' in request.data:
                profile.skills = request.data['skills']
            if 'difficulty_level' in request.data:
                profile.difficulty_level = request.data['difficulty_level']
            if 'course_rating' in request.data:
                profile.course_rating = request.data['course_rating']
            if 'description_keywords' in request.data:
                profile.description_keywords = request.data['description_keywords']
            
            profile.save()
            serializer = UserProfileSerializer(profile)
            return Response(serializer.data)
            
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class UserInteractionHistoryView(APIView):
    """
    API endpoint to get a user's course interaction history
    """
    permission_classes = [AllowAny]
    
    def get(self, request):
        try:
            user_id = request.query_params.get('user_id')
            if not user_id:
                return Response(
                    {"error": "user_id is required"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
           
            interactions = CourseInteraction.objects.filter(user_id=user_id).select_related('course')
            
            
            history = []
            for interaction in interactions:
                history.append({
                    'course_name': interaction.course.name,
                    'interaction_type': interaction.interaction_type,
                    'rating': interaction.rating,
                    'timestamp': interaction.timestamp,
                    'course_id': interaction.course.course_id
                })
            
            return Response({
                'user_id': user_id,
                'interaction_history': history
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class CoursePopularityView(APIView):
    """
    API endpoint to get popular courses based on user interactions.
    """
    permission_classes = [AllowAny]

    def get(self, request):
        try:
            popular_courses = Course.objects.annotate(
                view_count=Count('courseinteraction', filter=Q(courseinteraction__interaction_type='view')),
            ).order_by('-view_count')[:10]  

            
            results = [
                {
                    'course_id': course.course_id,
                    'name': course.name,
                    'university': course.university,
                    'difficulty': course.difficulty,
                    'view_count': course.view_count,
                    'description': course.description,
                    'rating': course.rating
                }
                for course in popular_courses
            ]

            return Response({'popular_courses': results}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response(
                {"error": f"Internal Server Error: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class UserInterestsView(APIView):
    """
    API endpoint for handling user interests
    """
    permission_classes = [AllowAny]

  
    topic_categories = {
        "Technology": ["Docker", "Machine Learning", "Artificial Intelligence", "Cybersecurity", "Cloud Computing", 
                      "Data Science", "Blockchain", "Programming", "Software Development"],
        "Business": ["Finance", "Marketing", "Entrepreneurship", "Management", "Business Strategy", "Leadership", 
                    "Accounting", "Project Management"],
        "Science": ["Physics", "Chemistry", "Biology", "Astronomy", "Environmental Science", "Mathematics", "Geology"],
        "Health & Medicine": ["Orthodontics", "Animal Health", "Medical Science", "Healthcare", "Mental Health", "Nutrition"],
        "Arts & Humanities": ["Screenwriting", "Graphic Design", "Music", "History", "Philosophy", "Archaeology"],
        "Social Sciences": ["Psychology", "Sociology", "Political Science", "Economics", "Education"]
    }

    def post(self, request):
        try:
            print(f"Received request data: {request.data}")  
            user_id = request.data.get('user_id')
            selected_categories = request.data.get('categories', [])
            difficulty_level = request.data.get('difficulty_level', 'Beginner')
            print(f"Extracted values - user_id: {user_id}, categories: {selected_categories}, difficulty: {difficulty_level}")  # Debug print

            if not user_id or not selected_categories:
                return Response({
                    'error': 'Both user_id and categories are required'
                }, status=status.HTTP_400_BAD_REQUEST)

            
            if not all(category in self.topic_categories.keys() for category in selected_categories):
                return Response({
                    'error': 'Invalid categories provided'
                }, status=status.HTTP_400_BAD_REQUEST)

            
            interests_str = ','.join(selected_categories)
            print(f"Attempting to save interests_str: {interests_str}") 
            try:
                obj, created = UserInterests.objects.update_or_create(
                    user_id=user_id,
                    defaults={
                        'interests': interests_str,
                        'difficulty_level': difficulty_level
                    }
                )
                print(f"Save result - created: {created}, object: {obj}") 
            except Exception as save_error:
                print(f"Error saving interests: {str(save_error)}")
                raise save_error

            return Response({
                'message': 'Interests saved successfully',
                'user_id': user_id,
                'interests': selected_categories,
                'difficulty_level': difficulty_level
            }, status=status.HTTP_200_OK)

        except Exception as e:
            print(f"Error in UserInterestsView: {str(e)}")
            import traceback
            traceback.print_exc()
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )



































class CourseDetailView(APIView):
    """
    API endpoint to get course details and automatically record view interactions
    """
    permission_classes = [AllowAny]
    
    def get(self, request, course_id):
        try:
            # Get user_id from query params or authenticated user
            user_id = request.query_params.get('user_id')
            if not user_id and request.user.is_authenticated:
                user_id = request.user.id
            
            if not user_id:
                return Response(
                    {"error": "user_id is required"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Get the course
            try:
                course = Course.objects.get(course_id=course_id)
            except Course.DoesNotExist:
                return Response(
                    {"error": f"Course with id {course_id} not found"},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Automatically record the view interaction
            CourseInteraction.objects.create(
                user_id=user_id,
                course=course,
                interaction_type='view'
            )
            
            # Get course details including interaction stats
            view_count = CourseInteraction.objects.filter(
                course=course,
                interaction_type='view'
            ).count()
            
            avg_rating = CourseInteraction.objects.filter(
                course=course,
                interaction_type='rate'
            ).aggregate(Avg('rating'))['rating__avg']
            
            # Return course details with interaction stats
            response_data = {
                'course_id': course.course_id,
                'name': course.name,
                'university': course.university,
                'difficulty': course.difficulty,
                'description': course.description,
                'skills': course.skills,
                'url': course.url,
                'stats': {
                    'view_count': view_count,
                    'average_rating': round(avg_rating, 2) if avg_rating else None
                }
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

class CourseInteractionView(APIView):
    """
    API endpoint for tracking user interactions with courses.
    Records user interactions with courses for personalized recommendations.
    """
    permission_classes = [AllowAny]
    
    def post(self, request):
        try:
            user_id = request.data.get('user_id')
            course_id = request.data.get('course_id')
            interaction_type = request.data.get('interaction_type')  # 'view', 'rate'
            rating = request.data.get('rating')  # only for 'rate' interaction
            
            if not all([user_id, course_id, interaction_type]):
                return Response(
                    {"error": "user_id, course_id, and interaction_type are required"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Get the course
            try:
                course = Course.objects.get(course_id=course_id)
            except Course.DoesNotExist:
                return Response(
                    {"error": f"Course with id {course_id} not found"},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Validate interaction type
            if interaction_type not in ['view', 'rate']:
                return Response(
                    {"error": "interaction_type must be either 'view' or 'rate'"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # For rate interactions, validate rating
            if interaction_type == 'rate' and (rating is None or not (0 <= float(rating) <= 5)):
                return Response(
                    {"error": "rating is required for rate interactions and must be between 0 and 5"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            # Record the interaction
            try:
                interaction = CourseInteraction.objects.create(
                    user_id=user_id,
                    course=course,
                    interaction_type=interaction_type,
                    rating=rating if interaction_type == 'rate' else None
                )
                
                # Update the course's average rating if this is a rate interaction
                if interaction_type == 'rate':
                    from django.db.models import Avg
                    avg_rating = CourseInteraction.objects.filter(
                        course=course,
                        interaction_type='rate',
                        rating__isnull=False
                    ).aggregate(Avg('rating'))['rating__avg']
                    
                    if avg_rating is not None:
                        course.rating = avg_rating
                        course.save()
                return Response({
                    "message": "Interaction recorded successfully",
                    "interaction": {
                        "user_id": user_id,
                        "course_id": course_id,
                        "interaction_type": interaction_type,
                        "rating": rating if interaction_type == 'rate' else None
                    }
                }, status=status.HTTP_201_CREATED)

            except Exception as e:
                print(f"Error recording interaction: {str(e)}")
                return Response(
                    {"error": "Failed to record interaction"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

        except Exception as e:
            print(f"Error in CourseInteractionView: {str(e)}")
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
