from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from django.contrib.auth import authenticate, login
from rest_framework.parsers import JSONParser, FormParser, MultiPartParser
from rest_framework_simplejwt.tokens import RefreshToken
from .serializers import SignupSerializer, LoginSerializer
from .utils import send_welcome_email
from .utils import filter_dataframe_function
from .utils import recommend_courses
from rest_framework.permissions import AllowAny
from django.http import JsonResponse
from .models import Course, UserProfile, CourseInteraction
import pandas as pd
import numpy as np
from .utils import PreprocessTexte, CustomTFIDFVectorizer, clean_and_process_data, books_id_recommended
from django.db.models import Count, Avg, Q
from django.conf import settings
from .serializers import CourseSerializer, UserProfileSerializer
from .utils import cosine_similarity
import json
from rest_framework.pagination import PageNumberPagination

# class SearchCourseView(APIView):
#     def get(self, request):
#         # Get search parameters from query
#         name = request.query_params.get('name', '').strip()
#         description = request.query_params.get('description', '').strip()
#         difficulty_level = request.query_params.get('difficulty_level', '').strip()
#         min_rating = request.query_params.get('min_rating')

#         # If no search parameters provided, return empty result
#         if not any([name, description, difficulty_level, min_rating]):
#             return Response({
#                 "message": "Please provide at least one search parameter (name, description, difficulty_level, or min_rating)",
#                 "recommendations": []
#             }, status=status.HTTP_200_OK)

#         try:
#             # Start with the base queryset
#             queryset = Course.objects.all()

#             # Apply Django ORM filters first
#             if difficulty_level:
#                 queryset = queryset.filter(difficulty__iexact=difficulty_level)
#             if min_rating:
#                 try:
#                     min_rating = float(min_rating)
#                     queryset = queryset.filter(rating__gte=min_rating)
#                 except ValueError:
#                     pass

#             # Convert filtered queryset to DataFrame
#             df = pd.DataFrame.from_records(queryset.values(
#                 'course_id', 'name', 'university', 'difficulty', 'rating', 'url', 'description'
#             ))

#             if df.empty:
#                 return Response({
#                     "message": "No courses found matching your criteria",
#                     "recommendations": []
#                 }, status=status.HTTP_200_OK)

#             # Clean and process the data
#             df = clean_and_process_data(df)
#             recommended_df = df.copy()

#             # Search by name or description using TF-IDF
#             if name or description:
#                 search_text = " ".join(filter(None, [name, description]))

#                 # Create search text field for TF-IDF
#                 df["search_text_field"] = df.apply(
#                     lambda x: ' '.join([
#                         str(x['name'] or '') * 3,
#                         str(x['description'] or ''),
#                         str(x['university'] or '')
#                     ]), axis=1
#                 )
#                 df["search_text_field"] = df["search_text_field"].apply(PreprocessTexte)

#                 # TF-IDF Vectorization
#                 vectorizer = CustomTFIDFVectorizer(max_features=10000, stop_words='english')
#                 vectors = vectorizer.fit_transform(df["search_text_field"])

#                 # Get recommended indices
#                 recommended_indices = books_id_recommended(search_text, vectorizer, vectors, number_of_recommendation=50)
#                 recommended_df = df.iloc[recommended_indices]

#             # If no results match after all filters
#             if recommended_df.empty:
#                 return Response({
#                     "message": "No courses found matching your criteria",
#                     "recommendations": []
#                 }, status=status.HTTP_200_OK)

#             # Replace NaN values with None for JSON serialization
#             recommended_df = recommended_df.replace({np.nan: None})
#             recommendations = recommended_df.to_dict(orient='records')

#             return Response({
#                 'message': f"Found {len(recommendations)} courses matching your criteria",
#                 'recommendations': recommendations
#             }, status=status.HTTP_200_OK)

#         except Exception as e:
#             return Response({
#                 "error": str(e),
#                 "recommendations": []
#             }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# class ContentBasedRecommenderView(APIView):
#     def get(self, request):
#         user_id = request.query_params.get('user_id')
#         if not user_id:
#             return Response({"error": "user_id is required"}, status=status.HTTP_400_BAD_REQUEST)

#         user_profiles = UserProfile.objects.filter(user_id=user_id)
#         if not user_profiles.exists():
#             return Response({"error": f"No profile found for user {user_id}"}, status=status.HTTP_404_NOT_FOUND)

#         # Process user profile
#         user_df = pd.DataFrame.from_records(user_profiles.values())

#         queryset = Course.objects.all()
#         df = pd.DataFrame.from_records(queryset.values(
#             'course_id', 'name', 'university', 'difficulty', 'rating', 'url', 'description'
#         ))

#         # Replace NaN values before processing
#         df['name'] = df['name'].fillna('')
#         df['university'] = df['university'].fillna('')
#         df['description'] = df['description'].fillna('')
#         df['difficulty'] = df['difficulty'].fillna(0)
#         df['rating'] = df['rating'].fillna(0.0)
#         df['url'] = df['url'].fillna('')

#         df = clean_and_process_data(df)
#         df["search_text_field"] = df.apply(
#             lambda x: ' '.join([str(x['name']) * 3, str(x['description']), str(x['university'])]), axis=1
#         )
#         df["search_text_field"] = df["search_text_field"].apply(PreprocessTexte)

#         # Handle potential None values in user profile
#         user_interests = ' '.join([
#             str(user_df['course_name'].iloc[0] or ''),
#             str(user_df['course_description'].iloc[0] or ''),
#             str(user_df['skills'].iloc[0] or '')
#         ])

#         vectorizer = CustomTFIDFVectorizer(max_features=10000, stop_words='english')
#         vectors = vectorizer.fit_transform(df["search_text_field"])
#         recommended_indices = books_id_recommended(user_interests, vectorizer, vectors, number_of_recommendation=10)

#         recommended_df = df.iloc[recommended_indices]
#         recommendations = json.loads(recommended_df.to_json(orient='records'))

#         return Response({'recommendations': recommendations}, status=status.HTTP_200_OK)


class CustomPagination(PageNumberPagination):
    page_size = 10  # Default page size
    page_size_query_param = 'page_size'
    max_page_size = 50


class SearchCourseView(APIView):
    def get(self, request):
        paginator = CustomPagination()  # Initialize paginator
        name = request.query_params.get('name', '').strip()
        description = request.query_params.get('description', '').strip()
        difficulty_level = request.query_params.get('difficulty_level', '').strip()
        min_rating = request.query_params.get('min_rating')

        if not any([name, description, difficulty_level, min_rating]):
            return Response({
                "message": "Please provide at least one search parameter (name, description, difficulty_level, or min_rating)",
                "recommendations": []
            }, status=status.HTTP_200_OK)

        try:
            queryset = Course.objects.all()

            if difficulty_level:
                queryset = queryset.filter(difficulty__iexact=difficulty_level)
            if min_rating:
                try:
                    min_rating = float(min_rating)
                    queryset = queryset.filter(rating__gte=min_rating)
                except ValueError:
                    pass

            df = pd.DataFrame.from_records(queryset.values(
                'course_id', 'name', 'university', 'difficulty', 'rating', 'url', 'description'
            ))

            if df.empty:
                return Response({
                    "message": "No courses found matching your criteria",
                    "recommendations": []
                }, status=status.HTTP_200_OK)

            df = clean_and_process_data(df)
            recommended_df = df.copy()

            if name or description:
                search_text = " ".join(filter(None, [name, description]))

                df["search_text_field"] = df.apply(
                    lambda x: ' '.join([
                        str(x['name'] or '') * 3,
                        str(x['description'] or ''),
                        str(x['university'] or '')
                    ]), axis=1
                )
                df["search_text_field"] = df["search_text_field"].apply(PreprocessTexte)

                vectorizer = CustomTFIDFVectorizer(max_features=10000, stop_words='english')
                vectors = vectorizer.fit_transform(df["search_text_field"])

                recommended_indices = books_id_recommended(search_text, vectorizer, vectors, number_of_recommendation=50)
                recommended_df = df.iloc[recommended_indices]

            if recommended_df.empty:
                return Response({
                    "message": "No courses found matching your criteria",
                    "recommendations": []
                }, status=status.HTTP_200_OK)

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
    def get(self, request):
        paginator = CustomPagination()  # Initialize paginator
        user_id = request.query_params.get('user_id')
        if not user_id:
            return Response({"error": "user_id is required"}, status=status.HTTP_400_BAD_REQUEST)

        user_profiles = UserProfile.objects.filter(user_id=user_id)
        if not user_profiles.exists():
            return Response({"error": f"No profile found for user {user_id}"}, status=status.HTTP_404_NOT_FOUND)

        user_df = pd.DataFrame.from_records(user_profiles.values())

        queryset = Course.objects.all()
        df = pd.DataFrame.from_records(queryset.values(
            'course_id', 'name', 'university', 'difficulty', 'rating', 'url', 'description'
        ))

        df['name'] = df['name'].fillna('')
        df['university'] = df['university'].fillna('')
        df['description'] = df['description'].fillna('')
        df['difficulty'] = df['difficulty'].fillna(0)
        df['rating'] = df['rating'].fillna(0.0)
        df['url'] = df['url'].fillna('')

        df = clean_and_process_data(df)
        df["search_text_field"] = df.apply(
            lambda x: ' '.join([str(x['name']) * 3, str(x['description']), str(x['university'])]), axis=1
        )
        df["search_text_field"] = df["search_text_field"].apply(PreprocessTexte)

        user_interests = ' '.join([
            str(user_df['course_name'].iloc[0] or ''),
            str(user_df['course_description'].iloc[0] or ''),
            str(user_df['skills'].iloc[0] or '')
        ])

        vectorizer = CustomTFIDFVectorizer(max_features=10000, stop_words='english')
        vectors = vectorizer.fit_transform(df["search_text_field"])
        recommended_indices = books_id_recommended(user_interests, vectorizer, vectors, number_of_recommendation=10)

        recommended_df = df.iloc[recommended_indices]
        recommendations = json.loads(recommended_df.to_json(orient='records'))

        paginated_results = paginator.paginate_queryset(recommendations, request)
        return paginator.get_paginated_response({'recommendations': paginated_results})





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
            
            # Send welcome email
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
            # Get data from request
            user_id = request.data.get('user_id')
            course_id = request.data.get('course_id')
            
            if not user_id or not course_id:
                return Response(
                    {"error": "Both user_id and course_id are required"},
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
            
            # Check if profile already exists
            existing_profile = UserProfile.objects.filter(user_id=user_id, course=course).first()
            if existing_profile:
                return Response(
                    {"error": f"Profile already exists for user {user_id} and course {course_id}"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Create user profile
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


class CourseInteractionView(APIView):
    """
    API endpoint for tracking user interactions with courses.
    This automatically creates/updates user profiles based on their interactions.
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
            
            # Create or get user profile
            # Use authenticated user's ID if available, otherwise use provided user_id
            current_user_id = request.user.id if request.user.is_authenticated else user_id
            profile = UserProfile.objects.filter(
                user_id=current_user_id,
                course=course
            ).first()
            
            if not profile:
                # Create new profile
                profile = UserProfile.objects.create(
                    user_id=current_user_id,
                    course=course,
                    course_name=course.name,
                    course_description=course.description,
                    skills=course.skills,
                    difficulty_level=course.difficulty,
                    course_rating=rating if rating is not None else course.rating
                )
            elif interaction_type == 'rate' and rating is not None:
                # Update rating for existing profile
                profile.course_rating = rating
                profile.save()
            
            # Record the interaction
            interaction = CourseInteraction.objects.create(
                user_id=current_user_id,
                course=course,
                interaction_type=interaction_type,
                rating=rating if interaction_type == 'rate' else None
            )
            
            return Response({
                'message': 'Interaction recorded successfully',
                'profile': {
                    'id': profile.id,
                    'user_id': profile.user_id,
                    'course_name': profile.course_name,
                    'skills': profile.skills,
                    'difficulty_level': profile.difficulty_level,
                    'course_rating': profile.course_rating
                },
                'interaction': {
                    'id': interaction.id,
                    'type': interaction.interaction_type,
                    'timestamp': interaction.timestamp
                }
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            print(f"Error recording interaction: {str(e)}")
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
            
            # Get user's interactions
            interactions = CourseInteraction.objects.filter(user_id=user_id).select_related('course')
            
            # Format the response
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
    API endpoint to get popular courses based on user interactions
    """
    permission_classes = [AllowAny]
    
    def get(self, request):
        try:
            # Get interaction counts and average ratings for each course
            from django.db.models import Count, Avg
            
            popular_courses = Course.objects.annotate(
                view_count=Count('courseinteraction', filter=Q(courseinteraction__interaction_type='view')),
                rating_count=Count('courseinteraction', filter=Q(courseinteraction__interaction_type='rate')),
                avg_rating=Avg('courseinteraction_rating', filter=Q(courseinteraction_interaction_type='rate'))
            ).order_by('-view_count', '-avg_rating')[:10]  # Get top 10 courses
            
            # Format the response
            results = []
            for course in popular_courses:
                results.append({
                    'course_id': course.course_id,
                    'name': course.name,
                    'university': course.university,
                    'difficulty': course.difficulty,
                    'view_count': course.view_count,
                    'rating_count': course.rating_count,
                    'average_rating': round(course.avg_rating, 2) if course.avg_rating else None
                })
            
            return Response({
                'popular_courses': results
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
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
