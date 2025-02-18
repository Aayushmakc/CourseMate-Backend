from rest_framework import serializers
from django.contrib.auth.models import User
from django.contrib.auth.hashers import make_password
from rest_framework_simplejwt.tokens import RefreshToken
from .models import Course ,UserProfile

class SignupSerializer(serializers.Serializer):
    first_name = serializers.CharField(max_length=100, required=True)
    last_name = serializers.CharField(max_length=100, required=True)
    email = serializers.EmailField(required=True)
    phone_number = serializers.CharField(max_length=15, required=True)
    password = serializers.CharField(write_only=True, required=True)

    def validate_email(self, value):
        if User.objects.filter(email=value).exists():
            raise serializers.ValidationError("Email already exists")
        return value

    def validate_phone_number(self, value):
        cleaned_number = value.replace(' ', '')
        if not cleaned_number[1:].isdigit():
            raise serializers.ValidationError("Invalid phone number format")
        return cleaned_number

    def validate_password(self, value):
        if len(value) < 8:
            raise serializers.ValidationError("Password must be at least 8 characters long")
        return value

    def create(self, validated_data):
        user = User.objects.create_user(
            username=validated_data['email'],
            email=validated_data['email'],
            password=validated_data['password'],
            first_name=validated_data['first_name'],
            last_name=validated_data['last_name']
        )
        
        # Generate tokens
        refresh = RefreshToken.for_user(user)
        validated_data['tokens'] = {
            'refresh': str(refresh),
            'access': str(refresh.access_token),
        }
        
        return user

    def to_representation(self, instance):
        """Convert the user instance to JSON-serializable format"""
        data = {
            'user': {
                'id': instance.id,
                'email': instance.email,
                'first_name': instance.first_name,
                'last_name': instance.last_name
            }
        }
        # Add tokens if they were generated
        if hasattr(self, 'validated_data') and 'tokens' in self.validated_data:
            data['tokens'] = self.validated_data['tokens']
        
        return data

class LoginSerializer(serializers.Serializer):
    email = serializers.EmailField(required=True)
    password = serializers.CharField(write_only=True, required=True)

    def validate(self, data):
        user = User.objects.filter(email=data['email']).first()
        if user and user.check_password(data['password']):
            # Generate tokens
            refresh = RefreshToken.for_user(user)
            data['tokens'] = {
                'refresh': str(refresh),
                'access': str(refresh.access_token),
            }
            data['user'] = user
            return data
        raise serializers.ValidationError("Invalid credentials")
    


class CourseSerializer(serializers.ModelSerializer):
    class Meta:
        model = Course
        fields = [
            'id', 'name', 'university', 'difficulty', 
            'rating', 'url', 'description', 'skills'
        ]
        
    def validate(self, data):
        # Set default values for optional fields
        if 'difficulty' not in data:
            data['difficulty'] = None
        if 'rating' not in data:
            data['rating'] = None
        if 'url' not in data:
            data['url'] = 'http://example.com'  # Default URL
        if 'description' not in data:
            data['description'] = None
        if 'skills' not in data:
            data['skills'] = None
        return data

class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserProfile
        fields = '__all__' 