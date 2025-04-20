from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.utils import timezone
from .serializers import QuerySerializer, RecommendationSerializer
from .recommendation_engine import recommend_assessments

class HealthCheckView(APIView):
    """Health check endpoint to verify API status"""
    
    def get(self, request, *args, **kwargs):
        """Return API status"""
        return Response({
            "status": "healthy",
            "timestamp": timezone.now().isoformat(),
            "message": "SHL Assessment Recommendation API is operational"
        })

class RecommendationView(APIView):
    """Endpoint for SHL assessment recommendations"""
    
    def post(self, request, *args, **kwargs):
        """Process job description query and return recommendations"""
        serializer = QuerySerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        # Extract parameters
        query = serializer.validated_data.get("query")
        top_k = serializer.validated_data.get("top_k", 15)
        temperature = serializer.validated_data.get("temperature", 0)
        model = serializer.validated_data.get("model", "gpt-4.1-mini")
        
        # Get recommendations
        try:
            result = recommend_assessments(
                query=query,
                top_k=top_k,
                temperature=temperature,
                model=model
            )
            
            # Check for error
            if "error" in result:
                return Response({"error": result["error"]}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
            # Validate output with serializer to ensure correct format
            recommendation_serializer = RecommendationSerializer(data=result)
            if recommendation_serializer.is_valid():
                return Response(recommendation_serializer.validated_data)
            else:
                # If the output doesn't match our expected format, return the raw output
                return Response(result)
            
        except Exception as e:
            return Response(
                {"error": f"Failed to process recommendation: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )