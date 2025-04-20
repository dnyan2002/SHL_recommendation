from rest_framework import serializers

class QuerySerializer(serializers.Serializer):
    """Serializer for recommendation query"""
    query = serializers.CharField(required=True, help_text="Job description query or URL")
    top_k = serializers.IntegerField(required=False, default=15, help_text="Number of documents to retrieve")
    temperature = serializers.FloatField(required=False, default=0, help_text="Temperature setting (0-1)")
    model = serializers.CharField(required=False, default="gpt-4.1-mini", help_text="OpenAI model to use")

class AssessmentSerializer(serializers.Serializer):
    """Serializer for assessment object"""
    url = serializers.URLField()
    adaptive_support = serializers.CharField()
    description = serializers.CharField()
    duration = serializers.IntegerField()
    remote_support = serializers.CharField()
    test_type = serializers.ListField(child=serializers.CharField())

class RecommendationSerializer(serializers.Serializer):
    """Serializer for recommendation response"""
    recommended_assessments = AssessmentSerializer(many=True)